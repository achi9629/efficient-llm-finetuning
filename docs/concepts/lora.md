# LoRA: Low-Rank Adaptation of Large Language Models

## The Problem LoRA Solves

Full fine-tuning updates **all** parameters in a model. For Mistral-7B that's 7.25 billion parameters:

- Training requires storing model weights + optimizer states + gradients = ~3x model size in VRAM
- FP16 model = 13.5 GB → full fine-tuning needs ~42 GB+ VRAM
- Every fine-tuned variant is a full 13.5 GB checkpoint on disk

This is impractical for most hardware and wasteful when you only need task-specific adaptation.

## Core Idea

**Insight**: When you fine-tune a large model, the weight updates have low intrinsic rank. You don't need to update all 7B parameters — the meaningful changes live in a much smaller subspace.

Instead of updating the full weight matrix W (d x k), LoRA freezes W and adds a low-rank decomposition:

    W' = W + delta_W = W + BA

Where:

- B is (d x r)  — down-projection
- A is (r x k)  — up-projection
- r << min(d, k) is the **rank** (typically 8, 16, or 32)

For a layer with d = 4096, k = 4096:

- Full update: 4096 x 4096 = 16.7M parameters
- LoRA with r = 16: (4096 x 16) + (16 x 4096) = 131K parameters → **128x fewer**

## How It Works Step-by-Step

### 1. Freeze the base model

All original weights are frozen. No gradients flow through them during training.

### 2. Inject adapter matrices

For each target layer (typically attention projections), add two small matrices A and B:

```bash
Input x
|
|---> [Frozen W] ---> Wx
|
\---> [A] ---> [B] ---> BAx  (LoRA path)

Output = Wx + (alpha/r) * BAx
```

### 3. Scale with alpha

The LoRA output is scaled by alpha/r where:

- alpha (lora_alpha) controls the magnitude of the adaptation
- r is the rank
- The ratio alpha/r acts as a learning rate multiplier for the LoRA path

### 4. Train only A and B

Backpropagation only updates A and B. The optimizer only tracks states for these small matrices.

### 5. Merge at inference

After training, compute W' = W + (alpha/r) * BA and replace the original weights. Zero additional latency at inference.

## Which Layers to Target

In a transformer, the attention mechanism has four weight matrices per layer:

| Matrix | Purpose           | Size (Mistral-7B) |
|--------|-------------------|-------------------|
| q_proj | Query projection  | 4096 x 4096       |
| k_proj | Key projection    | 4096 x 1024 (GQA) |
| v_proj | Value projection  | 4096 x 1024 (GQA) |
| o_proj | Output projection | 4096 x 4096       |

**Common choices:**

- q_proj, v_proj — minimum viable LoRA (used in original paper)
- q_proj, k_proj, v_proj, o_proj — what we use, covers all attention weights
- Adding gate_proj, up_proj, down_proj (MLP layers) — more parameters, sometimes better

Our config targets ["q_proj", "k_proj", "v_proj", "o_proj"] — good balance of capacity vs parameter count.

## Hyperparameters Explained

### Rank (r)

- Controls capacity of the adaptation
- Higher rank = more parameters = more expressive but slower and more VRAM
- r=8: very lightweight, good for simple tasks
- r=16: our choice, standard for most tasks
- r=64: heavy, only needed for very complex adaptations

### Alpha (lora_alpha)

- Scaling factor for the LoRA update
- Effective scaling = alpha / r
- Common recipe: set alpha = 2 * r (so scaling = 2.0)
- Our config: alpha=32, r=16 → scaling = 2.0

### Dropout (lora_dropout)

- Applied to the LoRA path during training for regularization
- 0.05 is standard, prevents overfitting on small datasets
- Set to 0.0 if your dataset is large (>100k samples)

## Parameter Count for Our Setup

Mistral-7B has 32 transformer layers. Each layer has 4 target modules.

With r=16 and average dimension ~3072:

- ~32 layers x 4 modules x 2 x 16 x 3072 = **~12.6M trainable parameters**
- That is **0.17% of 7.25B** total parameters
- Adapter checkpoint size: ~50 MB (vs 13.5 GB for full model)

## Why It Works

1. **Pre-trained weights encode general knowledge** — you don't need to change them
2. **Task-specific adaptation is low-dimensional** — a rank-16 update captures most of what the model needs to learn for summarization
3. **Attention layers are where task routing happens** — adapting Q/K/V projections steers the model's "focus" toward task-relevant patterns

## LoRA vs Full Fine-Tuning

| Aspect            | Full Fine-Tuning | LoRA (r=16)           |
|-------------------|------------------|-----------------------|
| Trainable params  | 7.25B (100%)     | ~12.6M (0.17%)        |
| Training VRAM     | ~42 GB           | ~16-18 GB             |
| Checkpoint size   | 13.5 GB          | ~50 MB                |
| Training speed    | 1x               | ~1.5-2x faster        |
| Quality           | Best possible    | Within 1-2% typically |
| Inference latency | Same             | Same (after merge)    |

## LoRA in Code (PEFT Library)

```
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 12,582,912 || all params: 7,261,954,048 || trainable%: 0.1733
```

After training:
```
# Save just the adapter (~50 MB)
model.save_pretrained("outputs/checkpoints/lora")

# Load later
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "outputs/checkpoints/lora")

# Merge into base model (for deployment — zero overhead inference)
model = model.merge_and_unload()
```

## Alpha and Scaling — Why They Matter

### What scaling does

The forward pass computes:

    output = W·x + (alpha/r) · B·A·x

The `(alpha/r)` term controls **how much the LoRA path contributes** relative to the frozen base model.

### Example with real numbers

Say a frozen layer outputs `W·x = [10.0, 20.0, 30.0]` for some input.
After training, say the LoRA path `B·A·x` gives `[0.5, -0.3, 0.8]`.

| alpha | r  | scaling | LoRA contribution       | Final output           |
|-------|----|---------|-------------------------|------------------------|
| 8     | 16 | 0.5     | [0.25, -0.15, 0.4]      | [10.25, 19.85, 30.4]   |
| 16    | 16 | 1.0     | [0.5, -0.3, 0.8]        | [10.5, 19.7, 30.8]     |
| 32    | 16 | 2.0     | [1.0, -0.6, 1.6]        | [11.0, 19.4, 31.6]     |
| 64    | 16 | 4.0     | [2.0, -1.2, 3.2]        | [12.0, 18.8, 33.2]     |

### Effect on training

- **Scaling too low** (0.5): LoRA barely changes the output. Model learns very slowly or not at all.
- **Scaling too high** (4.0): LoRA changes dominate. The model forgets pretrained knowledge. Training becomes unstable.
- **Scaling = 2.0** (our config): Sweet spot. LoRA nudges the output enough to adapt, pretrained knowledge stays intact.

### Why alpha and r are separate

They serve different purposes:

- **r controls capacity**: increasing r from 16 to 32 doubles trainable parameters, lets LoRA learn more complex adaptations.
- **alpha controls magnitude**: increasing alpha from 32 to 64 keeps the same parameters but each update has 2x the effect.

This gives two tuning levers:

- **Underfitting** (loss not decreasing): raise r (more capacity) or raise alpha (bigger steps)
- **Overfitting** (val loss going up): lower alpha or increase dropout

### Common recipe

`alpha = 2 * r` (scaling = 2.0) works for ~90% of cases. Our config uses alpha=32, r=16.


## Inject → Train → Merge Lifecycle

LoRA has three distinct phases at the code level. Understanding when and why each happens is critical for implementing it from scratch.

### Phase 1: Inject (`inject_lora`)

**When**: Before training begins, after loading the frozen base model.

**What happens**:
1. Walk the model's module tree with `named_modules()`
2. Find every `nn.Linear` whose name matches a target (e.g. `q_proj`, `v_proj`)
3. Replace each with a `LoRALinear` wrapper that holds:
   - The original frozen `nn.Linear` (weights untouched, `requires_grad=False`)
   - Two new trainable parameters: `lora_A` (r × in) and `lora_B` (out × r)
4. Set the replacement on the parent module via `setattr`

**After injection, the forward pass becomes**:
```
output = original_linear(x) + dropout(x) @ A.T @ B.T * (alpha/r)
```

The base model contribution flows through unchanged. The LoRA path is additive.

**Why collect targets first**: We iterate `named_modules()` to find targets, then mutate in a second pass. Mutating during iteration would skip or double-visit modules because the module tree changes mid-walk.

### Phase 2: Train

**When**: After inject, during the training loop.

**What happens**:
- Only `lora_A` and `lora_B` parameters have `requires_grad=True`
- The optimizer only tracks these (~12.6M params vs 7.25B)
- Gradients flow through: loss → output → `B @ A @ dropout(x)` → update A and B
- The frozen `W` participates in the forward pass but never receives gradient updates

### Phase 3: Merge (`merge_lora`)

**When**: After training is complete, before saving for deployment or running inference.

**What happens**:
1. Walk the model again, find every `LoRALinear` module
2. Compute the merged weight: $W' = W + \frac{\alpha}{r} \cdot B \cdot A$
3. Create a fresh `nn.Linear` with $W'$ as its weight (and original bias if any)
4. Replace the `LoRALinear` with this plain `nn.Linear` via `setattr`

**After merge**:
- The model is a standard transformer again — no `LoRALinear` wrappers
- Forward pass is just `output = W' @ x + bias` — zero overhead
- The LoRA knowledge is baked into the weights permanently

### Why merge is necessary

| Scenario | Without merge | With merge |
|----------|---------------|------------|
| Inference speed | Extra matmul per layer (A and B) | Same as base model |
| ONNX/TensorRT export | Custom module breaks exporters | Standard `nn.Linear` exports cleanly |
| Checkpoint size | Must save base + adapter separately | Single merged checkpoint |
| Serving complexity | Need adapter loading logic | Load like any other model |

### Full lifecycle in code

```python
# 1. Load frozen base
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")

# 2. Inject LoRA
model = inject_lora(model, ["q_proj","k_proj","v_proj","o_proj"], r=16, alpha=32, dropout=0.05)

# 3. Train (only A and B update)
trainer.train()

# 4. Save adapter weights (just A and B — ~50 MB)
save_lora_weights(model, "outputs/checkpoints/lora_scratch")

# 5. Merge for deployment
model = merge_lora(model)

# 6. Now model is a plain transformer — export to ONNX, TensorRT, etc.
model.save_pretrained("outputs/checkpoints/lora_merged")
```

## References

- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) — https://arxiv.org/abs/2106.09685
- PEFT Library — https://github.com/huggingface/peft
