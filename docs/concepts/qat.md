# QAT: Quantization-Aware Training

## The Problem QAT Solves

PTQ (Post-Training Quantization) works well for INT8 but has a quality cliff at INT4 — naive quantization loses 3-8 ROUGE-L points. GPTQ/AWQ recover most of that loss through calibration-aware methods, but even then, INT4 is fragile for some models or extreme bit-widths.

QAT solves this by **learning to be robust to quantization during training**. Instead of quantizing a pre-trained model, you fine-tune while simulating quantization — the model adapts its weights to minimize quality loss under quantization.

```bash
PTQ pipeline:  trained weights → (instantly) quantize → quality loss ← too late to fix
QAT pipeline:  pre-trained base → fine-tune with quantization simulation → weights naturally aligned to quantization → minimal quality loss
```

The tradeoff:

- **PTQ**: Fast (minutes), but quality cliff at low bits
- **QAT**: Expensive (retraining), but recovers quality for INT4/INT2 where PTQ fails

## Core Idea

**QAT = simulate quantization during fine-tuning so the optimizer learns to minimize the combined loss (task loss + quantization error).**

```python
# Pseudocode
for epoch in training:
    for batch in dataloader:
        outputs = model(batch)
        
        # Within forward pass:
        # For each weight tensor W:
        W_q = quantize(W, bits=4)  # Simulate quantization
        W_dq = dequantize(W_q)      # (Usually straight-through estimator for backward)
        
        # Use quantized weights for forward, but gradients flow through
        task_loss = criterion(outputs, targets)
        
        # Optimizer sees: "if you adjust weights this way, 
        # quantization error will be lower"
        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()
```

The key: quantization is **simulated in forward** (actual quantized values), but **gradients flow through** (straight-through estimator) so the model learns to minimize quantization-induced errors.

## QAT vs PTQ vs QLoRA: When to Use Which

| Aspect                        | PTQ (GPTQ/AWQ)                | QAT                           | QLoRA                              |
|-------------------------------|-------------------------------|-------------------------------|------------------------------------|
| **When applied**              | After training                | During fine-tuning            | During training                    |
| **What it solves**            | Inference cost                | Inference cost + quality      | Training VRAM                      |
| **Precision**                 | INT8 or INT4 full model       | INT8 or INT4 full model       | NF4 base + bf16 adapters           |
| **Training cost**             | None (calibration only)       | ~40-60% of normal fine-tuning | ~30-40% of normal training         |
| **Quality at INT4**           | -0.5 to -2 ROUGE-L            | -0.1 to -0.5 ROUGE-L          | N/A (adapters are fp16)            |
| **Extreme bit-widths (INT2)** | Not practical                 | Possible (with loss)          | N/A                                |
| **Best use case**             | Good baseline, minimal effort | When PTQ quality unacceptable | Limited training GPU               |
| **Combine with others**       | Yes — baseline then tune      | No (replaces fine-tuning)     | Rarely (expensive double training) |

**When to choose**:

- **PTQ first**: 60% of projects. Fast, good enough for INT8, often acceptable for INT4 with GPTQ.
- **QAT**: When PTQ quality unacceptable. Switch to QAT on top of best baseline checkpoint.
- **QLoRA**: When training VRAM is the constraint. Can combine with PTQ afterward.

## Per-Layer Sensitivity Analysis

Not all layers degrade equally under quantization. Early and late transformer blocks are typically most sensitive; middle layers are robust.

### What Per-Layer Sensitivity Measures

For each layer, quantize it independently (while keeping other layers in fp16) and measure quality loss:

```bash
Layer 0 (attn): Quantize layer 0 only → ROUGE-L drops 0.8 pts  ← SENSITIVE
Layer 1 (attn): Quantize layer 1 only → ROUGE-L drops 0.1 pts
Layer 2 (attn): Quantize layer 2 only → ROUGE-L drops 0.05 pts
...
Layer 15 (attn): Quantize layer 15 only → ROUGE-L drops 0.3 pts  ← SENSITIVE
Layer 16 (ffn): Quantize layer 16 only → ROUGE-L drops 0.02 pts ← ROBUST
...
Layer 31 (ffn): Quantize layer 31 only → ROUGE-L drops 0.2 pts   ← SENSITIVE (near output)
```

### Sensitivity Pattern for Mistral-7B

Empirically, Mistral-7B shows this pattern (INT4, GPTQ):

| Layer Type     | Layer IDs | Avg Sensitivity        | Pattern                                        |
|----------------|-----------|------------------------|------------------------------------------------|
| Early attn     | 0-3       | High (0.5-1.2 ROUGE-L) | Processes raw token embeddings; errors cascade |
| Early-mid attn | 4-10      | Medium (0.1-0.3)       | Abstraction learned; more robust               |
| Mid-late attn  | 11-25     | Low (0.02-0.1)         | Deep abstraction; robust to rounding           |
| Late attn      | 26-31     | High (0.3-1.0)         | Concentration for output; critical             |
| All FFN        | 0-31      | Very low (<0.05)       | Feed-forward layers surprisingly robust        |

**Key insight**: Attention layers → sensitive. FFN layers → robust. Early and late blocks → sensitive.

### Measuring Sensitivity

```python
# Pseudocode for per-layer sensitivity
baseline_score = evaluate_full_precision(model, val_set)  # e.g., 42.5 ROUGE-L

sensitivities = {}
for layer_id in range(num_layers):
    # Quantize only this layer
    model_temp = copy(model)
    model_temp.layers[layer_id] = quantize_layer(model.layers[layer_id], bits=4)
    
    quantized_score = evaluate(model_temp, val_set)
    sensitivity = baseline_score - quantized_score
    sensitivities[layer_id] = sensitivity
    
# Result: dict of {layer_id: quality_loss}
# sorted: [0: 0.8, 1: 0.1, 2: 0.05, ..., 31: 0.2]
```

### Using Sensitivity for Mixed-Precision Quantization

If uniform INT4 quantization drops quality too much (e.g., 2.0 ROUGE-L loss), use mixed precision:

1. Rank layers by sensitivity
2. Keep top-K sensitive layers in fp16
3. Quantize the rest to INT4

```bash
Example for Mistral-7B with 2.0 pt quality loss:

All INT4: -2.0 ROUGE-L
├─ Layer 0 contributes: -0.8
├─ Layer 31 contributes: -0.5
├─ Layer 5 contributes: -0.3
├─ Layer 2 contributes: -0.2
└─ others: -0.2

Mixed precision (fp16 top-4 sensitive, INT4 rest):
├─ Keep layers [0, 31, 5, 2] in fp16 (don't quantize)
└─ Quantize other 28 layers to INT4
Result: -0.2 ROUGE-L loss + ~30% memory saved
```

### Memory Impact of Mixed Precision

| Strategy            | INT4 Layers | fp16 Layers | Total Memory | Savings vs fp16 |
|---------------------|-------------|-------------|--------------|-----------------|
| Full fp16           | 0           | 32          | 13.5 GB      | -               |
| Full INT4           | 32          | 0           | 3.6 GB       | 3.75x           |
| Mixed (top-4 fp16)  | 28          | 4           | ~4.5 GB      | 3.0x            |
| Mixed (top-8 fp16)  | 24          | 8           | ~5.4 GB      | 2.5x            |
| Mixed (top-16 fp16) | 16          | 16          | ~8.5 GB      | 1.6x            |

Mixed precision recovers quality without losing much memory savings — usually 2-3 layers in fp16 is the sweet spot.

## QAT Implementation Strategy

### Approach 1: Full QAT (Entire Model)

Fine-tune the full model with quantization simulation. Most expensive but best quality recovery.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.nn.utils.quantization import quantize_aware_training

model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Prepare model for QAT (insert fake-quantization nodes)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# Fine-tune
training_args = TrainingArguments(
    output_dir="qat_mistral",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    warmup_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
)

trainer.train()

# Convert to actual quantized model
torch.quantization.convert(model, inplace=True)
model.save_pretrained(output_dir)
```

**Cost**: ~40-60% of baseline fine-tuning (extra quantization simulation overhead).

### Approach 2: Layer-Selective QAT (Sensitive Layers Only)

Quantize only the most sensitive layers during fine-tuning. Balances cost vs quality.

```python
# Based on sensitivity analysis:
sensitive_layers = [0, 2, 5, 31]  # top-K from sensitivity ranking

for layer_id in range(num_layers):
    layer = model.layers[layer_id]
    
    if layer_id in sensitive_layers:
        # Insert QAT nodes only in sensitive layers
        layer.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    else:
        layer.qconfig = None  # Skip quantization

torch.quantization.prepare_qat(model, inplace=True)
# ... fine-tune as normal ...
torch.quantization.convert(model, inplace=True)
```

**Cost**: ~20-30% of baseline (only sensitive layers use QAT overhead).  
**Quality**: Often recovers 70-80% of the quality loss that full INT4 incurs.

### Approach 3: Hybrid PTQ + QAT

Use PTQ as baseline, then QAT only the layers that still degrade quality.

1. Apply PTQ (GPTQ/AWQ) to the fine-tuned model
2. Measure per-layer quality loss
3. QAT only layers that exceed quality threshold
4. Keep other layers PTQ-quantized

**Cost**: Medium (retraining but only for few layers).  
**Quality**: Usually best (combines calibration + learning).

## QAT Training Dynamics

### Why QAT Works

During QAT fine-tuning, the optimizer sees:

- **Forward**: Quantized weights (actual INT4 values) → realistic quantization error
- **Backward**: Gradients flow through (typically straight-through estimator) → gradients tell weights "shift to reduce quantization error"

Over epochs:

- Early epochs: Weights adapt to minimize task loss (normal fine-tuning)
- Mid epochs: Optimizer learns which weights matter most under quantization
- Late epochs: Weights converge to values that maintain task performance even after quantization

### Comparison: QAT vs Post-Hoc Quantization

```bash
PTQ (GPTQ):
Epoch 0 (trained on fp16):
   weights = [1.234, 5.678, -2.345, ...]
   quantize → [1, 6, -2, ...]
   error = 0.234 + 0.322 + 0.345 = 0.901  ← high error

QAT (trained on INT4):
Epoch 0: weights ≈ [1.234, 5.678, -2.345, ...]
   quantize → [1, 6, -2, ...]
   error ≈ 0.901 (same)
   
Epoch 1: Optimizer shifts weights to quantization-friendly values
   weights ≈ [1.000, 6.000, -2.000, ...]
   quantize → [1, 6, -2, ...]
   error ≈ 0.000  ← low error

Epoch 20 (converged):
   weights = [0.998, 6.002, -1.998, ...]
   Natural alignment to INT4 boundaries → minimal quantization error
```

## Quality vs Training Cost Tradeoff

```bash
Quality recovery (ROUGE-L) vs Training Cost for Mistral-7B to INT4

        Quality
        │
  100%  │ ████████████ fp16 baseline
        │ ████████████ QLoRA baseline (frozen base, only adapters)
   99%  │ ███████████████ Full QAT (40-60% extra cost)
   98%  │ ███████████ Layer-selective QAT (20-30% extra cost)
   97%  │ ██████████ Hybrid PTQ + QAT (10-20% extra cost)
   96%  │ ████████ PTQ INT4 (GPTQ/AWQ, calibration only)
   94%  │ ████ PTQ INT4 naive (round-to-nearest)
        │
        └──────────────────────────
          0%   20%   40%   60%   100%
          Training Cost
```

## QAT in the Project Pipeline

### Scenario 1: PTQ Sufficient (Most Common)

```bash
Day 1-6: Fine-tune with LoRA/QLoRA → best checkpoint (fp16)
Day 7: PTQ (GPTQ/AWQ) → INT4 model
Day 8: Evaluate
   ├─ If ROUGE-L loss < 1 pt: STOP, use PTQ model
   └─ If ROUGE-L loss > 1 pt: proceed to QAT
```

### Scenario 2: Quality Unacceptable, Use QAT

```bash
Day 1-6: Fine-tune with LoRA → best checkpoint (fp16)
Day 7: Try PTQ → quality loss 2.5 pts (unacceptable)
Day 8-9: Layer sensitivity analysis
   Result: Layers [0, 5, 31] are sensitive
Day 10-12: QAT with layer-selective quantization
   Train 3 epochs with only layers [0, 5, 31] quantized
   Cost: ~30% of baseline fine-tuning
   Quality: -0.4 ROUGE-L (acceptable)
Day 13: Deploy INT4 model
```

### Scenario 3: Mixed Precision (Balanced)

```bash
Day 1-6: Fine-tune with LoRA → checkpoint
Day 7: PTQ + sensitivity analysis
   Ranking: [0: -0.8, 31: -0.5, 5: -0.3, 2: -0.2, ...]
Day 8: Mixed-precision quantization
   ├─ Keep layers [0, 31, 5, 2] in fp16
   ├─ Quantize layers [1, 3, 4, 6-30] to INT4
   └─ Result: 3.0x memory reduction, -0.1 ROUGE-L loss
Day 9: Deploy mixed-precision model
   Memory: 4.5 GB, Speed: 1.3x, Quality: ~98% baseline
```

## Straight-Through Estimator (STE)

QAT relies on the **straight-through estimator (STE)** for backpropagation through the quantization function:

```bash
Forward: x_q = quantize(x)  # {0, 1, 2, ..., 15} for INT4
Backward: ∂L/∂x = ∂L/∂x_q   # Gradients flow "straight through" the quantization
```

Mathematically, `quantize()` has no gradient (it's a discrete function), but STE pretends it has gradient 1:

$$\frac{\partial \text{quantize}(x)}{\partial x} = \begin{cases} 1 & \text{if } x \in [x_{\min}, x_{\max}] \\ 0 & \text{otherwise} \end{cases}$$

This is **an approximation** but it works: the optimizer adjusts weights to stay in quantization-friendly ranges (naturally aligned to INT4 levels).

## Debugging QAT

### Issue: QAT Doesn't Improve Over PTQ

**Cause**: Quantization simulation isn't actually activated.  
**Check**: Print model after `prepare_qat()` — should see `QuantStub` and `DeQuantStub` nodes.

### Issue: Training is Too Slow

**Cause**: Quantization simulation adds 20-40% overhead; combined with normal training is expensive.

**Solution**:

- Use layer-selective QAT (quantize only sensitive layers)
- Reduce training epochs (1-2 epochs is often sufficient)
- Use smaller learning rate with QAT (changes are delicate)

### Issue: Quality Doesn't Recover

**Cause**: Quantization simulation alone isn't enough; may need more varied calibration or different quantization scheme.

**Solution**:

- Increase training epochs (up to 5-10)
- Lower learning rate (changes should be gradual)
- Try SmoothQuant + QAT (both quantize weights AND activations)

## QAT vs TensorRT INT8

- **QAT**: Purely quantization-aware, applies to the model itself
- **TensorRT INT8**: Engine-level INT8 optimization, includes calibration + kernel fusion
- **Can combine**: QAT model → TensorRT → further optimization (rarely needed, TensorRT INT8 is very good)

QAT is primarily useful for weight quantization; TensorRT adds activation quantization and kernel-level optimization.

## QAT in Our Repo

Our `qat_config.yaml` specifies:

```yaml
qat:
  base_model: "assets/models/Mistral-7B-v0.3"
  bits: 4  # INT4
  num_epochs: 3
  learning_rate: 1e-5
  warmup_steps: 100
  quantize_layers: "all"  # Options: "all", "sensitive", "attention"
```

Run with:

```bash
python src/finetuning/quantization/qat_pipeline.py \
    --base_model assets/models/Mistral-7B-v0.3 \
    --train_data assets/datasets/cnn_dailymail \
    --output_dir outputs/qat_mistral_int4 \
    --bits 4
```

This will:

1. Load the base Mistral model
2. Insert quantization-aware training nodes
3. Fine-tune on CNN/DailyMail for 3 epochs (with quantization simulation)
4. Convert to actual INT4 model
5. Evaluate ROUGE-L vs baseline

## Summary Table: PTQ vs QAT

| Criterion                 | PTQ (GPTQ/AWQ)        | QAT (Full)               | QAT (Layer-Selective)             |
|---------------------------|-----------------------|--------------------------|-----------------------------------|
| Training cost             | Negligible            | 40-60% extra             | 20-30% extra                      |
| Quality at INT4           | -0.5 to -2 pt         | -0.1 to -0.5 pt          | -0.3 to -0.8 pt                   |
| Memory (INT4)             | 3.6 GB                | 3.6 GB                   | 4.5 GB (mixed)                    |
| Implementation complexity | Low                   | Medium                   | Medium-High                       |
| When to use               | Baseline, good enough | PTQ quality unacceptable | Quality critical + budget limited |
| Typical choice            | 60-70% of projects    | 20-30% of projects       | 10-20% of projects                |
