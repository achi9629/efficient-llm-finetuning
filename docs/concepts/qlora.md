# QLoRA: Quantized Low-Rank Adaptation

## The Problem QLoRA Solves

LoRA freezes the base model and trains small adapter matrices, but the frozen base weights still sit in GPU memory in fp16/bf16. For Mistral-7B:

- Base model in fp16: ~13.5 GB VRAM just for weights
- Optimizer states + activations + LoRA params push training to ~56 GB peak
- This requires an 80 GB A100 — most GPUs can't fit it

QLoRA compresses the base model to 4-bit at load time, cutting weight memory by ~4x while keeping LoRA adapters in bf16 for training.

## Core Idea

QLoRA = **4-bit quantized base model** + **bf16 LoRA adapters**

```bash
Standard LoRA:   W_frozen (fp16)  + ΔW = B·A (fp16)     → ~56 GB training
QLoRA:           W_frozen (NF4)   + ΔW = B·A (bf16)     → ~20-25 GB training
```

The key insight: base weights are frozen anyway, so compressing them to 4-bit loses nothing during training — gradients only flow through the LoRA adapters which remain in full precision.

## Three Innovations from the QLoRA Paper

### 1. NormalFloat 4-bit (NF4) Quantization

Pre-trained model weights approximately follow a normal distribution. NF4 is an information-theoretically optimal 4-bit data type for normally distributed data — it spaces quantization levels unevenly to match the bell curve, putting more levels near the center where most values cluster.

```bash
Regular INT4:  uniformly spaced levels → wastes precision on rare outlier regions
NF4:           levels spaced by quantiles of N(0,1) → optimal for normal distributions
```

### 2. Double Quantization

Quantization itself requires storing quantization constants (scale and zero-point) for each block of weights (typically 64 weights per block). These constants are stored in fp32 by default, adding ~0.5 GB for a 7B model.

Double quantization quantizes these constants themselves to 8-bit:

```bash
First quantization:   fp16 weights → NF4 (4-bit) + fp32 constants
Double quantization:  fp32 constants → int8 constants

Memory saving: ~0.4 GB for Mistral-7B (from fp32 → int8 on constants)
```

### 3. Paged Optimizers

Uses NVIDIA unified memory to page optimizer states to CPU when GPU memory is exhausted, preventing OOM errors during gradient spikes. Handled automatically by bitsandbytes.

## How It Works Step-by-Step

### 1. Load model in 4-bit

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

Weights are quantized to NF4 during loading. No separate calibration step needed — NF4 quantile levels are precomputed from N(0,1).

### 2. Prepare for k-bit training

```python
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
```

This does two things:

- Freezes all base model parameters
- Enables gradient checkpointing so backward passes work correctly through quantized layers

### 3. Inject LoRA adapters

```python
model = inject_lora(model, target_modules, r=16, alpha=32, dropout=0.05)
```

Same scratch LoRA from `lora_layer.py`. The A and B matrices are created in bf16. Only these ~12.6M parameters are trainable.

### 4. Forward pass: dequantize on the fly

During forward pass, 4-bit weights are dequantized to bf16 for matrix multiplication, then discarded. This happens per-layer, so only one layer's dequantized weights exist in memory at a time.

```bash
Forward:  NF4 weights → dequantize to bf16 → matmul with input → discard bf16 copy
Backward: gradients flow through LoRA A,B only (base weights frozen)
```

### 5. Train and save

Training uses HF Trainer as usual. Only LoRA adapter weights are saved — the 4-bit base model is not modified.

## Memory Breakdown: LoRA vs QLoRA

| Component                | LoRA (fp16 base) | QLoRA (NF4 base)       |
|--------------------------|------------------|------------------------|
| Base model weights       | ~13.5 GB         | ~3.5 GB                |
| Quantization constants   | 0                | ~0.1 GB (double quant) |
| LoRA A,B params          | ~50 MB           | ~50 MB                 |
| Optimizer states (AdamW) | ~100 MB          | ~100 MB                |
| Activations + gradients  | ~40 GB           | ~18 GB                 |
| **Peak VRAM**            | **~56 GB**       | **~20-25 GB**          |

The activation memory is also lower because gradient checkpointing (enabled by `prepare_model_for_kbit_training`) trades compute for memory by recomputing activations during backward.

## BitsAndBytesConfig Parameters

| Parameter                   | Value      | Why                                                                     |
|-----------------------------|------------|-------------------------------------------------------------------------|
| `load_in_4bit`              | `True`     | Load weights as 4-bit instead of fp16                                   |
| `bnb_4bit_quant_type`       | `"nf4"`    | NF4 is optimal for normally distributed weights                         |
| `bnb_4bit_compute_dtype`    | `bfloat16` | Dequantize to bf16 for matmul (better than fp16 for training stability) |
| `bnb_4bit_use_double_quant` | `True`     | Quantize the quantization constants too (~0.4 GB saving)                |

## Why Not Just Use INT4?

INT4 uses uniformly spaced quantization levels. Pre-trained model weights are approximately normally distributed — most values are clustered near zero, with few outliers. Uniform levels waste 2 of 16 levels on the tails where almost no values exist. NF4 places levels at quantiles of the normal distribution, so each level represents an equal probability mass. This is information-theoretically optimal for N(0,σ) data.

## QLoRA vs LoRA Quality Tradeoff

The quality loss from 4-bit quantization is typically within noise — the QLoRA paper shows <0.5 ROUGE difference on most tasks. Our project will measure this directly:

- LoRA (fp16 base): serves as quality ceiling
- QLoRA (NF4 base): should match within measurement noise
- Comparison in `outputs/reports/day4_comparison.csv` (extended on Day 6)

## Inference After QLoRA Training

Two options for inference with QLoRA-trained adapters:

### Option A: Load base in 4-bit + apply adapters (memory efficient)

```python
model = load_4bit_model(...)
model = load_lora_weights(model, "lora_adapter.pt", target_modules)
# Run inference in 4-bit with LoRA applied
```

### Option B: Load base in fp16 + apply adapters + merge (higher quality)

```python
model = load_fp16_model(...)
model = load_lora_weights(model, "lora_adapter.pt", target_modules)
model = merge_lora(model)
# Run inference on merged fp16 model — no quantization at inference time
```

Option B is preferred for evaluation since it isolates the quality impact of training with a 4-bit base from inference-time quantization.

## Our Configuration

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Same LoRA hyperparameters as the fp16 LoRA run — identical r, alpha, dropout, targets — so the only variable is the 4-bit base model.

## References

- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023), arXiv:2305.14314
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)
- bitsandbytes library: https://github.com/TimDettmers/bitsandbytes
