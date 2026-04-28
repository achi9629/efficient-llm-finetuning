# PTQ: Post-Training Quantization

## The Problem PTQ Solves

After fine-tuning, your model weights are in fp16/bf16. For Mistral-7B that means:

- Model weights: ~13.5 GB VRAM just to load
- Each concurrent request adds KV cache memory on top
- Throughput is bottlenecked by memory bandwidth (weight loading dominates over compute for autoregressive generation)

PTQ compresses weights **after training is done** — no retraining, no gradient computation, no GPU hours. You take a trained checkpoint and make it smaller and faster for serving.

## Core Idea

**PTQ = reduce the precision of trained weights from fp16 → INT8/INT4 using calibration data to minimize quality loss.**

```bash
Training pipeline:     base model → fine-tune (LoRA/QLoRA) → fp16 checkpoint
PTQ pipeline:          fp16 checkpoint → calibrate → INT8/INT4 checkpoint → deploy
```

The key distinction from QLoRA:

- **QLoRA** quantizes the base model **during training** to save training VRAM
- **PTQ** quantizes the final model **after training** to save inference cost
- They solve different bottlenecks. You often want both.

## Quantization Fundamentals

### What Quantization Does

Maps continuous fp16 values to a discrete set of lower-precision values:

$$x_q = \text{round}\left(\frac{x}{\Delta}\right) + z$$

Where:

- $\Delta$ is the **scale** (step size between quantization levels)
- $z$ is the **zero-point** (offset to handle asymmetric distributions)
- $x_q$ is the quantized integer value

For INT8: 256 possible levels. For INT4: 16 possible levels.

### Per-Tensor vs Per-Channel Quantization

| Granularity | How it works                                  | Accuracy | Overhead       |
|-------------|-----------------------------------------------|----------|----------------|
| Per-tensor  | One scale/zero-point for entire weight matrix | Lowest   | Minimal        |
| Per-channel | One scale/zero-point per output channel       | Better   | Slight         |
| Per-group   | One scale/zero-point per group of N weights   | Best     | More constants |

Per-channel is the standard for weight quantization. Per-group (block size 64 or 128) is used by GPTQ and AWQ for INT4.

## PTQ Methods

### 1. Naive PTQ (Round-to-Nearest)

Simplest approach: quantize every weight independently using min/max range.

```bash
For each weight tensor W:
    scale = (max(W) - min(W)) / (2^bits - 1)
    W_q = round(W / scale) + zero_point
```

**Problem**: Treats all weights equally. Some weights are far more important than others — a small error in a critical weight can cascade through layers and destroy output quality.

For Mistral-7B:

- INT8 naive PTQ: typically <1 ROUGE-L drop — acceptable for most tasks
- INT4 naive PTQ: 3-8 ROUGE-L drop — often unacceptable (the "quality cliff")

### 2. GPTQ (GPT-Quantization)

Uses calibration data to quantize weights in an order that minimizes reconstruction error. Based on Optimal Brain Quantization (OBQ), adapted to scale to billions of parameters.

**Core idea**: Quantize one weight at a time, then adjust remaining weights to compensate for the error introduced.

$$\underset{\hat{W}}{\arg\min} \| WX - \hat{W}X \|_2^2$$

Where $X$ is the calibration data activations and $\hat{W}$ is the quantized weight matrix.

**Algorithm**:

1. Collect activations $X$ by running calibration samples through the model
2. Compute Hessian $H = 2X X^T$ (captures which weights matter most)
3. For each column of $W$:
   - Quantize the column
   - Compute quantization error
   - Update remaining columns to compensate: $\delta_F = -\frac{w_q - w}{H_{qq}} \cdot H_{:,q}$
4. Repeat layer-by-layer

**Result**: INT4 with GPTQ recovers most of the quality that naive PTQ destroys. Typical quality gap vs fp16 is <0.5 ROUGE-L.

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,         # per-group quantization
    desc_act=True,          # quantize in activation-order (more accurate, slower)
    damp_percent=0.01,      # Hessian dampening for numerical stability
)

model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quantize_config,
)

# Calibration — typically 128-512 samples from training distribution
model.quantize(calibration_dataset)
model.save_quantized(output_dir)
```

### 3. AWQ (Activation-Aware Weight Quantization)

**Insight**: Not all weights are equally important. Weights connected to large activations matter more — a small quantization error on those weights gets amplified by large activation values.

AWQ identifies **salient weight channels** by analyzing activation magnitudes, then protects them during quantization:

$$\text{importance}(w_i) \propto |a_i| \cdot |w_i|$$

Where $a_i$ is the average activation magnitude for channel $i$.

**Algorithm**:

1. Run calibration data, record per-channel activation magnitudes
2. Identify salient channels (top 1% by activation magnitude)
3. Scale salient channels up before quantization (protecting them from rounding error)
4. Scale corresponding subsequent layers down to compensate

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_path)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_samples,  # 128-512 samples
)
model.save_quantized(output_dir)
```

**AWQ vs GPTQ**: AWQ is faster to quantize (no Hessian computation) and often produces comparable or better quality. AWQ focuses on "which weights matter" via activations; GPTQ focuses on "how to compensate for errors" via Hessian.

### 4. SmoothQuant (Weight-Activation Co-Quantization)

Previous methods quantize **weights only**. SmoothQuant also quantizes **activations** to INT8, enabling fully integer matrix multiplication.

**Problem**: Activations have outliers — a few channels have values 100x larger than others. Direct INT8 quantization of activations destroys information.

**Solution**: Migrate quantization difficulty from activations to weights by channel-wise scaling:

$$Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) = \hat{X} \hat{W}$$

Choose $s$ to balance the quantization difficulty:

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

Where $\alpha$ (typically 0.5) controls the tradeoff between activation and weight quantization difficulty.

**Result**: Both weights AND activations in INT8 → W8A8 execution → use INT8 Tensor Cores → ~2x throughput on supported hardware.

## Calibration: The Critical Step

All PTQ methods beyond naive require **calibration data** — a small representative set of inputs used to understand the model's behavior.

### What calibration does

- **GPTQ**: Computes Hessian (which weights matter for reconstruction)
- **AWQ**: Measures per-channel activation magnitudes
- **SmoothQuant**: Identifies activation outlier channels

### Calibration dataset requirements

- **Size**: 128-512 samples is standard. More doesn't help significantly.
- **Distribution**: Must match inference distribution. If you fine-tuned on CNN/DailyMail summarization, calibrate on CNN/DailyMail samples.
- **Diversity**: Cover the range of inputs the model will see. Don't calibrate on only short articles if you'll serve long ones.

### Calibration anti-patterns

- Using random text (Wikipedia) to calibrate a task-specific model — distribution mismatch
- Using too few samples (<32) — unstable Hessian estimates
- Using too many samples (>1024) — diminishing returns, wastes compute

## Memory Impact for Mistral-7B

| Precision          | Weight Memory | Quantization Constants | Total   | Reduction |
|--------------------|---------------|------------------------|---------|-----------|
| fp16               | 13.5 GB       | 0                      | 13.5 GB | 1x        |
| INT8 (per-channel) | 6.75 GB       | ~50 MB                 | ~6.8 GB | 2x        |
| INT4 (GPTQ, g=128) | 3.4 GB        | ~0.2 GB                | ~3.6 GB | 3.8x      |
| INT4 (AWQ, g=128)  | 3.4 GB        | ~0.2 GB                | ~3.6 GB | 3.8x      |

Group size 128 means one scale/zero-point per 128 weights. Smaller groups = more constants but better accuracy.

## Quality vs Bit-Width: The Quality Cliff

```bash
Quality (ROUGE-L)
  |
  |  ████████████████████████  fp16 (baseline)
  |  ███████████████████████   INT8 naive (~<1 pt drop)
  |  ██████████████████████    INT8 SmoothQuant (~<0.5 pt)
  |  ████████████████████      INT4 GPTQ (~0.5-1 pt drop)
  |  ███████████████████       INT4 AWQ (~0.5-1 pt drop)
  |  ██████████████            INT4 naive (3-8 pt drop) <- quality cliff
  |  █████████                 INT3 (usually unacceptable)
  |
  +---------------------------- Bit-width ->
```

**The cliff is at INT4 naive** — this is where calibration-aware methods (GPTQ, AWQ) earn their keep. Without calibration, INT4 is often unusable. With calibration, INT4 is practical.

## Throughput Impact

PTQ affects inference speed through two mechanisms:

### 1. Reduced memory bandwidth (faster)

Smaller weights = less data moved from HBM to compute units. For memory-bandwidth-bound autoregressive generation, this is a direct speedup.

### 2. Dequantization overhead (slower)

INT4/INT8 weights must be dequantized to fp16/bf16 before matrix multiplication (unless using INT8 Tensor Cores with SmoothQuant).

| Method                   | Throughput vs fp16 | Why                              |
|--------------------------|--------------------|----------------------------------|
| INT8 (W8A16)             | ~1.0-1.2x          | Bandwidth savings ~ dequant cost |
| INT4 (W4A16)             | ~1.2-1.5x          | Bandwidth savings > dequant cost |
| INT8 (W8A8, SmoothQuant) | ~1.5-2.0x          | INT8 Tensor Cores, no dequant    |

## PTQ vs QLoRA: Different Problems, Different Solutions

| Aspect                 | QLoRA                                          | PTQ (GPTQ/AWQ)                   |
|------------------------|------------------------------------------------|----------------------------------|
| **When applied**       | During training                                | After training                   |
| **What it solves**     | Training VRAM                                  | Inference cost                   |
| **Precision**          | NF4 (4-bit) base + bf16 adapters               | INT8 or INT4 final model         |
| **Calibration needed** | No (NF4 levels are precomputed)                | Yes (128-512 samples)            |
| **Quality impact**     | None on training (adapters are full precision) | Depends on method and bit-width  |
| **Can combine**        | Yes — QLoRA train -> merge -> GPTQ quantize    | Yes — PTQ the QLoRA-merged model |

**The production recipe**: QLoRA to train cheaply -> merge adapters -> GPTQ/AWQ to serve cheaply.

## Debugging Quantization Quality Loss

When PTQ drops quality below threshold:

### Step 1: Which method?

Naive PTQ is lossy by construction — switch to GPTQ or AWQ with calibration data.

### Step 2: Which layers broke?

Run per-layer sensitivity analysis — quantize one layer at a time, measure quality delta. Attention layers in early/late transformer blocks are usually most sensitive.

```bash
Layer 0  (attention): quantize -> -0.8 ROUGE-L  <- sensitive
Layer 1  (attention): quantize -> -0.1 ROUGE-L  <- safe
...
Layer 31 (attention): quantize -> -0.5 ROUGE-L  <- sensitive
```

### Step 3: Mixed precision

Keep sensitive layers in fp16, quantize the rest. Typically keeping 2-4 layers in fp16 recovers most of the quality while retaining most of the memory savings.

### Step 4: If PTQ can't recover

QAT (Quantization-Aware Training) on the sensitive layers only — don't retrain the whole model. QAT learns to be robust to quantization during training, but costs GPU hours.

## GGUF: PTQ for CPU/Edge Deployment

GGUF (GPT-Generated Unified Format) is the llama.cpp quantization format, optimized for CPU inference:

| GGUF Variant | Bits  | Method                     | Typical Quality         |
|--------------|-------|----------------------------|-------------------------|
| Q8_0         | 8-bit | Per-block round-to-nearest | Near fp16               |
| Q6_K         | 6-bit | K-quant (super-block)      | Very close to fp16      |
| Q4_K_M       | 4-bit | K-quant mixed precision    | Good for most tasks     |
| Q4_0         | 4-bit | Per-block round-to-nearest | Noticeable degradation  |
| Q2_K         | 2-bit | K-quant                    | Significant degradation |

K-quants use mixed precision within the model — more bits for sensitive layers, fewer for robust ones. `Q4_K_M` is the standard choice for 4-bit deployment.

## PTQ in the Project Pipeline

```bash
Day 1-6: Fine-tune with LoRA/QLoRA -> best checkpoint (fp16)
     |
Day 7: PTQ pipeline
     |-- Load best fine-tuned checkpoint
     |-- Prepare calibration set (128 CNN/DailyMail samples)
     |-- Run GPTQ/AWQ quantization -> INT4 model
     |-- Evaluate: ROUGE-L, throughput, VRAM
     +-- Compare against fp16 baseline
     |
Day 8+: TensorRT INT8 deployment (separate pipeline)
```

Our comparison matrix:

| Method          | ROUGE-L       | VRAM    | Throughput    | Training Cost           |
|-----------------|---------------|---------|---------------|-------------------------|
| LoRA fp16       | Baseline      | 13.5 GB | 1x            | GPU hours               |
| QLoRA merged    | ~Baseline     | 13.5 GB | ~0.75x        | Less GPU hours          |
| PTQ INT8        | ~Baseline     | ~6.8 GB | ~1.0-1.2x     | Zero (calibration only) |
| PTQ INT4 (GPTQ) | -0.5 to -1 pt | ~3.6 GB | ~1.2-1.5x     | Zero (calibration only) |
| GGUF Q4_K_M     | -0.5 to -1 pt | ~3.6 GB | CPU-optimized | Zero                    |
