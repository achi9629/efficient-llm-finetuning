<!-- markdownlint-disable-file MD029 -->

# Experiment Results — Efficient LLM Fine-Tuning + Quantization

**Setup:** Mistral-7B-v0.3 · CNN/DailyMail · single GPU (A100/L40)

---

## 1. Inference Comparison (All Methods)

| Method                | ROUGE-L | Greedy tok/s p50 | Beam tok/s p50 | Peak VRAM (GB) | Model Size (GB) |
|-----------------------|---------|------------------|----------------|----------------|-----------------|
| Base FP16             | 0.139   | 174.0            | 29.5           | 14.79          | 13.50           |
| LoRA scratch (r=16)   | 0.183   | 178.5            | 29.7           | 14.53          | 13.50           |
| LoRA PEFT (r=16)      | 0.191   | 176.9            | 29.8           | 14.53          | 13.50           |
| QLoRA (r=2, 4-bit)    | 0.201   | 69.8             | 10.1           | 5.88           | 4.26            |
| PTQ INT8 (base)       | 0.137   | 40.9             | 7.1            | 8.13           | 7.00            |
| PTQ NF4 (base)        | 0.133   | 108.6            | 15.5           | 5.21           | 3.75            |
| GPTQ INT4 (base)      | 0.142   | 34.0             | 5.5            | 4.91           | 3.88            |

**Notes:**

- LoRA scratch/PEFT inference is on merged FP16 model (adapters folded into base weights → zero overhead).
- QLoRA inference keeps 4-bit base + LoRA adapters (no merge possible for NF4 weights).
- PTQ INT8/NF4 are applied to the un-fine-tuned base model. No training involved.
- GPTQ INT4 is applied to the un-fine-tuned base model with 128 validation samples as calibration data.

---

## 2. Training Cost (Phase A)

| Method                      | Peak VRAM (GB) | Train Time (min) | Train tok/s | Fits 24GB? |
|-----------------------------|----------------|------------------|-------------|------------|
| LoRA (PEFT, FP16 base)      | 56.15          | 7.84             | 2,847       | No         |
| QLoRA (4-bit base)          | 45.46          | 9.20             | 2,427       | No         |
| QLoRA + Grad Checkpointing  | 9.99           | 12.92            | 1,729       | **Yes**    |

Config: bs=4, grad_accum=4, bf16, 1 epoch, 100 steps, avg_seq_len=837.6

---

## 3. QLoRA Rank Sweep

| Rank | Alpha | ROUGE-L | ROUGE-Lsum | Greedy tok/s p50 | Beam tok/s p50 | Peak VRAM (GB) | TTFT avg (ms) |
|------|------:|--------:|-----------:|-----------------:|---------------:|---------------:|--------------:|
| 2    | 4     | 0.2013  | 0.2630     | 69.79            | 10.11          | 5.88           | 13.50         |
| 4    | 8     | 0.2008  | 0.2617     | 66.04            | 7.71           | 5.88           | 12.98         |
| 8    | 16    | 0.1912  | 0.2532     | 67.20            | 9.92           | 5.88           | 11.64         |
| 16   | 32    | 0.1923  | 0.2526     | 68.00            | 9.97           | 5.88           | 11.85         |
| 32   | 64    | 0.1892  | 0.2540     | 68.63            | 9.97           | 5.88           | 12.48         |
| 64   | 128   | 0.1802  | 0.2429     | 69.66            | 8.89           | 5.88           | 11.86         |

**Selected:** r=2, alpha=4 — highest ROUGE-L, best throughput, same VRAM as all ranks.

---

## 4. Key Findings

### Phase A — Adaptation

1. **LoRA merge = zero inference overhead.** After merging, LoRA/PEFT models have identical VRAM and throughput to base FP16. The adapter weights disappear into the base matrices.
2. **QLoRA outperformed LoRA on this small dataset** (0.201 vs 0.191 ROUGE-L) — likely 4-bit quantization acting as implicit regularization. On larger datasets, LoRA with full-precision base could win.
3. **Lower rank won the sweep.** r=2 beat r=64 by 2.1 pt ROUGE-L. The useful adaptation for this task/data size lives in a very low-rank subspace.
4. **Neither LoRA nor QLoRA fits 24GB without gradient checkpointing.** Activations dominate VRAM, not model weights. Even 4-bit base needs 45 GB without GC.
5. **Gradient checkpointing trades 40% wall-clock time for 82% VRAM reduction** (56→10 GB). QLoRA + GC is the only method that fits a single 24GB GPU.
6. **4-bit quantization adds ~17% training throughput overhead** (2,427 vs 2,847 tok/s) from dequantization during forward/backward passes.
7. **QLoRA inference throughput is 60% slower than FP16** (70 vs 174 tok/s) — dequantization overhead on every forward pass.
8. **Scratch LoRA matches PEFT within 0.9 pt ROUGE-L** — validates custom implementation correctness.

### Phase B — PTQ

9. **PTQ INT8 preserves quality** (0.137 vs 0.139 baseline, only 0.2 pt drop) but **throughput collapses** to 40.9 tok/s (76% slower than FP16). LLM.int8() outlier-aware decomposition is the bottleneck.
10. **PTQ NF4 has best VRAM efficiency** (5.21 GB, 65% reduction) with minimal quality drop (0.6 pt) and moderate throughput penalty (37% slower at 108.6 tok/s).
11. **INT8 is slower than NF4** despite higher precision — LLM.int8() mixed-precision decomposition has more overhead than NF4 bulk dequantization. This is counterintuitive but well-documented.
12. **PTQ is on the un-fine-tuned base model.** To get PTQ + fine-tuning gains, you'd merge LoRA adapters first, then apply PTQ to the merged checkpoint. That experiment is pending.

### Phase B — GPTQ

13. **GPTQ wins quality vs bitsandbytes PTQ** (0.142 vs 0.137/0.133). Calibration-based Hessian optimization recovers ~0.5-0.9 pt ROUGE-L over naive quantization at the same bit-width (4-bit).
14. **GPTQ has lowest VRAM** (4.91 GB) but **slowest throughput** (34.0 tok/s, 80% regression). The `auto_gptq` Triton kernels are slower than bitsandbytes NF4 dequantization. ExLlama/Marlin kernels would likely be 2-3x faster.
15. **All PTQ/GPTQ results are on the un-fine-tuned base model.** The ~5.9 pt gap vs QLoRA is not a fair comparison — applying GPTQ to the merged LoRA checkpoint would likely close most of this gap.

---

## 5. KPI Budget Check

| KPI                          | Threshold           | Base FP16 | QLoRA (r=2) | PTQ INT8 | PTQ NF4 | GPTQ INT4 |
|------------------------------|---------------------|-----------|-------------|----------|---------|-----------|
| ROUGE-L drop vs best FT      | ≤ 2 pt              | —         | **best**    | -6.4 pt* | -6.8 pt*| -5.9 pt*  |
| Latency regression (greedy)  | ≤ 20%               | baseline  | -60% ✗      | -76% ✗   | -37% ✗  | -80% ✗    |
| Inference VRAM (quantized)   | ≤ 10 GB             | 14.79 ✗   | 5.88 ✓      | 8.13 ✓   | 5.21 ✓  | 4.91 ✓    |
| Inference VRAM (FP16)        | ≤ 16 GB             | 14.79 ✓   | —           | —        | —       | —         |

*PTQ ROUGE-L drop is vs QLoRA best (0.201), not vs base. PTQ was run on the un-fine-tuned base model.

**Key gap:** No method currently passes the latency budget. All quantized methods (QLoRA, PTQ, GPTQ) exceed 20% throughput regression. GPTQ did NOT close the latency gap — 34 tok/s is even slower than NF4. LoRA merged FP16 passes latency but not the ≤10GB VRAM target. GGUF with llama.cpp kernels or ExLlama/Marlin backends for GPTQ may be needed. Also, applying quantization to the fine-tuned (merged LoRA) checkpoint — rather than the base model — would close the quality gap.
