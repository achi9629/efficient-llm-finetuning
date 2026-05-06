<!-- markdownlint-disable-file MD029 -->

# Experiment Results — Efficient LLM Fine-Tuning + Quantization

**Setup:** Mistral-7B-v0.3 · CNN/DailyMail · single A100 GPU

---

## 1. Inference Comparison (All Methods)

| Method                | ROUGE-L | Greedy tok/s p50 | Beam tok/s p50 | TTFT avg (ms) | Peak VRAM (GB) | Model Size (GB) |
|-----------------------|---------|------------------|----------------|---------------|----------------|-----------------|
| Base FP16             | 0.139   | 174.0            | 29.5           | 14.7          | 14.79          | 13.50           |
| LoRA scratch (r=16)   | 0.183   | 178.5            | 29.7           | 15.4          | 14.53          | 13.50           |
| LoRA PEFT (r=2)       | 0.201   | 177.2            | 30.4           | 14.1          | 14.53          | 13.50           |
| QLoRA (r=2, 4-bit)    | 0.201   | 69.8             | 10.1           | 13.5          | 5.88           | 4.26            |
| PTQ INT8 (base)       | 0.137   | 40.1             | 6.9            | 15.3          | 8.13           | 7.01            |
| PTQ NF4 (base)        | 0.133   | 112.3            | 15.5           | 15.0          | 5.21           | 4.16            |
| GPTQ INT4 (base)      | 0.142   | 33.9             | 5.5            | 15.4          | 4.91           | 3.88            |
| PTQ INT8 (LoRA)       | 0.196   | 109.5            | 6.9            | 15.1          | 8.13           | 7.01            |
| PTQ NF4 (LoRA)        | 0.201   | 285.2            | 15.1           | 13.7          | 5.21           | 4.16            |
| GPTQ INT4 (LoRA)      | 0.196   | 84.4             | 5.5            | 14.5          | 4.91           | 3.88            |
| AWQ INT4 (LoRA)       | 0.196   | 341.1            | 23.3           | 12.7          | 4.96           | 3.88            |
| QAT BF16 (LoRA)       | 0.200   | 170.9            | 29.6           | 13.9          | 14.53          | 13.50           |
| GPTQ INT4 (QAT)       | 0.200   | 85.4             | 5.3            | 15.0          | 4.90           | 3.88            |
| AWQ INT4 (QAT)        | 0.201   | 368.1            | 23.4           | 13.3          | 4.90           | 3.88            |

**Notes:**

- LoRA scratch/PEFT inference is on merged FP16 model (adapters folded into base weights → zero overhead).
- QLoRA inference keeps 4-bit base + LoRA adapters (no merge possible for NF4 weights).
- PTQ INT8/NF4 (base) are applied to the un-fine-tuned base model. No training involved.
- GPTQ INT4 (base) is applied to the un-fine-tuned base model with 128 validation samples as calibration data.
- PTQ INT8/NF4 (LoRA) rows = bitsandbytes quantization applied to the LoRA-merged FP16 checkpoint.
- GPTQ/AWQ INT4 (LoRA) = calibration-based quantization applied to the LoRA-merged FP16 checkpoint.
- **Throughput caveat:** Base and fine-tuned models produce different output lengths (base: ~200 tokens verbose paragraphs; fine-tuned: ~30-60 tokens concise bullets). Since tok/s = generated_tokens / total_time, throughput is **not directly comparable** across base vs fine-tuned rows. Compare within the same model family only (e.g., base INT8 vs base NF4 vs base GPTQ).
- QAT BF16 (LoRA) = LoRA-merged FP16 checkpoint → selective QAT (fake-quant on blocks 14/26/19, down_proj+o_proj, 250 steps) → bf16 inference. Not yet INT4-quantized.
- GPTQ/AWQ INT4 (QAT) = QAT checkpoint → GPTQ/AWQ calibration-based INT4 quantization. Tests whether QAT pre-training reduces quality loss during subsequent PTQ.

---

## 2. Training Cost

| Method                      | Peak VRAM (GB) | Train Time (min) | Train tok/s | Fits 24GB? | Steps |
|-----------------------------|----------------|------------------|-------------|------------|-------|
| LoRA (PEFT, FP16 base)      | 56.15          | 7.84             | 2,847       | No         | 1875  |
| QLoRA (4-bit base)          | 45.46          | 9.20             | 2,427       | No         | 1875  |
| QLoRA + Grad Checkpointing  | 9.99           | 12.92            | 1,729       | **Yes**    | 1875  |
| QAT (selective, bf16)       | 28.7           | 14.08            | 3,922       | No         | 250   |

Config: bs=4, grad_accum=4, avg_seq_len=837.6. LoRA/QLoRA = 3 epoch full fine-tune; QAT = 250 steps on frozen model with 6 fake-quant layers.

---

## 3a. LoRA PEFT Rank Sweep

| Rank | Alpha | ROUGE-L | ROUGE-Lsum | Greedy tok/s p50 | Beam tok/s p50 | Peak VRAM (GB) | Model Size (GB) |
|------|------:|--------:|-----------:|-----------------:|---------------:|---------------:|----------------:|
| 2    | 4     | 0.2011  | 0.2638     | 177.3            | 30.4           | 14.53          | 13.50           |
| 4    | 8     | 0.1918  | 0.2539     | 179.4            | 30.4           | 14.53          | 13.50           |
| 8    | 16    | 0.1968  | 0.2565     | 184.0            | 30.2           | 14.53          | 13.50           |
| 16   | 32    | 0.1917  | 0.2484     | 181.0            | 30.5           | 14.53          | 13.50           |
| 32   | 64    | 0.1934  | 0.2594     | 184.1            | 29.9           | 14.53          | 13.50           |
| 64   | 128   | 0.1839  | 0.2487     | 183.4            | 30.5           | 14.53          | 13.50           |

**Selected:** r=2, alpha=4 — highest ROUGE-L (0.201).

**Key observations:**

- r=2 dominates across all ranks; r=64 is worst. The useful adaptation lives in a very low-rank subspace for this dataset size.
- VRAM and throughput are constant across ranks — after merging, rank does not affect inference cost.
- Merged LoRA inference runs at full FP16 speed (~180 tok/s greedy), confirming zero overhead post-merge.

## 3b. QLoRA Rank Sweep

| Rank | Alpha | ROUGE-L | ROUGE-Lsum | Greedy tok/s p50 | Beam tok/s p50 | Peak VRAM (GB) | TTFT avg (ms) |
|------|------:|--------:|-----------:|-----------------:|---------------:|---------------:|--------------:|
| 2    | 4     | 0.2013  | 0.2630     | 69.79            | 10.11          | 5.88           | 13.50         |
| 4    | 8     | 0.2008  | 0.2617     | 66.04            | 7.71           | 5.88           | 12.98         |
| 8    | 16    | 0.1912  | 0.2532     | 67.20            | 9.92           | 5.88           | 11.64         |
| 16   | 32    | 0.1923  | 0.2526     | 68.00            | 9.97           | 5.88           | 11.85         |
| 32   | 64    | 0.1892  | 0.2540     | 68.63            | 9.97           | 5.88           | 12.48         |
| 64   | 128   | 0.1802  | 0.2429     | 69.66            | 8.89           | 5.88           | 11.86         |

**Selected:** r=2, alpha=4 — highest ROUGE-L (0.201).

**Key observations:**

- Same rank ordering as LoRA: r=2 wins, r=64 is worst. Low-rank subspace is sufficient for this task/dataset.
- 4-bit base quantization does not degrade the fine-tuned solution — QLoRA r=2 (0.201) matches LoRA r=2 (0.201) exactly.
- QLoRA trades inference speed (~70 tok/s vs ~180 tok/s) for 2.5× VRAM reduction (5.88 vs 14.53 GB) — a direct memory-latency tradeoff.

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

### Phase B — Post-Training Quantization

9. **Fine-tuning + quantization is the winning combo.** LoRA-merged + PTQ NF4 achieves ROUGE-L 0.201 at 5.21 GB VRAM — matching QLoRA quality at similar memory with 4× higher throughput (285 vs 70 tok/s). NF4 is faster than FP16 for autoregressive decode because generation is memory-bandwidth-bound — 4-bit weights read 3× fewer bytes per token from HBM, and the dequantization compute is fully overlapped on A100.
10. **GPTQ wins quality among base-model PTQ methods** (0.142 vs 0.137/0.133). Calibration-based Hessian optimization recovers ~0.5-0.9 pt ROUGE-L over naive quantization at 4-bit.
11. **INT8 is slower than NF4 despite higher precision.** LLM.int8() mixed-precision decomposition (outlier handling) has more overhead than NF4 bulk dequantization. This is counterintuitive but well-documented.
12. **GPTQ has lowest VRAM (4.91 GB) but slowest 4-bit throughput** (33.9 tok/s greedy). The `auto_gptq` Triton kernels underperform bitsandbytes NF4. ExLlama/Marlin backends would likely recover 2-3× throughput.
13. **AWQ dominates GPTQ on throughput** (341 vs 84 tok/s for LoRA-merged) with identical quality (0.196 ROUGE-L) and similar VRAM (4.96 vs 4.91 GB). The `autoawq` GEMM kernels are well-optimized for Mistral's architecture on A100, delivering 2× faster decode than FP16. GPTQ's `auto_gptq` Triton kernels remain the bottleneck.
14. **Applying PTQ to the fine-tuned model closes the quality gap entirely.** Base GPTQ: 0.142 → LoRA-merged GPTQ: 0.196 (+5.4 pt). The quantization method doesn't matter as much as what you quantize.

### Throughput Measurement Note

15. **Throughput (tok/s) is not comparable between base and fine-tuned models.** The base model generates verbose paragraphs (~200+ tokens, often hitting max_new_tokens=256). Fine-tuned models produce concise bullet summaries (~30-60 tokens, early EOS). Since tok/s = generated_tokens / (prefill_time + decode_time), different generation lengths change how prefill overhead is amortized. Compare throughput **within** the same model family only.

### Phase B — Per-Layer Sensitivity (Day 9)

16. **Mid-network blocks degrade most under INT4, not first/last.** Per-layer sweep (150 samples, simulated INT4 quantize-dequantize) found blocks 14 (-0.72pt), 26 (-0.62pt), 19 (-0.55pt) as top-3 most sensitive. Block 0 ranked 15th. Contradicts the common assumption that embedding-adjacent layers are most fragile.
17. **Output projections (`down_proj`, `o_proj`) are the sensitive module types.** MLP output projection (-0.29pt) and attention output projection (-0.11pt) degrade most when quantized across all 32 blocks. These feed directly into the residual stream — rounding errors propagate immediately.
18. **Single-block INT4 quantization is survivable.** The sensitivity sweep quantizes **one block at a time** (the other 31 stay FP16) and measures the isolated ROUGE-L drop. The worst single-block drop was only -0.72pt (block 14). This means no individual block is catastrophically fragile — the model can tolerate INT4 rounding in any single block without major quality loss.

![Per-block ROUGE-L delta under INT4](../assets/plots/sensitivity_per_block.png)
![Per-module ROUGE-L delta under INT4](../assets/plots/sensitivity_per_module.png)

### Phase C — Quantization-Aware Training (Day 10)

19. **QAT preserves full FP16 quality with zero latency cost.** Selective fake-quantization on 3 most sensitive blocks (14, 26, 19) × `down_proj`/`o_proj` for 250 steps → ROUGE-L 0.200, 170.9 tok/s, 14.53 GB. The -0.1pt quality drop and -2% throughput are within noise of the unquantized LoRA baseline.
20. **QAT alone doesn't solve the VRAM problem.** The QAT checkpoint is still bf16 weights (13.5 GB). QAT's value is as a *pre-processing step* before INT4 export — training the sensitive layers to be robust to rounding, so that subsequent GPTQ/AWQ quantization loses less quality.
21. **QAT→AWQ closes the quality gap AND wins throughput.** AWQ INT4 (QAT): 0.201 ROUGE-L, 368 tok/s, 4.90 GB — passes all 3 KPI gates. The 0.5pt recovery (0.196→0.201) confirms QAT prepared the sensitive layers to survive INT4 rounding without quality loss.
22. **QAT→GPTQ recovers quality but not throughput.** GPTQ INT4 (QAT): 0.200 ROUGE-L (+0.4pt vs naive GPTQ), 85.4 tok/s, 4.9 GB. Quality improves but GPTQ's slow Triton kernels remain the bottleneck — still fails the latency gate.

---

## 5. KPI Budget Check

| Method               | ROUGE-L drop (≤ 2 pt)  | Latency regression† (≤ 20%)  | VRAM (≤ 10 GB) | All Pass? |
|----------------------|------------------------|------------------------------|----------------|-----------|
| Base FP16            | —                      | baseline                     | 14.79 ✗        | ✗         |
| QLoRA (r=2)          | **best**               | -60% ✗                       | 5.88 ✓         | ✗         |
| PTQ INT8 (base)      | -6.4 pt ✗              | *                            | 8.13 ✓         | ✗         |
| PTQ NF4 (base)       | -6.8 pt ✗              | *                            | 5.21 ✓         | ✗         |
| GPTQ INT4 (base)     | -5.9 pt ✗              | *                            | 4.91 ✓         | ✗         |
| PTQ INT8 (LoRA)      | -0.5 pt ✓              | -38% ✗                       | 8.13 ✓         | ✗         |
| PTQ NF4 (LoRA)       | 0.0 pt ✓               | +61% ✓                       | 5.21 ✓         | **✓**     |
| GPTQ INT4 (LoRA)     | -0.5 pt ✓              | -52% ✗                       | 4.91 ✓         | ✗         |
| AWQ INT4 (LoRA)      | -0.5 pt ✓              | +92% ✓                       | 4.96 ✓         | **✓**     |
| QAT BF16 (LoRA)      | -0.1 pt ✓              | -2% ✓                        | 14.53 ✗        | ✗         |
| GPTQ INT4 (QAT)      | -0.1 pt ✓              | -52% ✗                       | 4.90 ✓         | ✗         |
| AWQ INT4 (QAT)       | 0.0 pt ✓               | +108% ✓                      | 4.90 ✓         | **✓**     |

†Throughput not directly comparable across base vs fine-tuned models — generation length differences invalidate cross-group latency comparisons (see Finding 15). Latency regression is only meaningful within the same model family.
*Not evaluated — base models generate different output lengths vs fine-tuned, making cross-family tok/s comparison invalid. LoRA variants compared against LoRA FP16 merged (177.2 tok/s).

**Key gap:** Quality and VRAM are solved across all LoRA-merged INT4 methods. Latency is now solved for three methods: PTQ NF4 (+61%), AWQ INT4 (LoRA) (+92%), and AWQ INT4 (QAT) (+108%). GPTQ still fails latency (-52%) due to slow `auto_gptq` Triton kernels.

**Recommendation:** AWQ INT4 (QAT) is the best overall method — 0.201 ROUGE-L (0 pt drop), 4.90 GB VRAM, 368 tok/s (+108% vs FP16). The full pipeline: LoRA fine-tune → merge → selective QAT (250 steps) → AWQ INT4 export. If QAT training cost is unacceptable, AWQ INT4 (LoRA) is nearly as good: 0.196 ROUGE-L, 341 tok/s, 4.96 GB.
