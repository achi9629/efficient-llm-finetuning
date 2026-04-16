# Metrics Reference

Every experiment in this project produces a set of numbers that answer three questions:
**Is the output good?** (Quality), **Is it fast?** (Latency), and **Does it fit?** (Memory).
This document defines each metric, explains the intuition behind it, and states why we report it.

---

## 1  Quality KPIs — ROUGE Family

ROUGE (**R**ecall-**O**riented **U**nderstudy for **G**isting **E**valuation) measures how much
of the reference summary is captured by the generated summary.  All ROUGE variants compare
n-gram overlap between a **prediction** (model output) and a **reference** (ground-truth
`highlights` field in CNN/DailyMail).

We report the **F1** variant of every ROUGE score (harmonic mean of precision and recall)
because it balances both directions: *Did the model say everything important?* (recall) and
*Did it avoid saying irrelevant things?* (precision).

### 1.1  ROUGE-1 (F1)

| Item | Detail |
|------|--------|
| **What it measures** | Unigram (single-word) overlap between prediction and reference. |
| **Formula** | $F_1 = \frac{2 \cdot P \cdot R}{P + R}$ where $P = \frac{\lvert \text{pred unigrams} \cap \text{ref unigrams} \rvert}{\lvert \text{pred unigrams} \rvert}$, $R = \frac{\lvert \text{pred unigrams} \cap \text{ref unigrams} \rvert}{\lvert \text{ref unigrams} \rvert}$ |
| **Intuition** | Counts how many individual words the model got right, regardless of order. A high ROUGE-1 means the model uses the right vocabulary even if the phrasing differs. |
| **Example** | Reference: `"The cat sat on the mat"` → unigrams: {the, cat, sat, on, mat}. Prediction: `"A cat is on the mat"` → {a, cat, is, on, the, mat}. Overlap: {cat, on, the, mat} → $P = 4/6$, $R = 4/5$, $F_1 \approx 0.73$ |
| **Why report it** | Captures content coverage at the word level. A sudden drop after quantization or fine-tuning means the model is missing key terms from the reference. |
| **Computed by** | `evaluate.load("rouge")` in `evaluate_task.py → compute_rouge()` |

### 1.2  ROUGE-2 (F1)

| Item | Detail |
|------|--------|
| **What it measures** | Bigram (two-consecutive-word) overlap between prediction and reference. |
| **Formula** | Same F1 formula as ROUGE-1, applied to bigrams instead of unigrams. |
| **Intuition** | Rewards the model for getting two-word phrases right — i.e., local fluency and phrasing. A model that uses the right words in the wrong order will score well on ROUGE-1 but poorly on ROUGE-2. |
| **Example** | Reference bigrams: {(the, cat), (cat, sat), (sat, on), (on, the), (the, mat)}. Prediction bigrams: {(a, cat), (cat, is), (is, on), (on, the), (the, mat)}. Overlap: {(on, the), (the, mat)} → $R = 2/5$, $P = 2/5$, $F_1 = 0.40$ |
| **Why report it** | Penalizes garbled or reordered output that ROUGE-1 misses. A ROUGE-2 drop with stable ROUGE-1 indicates fluency degradation — common after aggressive quantization. |
| **Computed by** | Same `compute_rouge()` call; returned as `rouge2`. |

### 1.3  ROUGE-L (F1)

| Item | Detail |
|------|--------|
| **What it measures** | Longest Common Subsequence (LCS) between prediction and reference at the **sentence** level. |
| **Formula** | $R_{lcs} = \frac{\text{LCS}(pred, ref)}{\lvert ref \rvert}$, $P_{lcs} = \frac{\text{LCS}(pred, ref)}{\lvert pred \rvert}$, $F_{lcs} = \frac{(1 + \beta^2) \cdot P_{lcs} \cdot R_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}$ with $\beta = P_{lcs}/R_{lcs}$. |
| **Intuition** | LCS does not require contiguous matches, so it captures sentence-level structural similarity. Think of it as "did the model follow roughly the same narrative arc as the reference?" without requiring exact bigram matches. |
| **Example** | Reference: `"police arrested the suspect yesterday"`. Prediction: `"the suspect was arrested by police"`. LCS: `"the suspect … police"` (length 3). $R = 3/5 = 0.60$, $P = 3/6 = 0.50$, $F_1 \approx 0.55$. Despite completely different word order, the subsequence captures shared structure. |
| **Why report it** | Our **primary go/no-go metric**. Threshold: ≤ 2-point drop from best fine-tuned checkpoint. ROUGE-L is less sensitive to minor word-order changes than ROUGE-2, making it a more robust single-number summary quality indicator. |
| **Computed by** | Same `compute_rouge()` call; returned as `rougeL`. |

### 1.4  rougeLsum

| Item | Detail |
|------|--------|
| **What it measures** | LCS computed at the **summary** level — splits the text on newlines/sentence boundaries first, computes LCS per sentence pair, then aggregates. |
| **Intuition** | When a summary has multiple sentences, rougeLsum treats each sentence independently before averaging. This avoids artificially low LCS from comparing one long concatenated string. |
| **Difference from ROUGE-L** | ROUGE-L treats the entire text as one sequence. rougeLsum splits on `\n` first, matching sentence-by-sentence. For single-sentence outputs the two are identical. |
| **Why report it** | CNN/DailyMail highlights are multi-sentence bullets. rougeLsum gives a fairer aggregate by not penalizing correct sentences that appear in a different global order. |
| **Computed by** | Same `compute_rouge()` call; returned as `rougeLsum`. |

---

## 2  Latency KPIs

All latency metrics are measured with `batch_size=1`, `torch.inference_mode()`, and a
configurable number of warmup runs discarded before recording begins.

### 2.1  Tokens/sec (p50) — Median Throughput

| Item | Detail |
|------|--------|
| **What it measures** | The 50th-percentile (median) of tokens generated per second across all measured samples. |
| **Formula** | For each sample: $\text{tok/s} = \frac{\text{number of generated tokens}}{\text{wall-clock time (s)}}$. p50 = `np.percentile(all_tok_per_sec, 50)`. |
| **Intuition** | The "typical" speed a user experiences. Half of all requests will be at least this fast. The median is preferred over the mean because a few outlier-slow generations (e.g., very long inputs that hit max context) do not skew it. |
| **Example** | Baseline Mistral-7B FP16 greedy: **174 tok/s** (from `base_mistral7bv03` perf report). This means a 100-token summary takes ~0.57 seconds. |
| **Why report it** | **Primary latency go/no-go metric.** Threshold: ≤ 20% regression vs FP16 baseline. If a quantized model is *slower* than FP16, something is broken (FP32 fallback, CPU offload, etc.). |
| **Computed by** | `measure_latency()` in `evaluate_perf.py`. |

### 2.2  Tokens/sec (p90)

| Item | Detail |
|------|--------|
| **What it measures** | 90th-percentile throughput — the speed below which 90% of samples fall. |
| **Intuition** | Captures the "fast tail." If p90 is much higher than p50, most samples generate quickly but a few are slow (likely long prompts). Conversely, if p90 ≈ p50, throughput is uniform. |
| **Why report it** | Helps diagnose input-length-dependent bottlenecks. A big gap between p50 and p90 after quantization may indicate that some layers are falling back to FP32. |
| **Computed by** | Same `measure_latency()` call. |

### 2.3  Tokens/sec (p95)

| Item | Detail |
|------|--------|
| **What it measures** | 95th-percentile throughput — the speed below which 95% of samples fall. |
| **Intuition** | Tail-end performance. Useful for capacity planning: if you commit an SLA of "200 tok/s," you want p95 ≥ 200 so only 5% of requests miss it. |
| **Why report it** | SLA-grade reporting. Even if p50 looks good, a low p95 means the model is inconsistent and may not be production-ready. |
| **Computed by** | Same `measure_latency()` call. |

### 2.4  Time-to-First-Token (TTFT) — ms

| Item | Detail |
|------|--------|
| **What it measures** | Wall-clock time from calling `model.generate()` to receiving the first decoded token, in milliseconds. |
| **How it's measured** | A `TextIteratorStreamer` runs generation in a separate thread. The moment the first chunk arrives, `time.perf_counter()` records the timestamp. $\text{TTFT} = (t_{\text{first\_token}} - t_{\text{start}}) \times 1000$ |
| **Intuition** | Users perceive latency from the moment they hit "send" to the moment text starts appearing. Even if overall tok/s is high, a 2-second blank screen feels slow. TTFT captures this "time to first impression." |
| **Example** | Baseline Mistral-7B FP16 greedy: **14.8 ms** TTFT. This is the prefill time — encoding the input prompt and producing the first output token. |
| **Why report it** | Critical for interactive applications (chatbots, copilots). A model with great tok/s but high TTFT (e.g., due to slow KV-cache initialization) feels sluggish. Reported under greedy mode only (beam search TTFT is not meaningful since all beams must prefill). |
| **Computed by** | `run_single_generation()` with `measure_ttft=True` in `evaluate_perf.py`. |

---

## 3  Memory KPIs

### 3.1  Peak Inference VRAM (GB)

| Item | Detail |
|------|--------|
| **What it measures** | The highest GPU memory allocation recorded during a single forward + generate pass. |
| **How it's measured** | `torch.cuda.reset_peak_memory_stats()` → run generation → `torch.cuda.max_memory_allocated() / 1024³` |
| **Intuition** | This is the worst-case GPU appetite. Even if the model weights are 13.5 GB, the peak may be higher (e.g., 14.8 GB) because of KV-cache, activations, and temporary buffers during generation. |
| **Example** | Baseline FP16: **14.79 GB** peak → the model weights (13.5 GB) plus ~1.3 GB of KV-cache and activations. |
| **Why report it** | Determines which GPU the model fits on. If peak VRAM exceeds GPU memory, OOM. **Go/no-go threshold for quantized models: ≤ 50% of FP16 peak.** A 4-bit quantized model should peak around 7 GB. |
| **Computed by** | `measure_memory()` in `evaluate_perf.py` → `peak_vram_gb`. |

### 3.2  Current Inference VRAM (GB)

| Item | Detail |
|------|--------|
| **What it measures** | GPU memory allocated at the moment of measurement (after generation completes). |
| **How it's measured** | `torch.cuda.memory_allocated() / 1024³` |
| **Intuition** | After generation, temporary buffers and KV-cache are freed. What remains is roughly the model weights + optimizer states (if any). The gap between peak and current tells you how much transient memory generation requires. |
| **Example** | Baseline: current = **13.51 GB** vs peak = 14.79 GB → ~1.3 GB was transient. |
| **Why report it** | Useful for multi-model serving: if two models share a GPU, you need to know the *steady-state* footprint, not just the peak. |
| **Computed by** | `measure_memory()` → `current_vram_gb`. |

### 3.3  Reserved VRAM (GB)

| Item | Detail |
|------|--------|
| **What it measures** | Total GPU memory reserved by PyTorch's caching allocator, including free blocks not yet returned to CUDA. |
| **How it's measured** | `torch.cuda.memory_reserved() / 1024³` |
| **Intuition** | PyTorch pre-allocates memory in large blocks for efficiency. Reserved > Allocated means PyTorch is holding onto extra memory "just in case." This gap grows on GPUs with lots of free memory. |
| **Example** | Baseline: reserved = **15.13 GB** vs allocated = 13.51 GB → PyTorch is caching ~1.6 GB of free blocks. |
| **Why report it** | If reserved VRAM is close to GPU capacity, other processes cannot allocate memory — even if actual usage is lower. Important for shared-GPU environments. |
| **Computed by** | `measure_memory()` → `reserved_vram_gb`. |

### 3.4  Peak Training VRAM (GB)

| Item | Detail |
|------|--------|
| **What it measures** | Maximum GPU memory consumed during a training step (forward + backward + optimizer update). |
| **How it's measured** | `memory_monitor.py` tracks `torch.cuda.max_memory_allocated()` across training steps via Trainer callbacks. |
| **Intuition** | Training consumes far more VRAM than inference because of gradient tensors, optimizer states (AdamW keeps two extra copies per parameter), and activation checkpoints. A 7B model that needs 14 GB for inference may need 28+ GB for full fine-tuning. LoRA/QLoRA dramatically reduce this. |
| **Why report it** | **Must fit a single GPU** — that is the hard pass/fail criterion. If training OOMs, you need a smaller batch size, more aggressive gradient checkpointing, or a different method (LoRA → QLoRA). |
| **Computed by** | `memory_monitor.py` → logged via `callbacks.py` during `train_lora.py` / `train_qlora.py` / `train_qat.py`. |

### 3.5  Model Disk Size (GB)

| Item | Detail |
|------|--------|
| **What it measures** | Total size of model weight files on disk. |
| **Formula** | $\text{size} = \sum_{p \in \text{params}} p.\text{numel}() \times p.\text{element\_size}()$ converted to GB ($/ 1024^3$). |
| **Intuition** | Disk size tracks the storage and download cost. A 4-bit GGUF of Mistral-7B is ~4 GB vs ~13.5 GB in FP16 — meaningful for edge deployment and cold-start time. |
| **Why report it** | Report-only (no threshold), but important for deployment planning. Smaller checkpoints mean faster model loading, cheaper storage, and faster cold starts in serverless environments. |
| **Computed by** | `measure_model_size()` in `evaluate_perf.py` → `model_size_gb`. |

### 3.6  Parameter Count (Billions)

| Item | Detail |
|------|--------|
| **What it measures** | Total number of learnable parameters in the model. |
| **Formula** | $\text{params} = \sum_{p \in \text{model.parameters()}} p.\text{numel}() \;/\; 10^9$ |
| **Intuition** | A sanity check. Mistral-7B should report ~7.25B. If a LoRA adapter reports 7.25B instead of a small fraction, something is wrong with the adapter loading. |
| **Why report it** | Validates that the correct model/adapter is loaded. Also useful for comparing trainable vs total params in LoRA reports. |
| **Computed by** | `measure_model_size()` → `param_count_billion`. |

---

## 4  Training KPIs

These metrics are logged step-by-step during LoRA, QLoRA, and QAT training runs via Trainer callbacks.

### 4.1  Training Loss

| Item | Detail |
|------|--------|
| **What it measures** | Cross-entropy loss on the training batch at each step. |
| **Intuition** | Should decrease monotonically (with noise). A sudden spike means a bad batch or learning-rate issue. If loss plateaus early, the model is under-fitting. |
| **Why report it** | Primary signal that training is progressing. Used to detect divergence, learning-rate problems, and data issues in real time. |

### 4.2  Validation Loss

| Item | Detail |
|------|--------|
| **What it measures** | Cross-entropy loss on a held-out validation set, measured at the end of each epoch. |
| **Intuition** | If val loss diverges from train loss, the model is overfitting. The checkpoint with lowest val loss is selected as the best. |
| **Why report it** | Drives early stopping and checkpoint selection. A model with low train loss but high val loss will perform poorly on the test-set ROUGE evaluation. |

### 4.3  Learning Rate Schedule

| Item | Detail |
|------|--------|
| **What it measures** | The effective learning rate at each training step. |
| **Intuition** | Verifies that warmup and decay are working as configured. A flat learning rate when you expected cosine decay means the scheduler is misconfigured. |
| **Why report it** | Diagnostic only. Lets you confirm the optimizer is behaving as intended, especially when using LoRA-specific learning rates. |

---

## 5  Summary Table

| Category | Metric | Key Insight | Go/No-Go? |
|----------|--------|-------------|-----------|
| Quality | ROUGE-1 (F1) | Word-level content coverage | ≤ 2 pt drop |
| Quality | ROUGE-2 (F1) | Phrase-level fluency | ≤ 2 pt drop |
| Quality | ROUGE-L (F1) | Sentence-level structural similarity | **≤ 2 pt drop** (primary) |
| Quality | rougeLsum | Multi-sentence aggregate | Report only |
| Latency | tok/s p50 | Typical user-perceived speed | **≤ 20% regression** |
| Latency | tok/s p90 | Fast-tail uniformity check | Report only |
| Latency | tok/s p95 | SLA-grade tail latency | Report only |
| Latency | TTFT (ms) | Responsiveness / time to first impression | Report only |
| Memory | Peak inference VRAM | Will it fit on the target GPU? | **≤ 50% of FP16** (quantized) |
| Memory | Current VRAM | Steady-state footprint for co-location | Report only |
| Memory | Reserved VRAM | Caching-allocator overhead | Report only |
| Memory | Peak training VRAM | Can we train on a single GPU? | Must not OOM |
| Memory | Model disk size | Storage / download / cold-start cost | Report only |
| Memory | Param count (B) | Sanity check on loaded model | Report only |
| Training | Train loss | Is training progressing? | Should decrease |
| Training | Val loss | Is the model overfitting? | Drives checkpoint selection |
| Training | Learning rate | Is the scheduler correct? | Diagnostic |

---

## 6  Where Metrics Are Saved

| Output | Path Pattern | Format |
|--------|-------------|--------|
| ROUGE scores | `outputs/reports/{run_name}_metrics.json` | `{"rouge1": 0.21, "rouge2": 0.05, ...}` |
| Latency + Memory | `outputs/reports/{run_name}_perf.json` | `{"greedy": {"tok_per_sec_p50": ...}, "peak_vram_gb": ...}` |
| Predictions | `outputs/predictions/{run_name}_preds.jsonl` | One JSON object per line with `prediction` and `reference` fields |
| Training logs | `outputs/runs/` | TensorBoard event files with loss, LR, memory per step |
