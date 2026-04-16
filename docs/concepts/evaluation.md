# Evaluation KPI Thresholds

## Project Context

- **Model**: Mistral-7B-v0.3 (`mistralai/Mistral-7B-v0.3`)
- **Task**: Abstractive summarization
- **Dataset**: CNN/DailyMail 3.0.0 (`article` → `highlights`)
- **Baseline**: FP16 inference on the base (un-finetuned) model

---

## KPI

- **Key Performance Indicator** — a measurable number that tells you whether an experiment succeeded or failed.

| Category | What it answers             | Example                                                                          |
|----------|-----------------------------|----------------------------------------------------------------------------------|
| Quality  | "Did the model get dumber?" | ROUGE-L dropped 1.5 pts after quantization → PASS (threshold is 2)               |
| Latency  | "Did it get slower?"        | Tokens/sec went from 40 → 35 after PTQ → PASS (12.5% drop, threshold is 20%)     |
| Memory   | "Does it fit on the GPU?"   | VRAM went from 14 GB (FP16) → 6 GB (4-bit) → PASS (57% reduction, target is 50%) |

---

## Quality KPIs

All quality metrics are computed on the **test split** (`test[:1000]`).

| Metric             | Library                    | Go/No-Go Threshold                            |
|--------------------|----------------------------|-----------------------------------------------|
| ROUGE-1 (F1)       | `rouge_score`              | ≤ 2 pts drop from best fine-tuned checkpoint  |
| ROUGE-2 (F1)       | `rouge_score`              | ≤ 2 pts drop from best fine-tuned checkpoint  |
| ROUGE-L (F1)       | `rouge_score`              | ≤ 2 pts drop from best fine-tuned checkpoint  |
| Qualitative review | Manual, 20 samples per run | No hallucinations, no mid-sentence truncation |

### Interpretation

- **Fine-tuned checkpoints** (LoRA, QLoRA): compared against un-finetuned base model. Higher is better.
- **Quantized variants** (PTQ, QAT, GGUF): compared against the best fine-tuned checkpoint. Drop must stay within threshold.

---

## Latency KPIs

Measured via timed generation over the test split with batch size = 1.

| Metric                   | How Measured                                  | Go/No-Go Threshold                |
|--------------------------|-----------------------------------------------|-----------------------------------|
| Tokens/sec (p50)         | `time.perf_counter` around `model.generate()` | ≤ 20% regression vs FP16 baseline |
| Tokens/sec (p90)         | Same                                          | Report only                       |
| Tokens/sec (p95)         | Same                                          | Report only                       |
| Time-to-first-token (ms) | Time from call to first token                 | Report only                       |

### Notes

- Quantized models should be **faster** than FP16. A regression means something is wrong (fallback to FP32 ops, CPU offload, etc.).
- Always warm up with 5 dummy generations before timing.

---

## Memory KPIs

| Metric                   | How Measured                        | Go/No-Go Threshold               |
|--------------------------|-------------------------------------|----------------------------------|
| Peak inference VRAM (GB) | `torch.cuda.max_memory_allocated()` | Quantized ≤ 50% of FP16 baseline |
| Peak training VRAM (GB)  | `memory_monitor.py` during training | Must fit single GPU              |
| Model disk size (GB)     | `os.path.getsize()` on checkpoint   | Report only                      |

---

## Go/No-Go Decision Rules

1. **PASS**: Method meets ALL quality thresholds AND does not exceed latency regression limit.
2. **FLAG**: Quality drop > 2 pts ROUGE-L → method gets one tuning iteration. If still fails, reject.
3. **REJECT**: Latency regression > 20% with no quality gain → do not promote to next stage.
4. **OVERRIDE**: If a method misses one threshold but excels on others (e.g., 3 pt ROUGE-L drop but 3x speedup), document the tradeoff explicitly and flag for manual review.

---

## When Each KPI Is Measured

| Day | Checkpoint          | KPIs Collected                      |
|-----|---------------------|-------------------------------------|
| 2   | Base model baseline | ROUGE-1/2/L, tok/s, VRAM, disk size |
| 4   | Best LoRA           | ROUGE-1/2/L, training VRAM, tok/s   |
| 6   | Best QLoRA          | ROUGE-1/2/L, training VRAM, tok/s   |
| 8   | LLM PTQ             | ROUGE-L drop, tok/s gain, VRAM      |
| 10  | CNN INT8 PTQ        | Accuracy drop, p95, throughput      |
| 13  | GGUF variants       | ROUGE-L, tok/s, disk size           |
| 19  | CNN QAT INT8        | Recovered accuracy, p95, throughput |
