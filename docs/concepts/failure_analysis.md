# Failure Analysis — Efficient LLM Fine-Tuning

Documenting breakages, surprises, and non-obvious behaviors encountered during experimentation.
All findings are specific to Mistral-7B-v0.3 on CNN/DailyMail summarization.

---

## 1. Higher LoRA rank degraded quality

**Observed:** Rank sweep r={2,4,8,16,32,64} showed monotonic ROUGE-L decrease. r=2 → 0.201, r=64 → 0.180.

**Expected:** Higher rank = more capacity = better or equal quality.

**Root cause:** Dataset too small (~1.6k training samples) to fill the capacity of high-rank adapters. Extra parameters fit noise rather than signal — classic overfitting without visible train loss divergence.

**Fix/Takeaway:** Always sweep rank. On small datasets (<10k), start from r=2-4. Higher rank needs proportionally more data to justify the capacity.

---

## 2. QLoRA without gradient checkpointing doesn't fit 24GB

**Observed:** QLoRA (4-bit base) peaked at 45.46 GB VRAM. LoRA (FP16 base) peaked at 56.15 GB. Neither fits on a 24GB GPU.

**Expected:** "QLoRA fits on consumer GPUs" is the common claim.

**Root cause:** Activation memory dominates during training, not model weights. Activations scale with batch_size × seq_len × hidden_dim × num_layers. With avg seq_len=838 and bs=4, activations alone exceed 24GB regardless of weight precision.

**Fix:** Gradient checkpointing reduces peak VRAM to 9.99 GB (82% reduction) by recomputing activations during backward pass instead of storing them. Cost: 40% slower wall-clock time.

**Takeaway:** When people say "QLoRA fits on 24GB," they implicitly mean QLoRA + gradient checkpointing + small batch. The 4-bit base alone is not sufficient.

---

## 3. Scratch LoRA vs PEFT gap (~0.9 pt ROUGE-L)

**Observed:** Scratch LoRA: 0.183 vs PEFT LoRA: 0.191 ROUGE-L with identical hyperparameters.

**Expected:** Same math → same results.

**Root cause:** Likely differences in: (1) initialization variance (Kaiming vs PEFT's default), (2) gradient computation through the frozen base (PEFT uses `requires_grad=False` uniformly; scratch implementation may have subtle differences in which parameters get gradients), (3) numerical precision in the scaling factor application.

**Fix:** Gap is small enough to validate correctness. For production, use PEFT. The scratch implementation's value is understanding, not deployment.

**Takeaway:** Custom implementations need validation against reference libraries. "Close enough" needs a threshold — here, <1pt ROUGE-L is acceptable.

---

## 4. QLoRA beam search throughput collapse

**Observed:** Beam search throughput dropped ~3x under QLoRA (10.1 vs 29.7 tok/s) — much worse than the 60% greedy penalty.

**Expected:** Throughput penalty should be constant across decoding strategies.

**Root cause:** Beam search multiplies forward passes by num_beams. Each forward pass dequantizes INT4→BF16 weights. So dequantization overhead is amplified by beam width. Additionally, KV cache grows per beam, increasing memory pressure.

**Fix:** If deploying with beam search, prefer FP16/GPTQ over QLoRA for inference. QLoRA's value is training-time memory, not inference-time speed.

**Takeaway:** Always benchmark your actual decoding strategy, not just greedy. Production pipelines using beam search/sampling will see amplified quantization overhead.

---

## 5. 4-bit quantization adds hidden throughput tax during training

**Observed:** QLoRA training throughput was 17% slower than LoRA (2,427 vs 2,847 tok/s), even without gradient checkpointing.

**Expected:** 4-bit base should be faster — fewer bytes to move.

**Root cause:** bitsandbytes dequantizes NF4→BF16 on every forward and backward pass. This per-step overhead outweighs the memory bandwidth savings from smaller weights. The dequantization is on the compute path, not just the memory path.

**Fix:** Accept as a known cost. If training speed matters more than VRAM, use FP16 LoRA on a GPU with enough memory.

**Takeaway:** Quantization reduces memory, not compute. Throughput and memory are independent axes — optimizing one can hurt the other.

---

## 6. _train_perf.json not saving (debugged)

**Observed:** `save_training_report()` was called but no JSON was written. No error raised.

**Root cause:** Originally retrieved the callback via `isinstance` check on `trainer.callback_handler.callbacks`. HF Trainer wraps callbacks internally, so `isinstance` failed silently. `training_summary` was never set on the retrieved object because it was a different instance.

**Fix:** Hold a direct reference to the `GPUMemoryCallback` instance (`gpu_cb`) before passing it to Trainer, then call `gpu_cb.save_training_report()` after training. Don't try to fish it back out of the Trainer.

**Takeaway:** HF Trainer's callback handler may wrap or copy callbacks. Never retrieve callbacks by type after passing them in — keep your own reference.

---

## 7. Custom metrics missing from trainer_state.json

**Observed:** `samples_per_sec` and `peak_gpu_memory_gb` appeared in console logs but not in `trainer_state.json` log_history.

**Root cause:** HF Trainer copies the `logs` dict into `state.log_history` *before* calling `on_log` callbacks. So modifications to `logs` inside `on_log` are visible in console output (WandB/TensorBoard loggers) but not persisted to state.

**Fix:** At the end of `on_log`, explicitly update `state.log_history[-1]` with the custom metrics:

```python
if state.log_history:
    state.log_history[-1].update({k: v for k, v in logs.items()
                                  if k not in state.log_history[-1]})
```

**Takeaway:** HF Trainer's callback ordering has subtle timing issues. on_log receives a copy-on-write dict — modifications to it are ephemeral unless you also patch state.log_history directly.

---

## 8. GPTQ base evaluation loaded wrong checkpoint

**Observed:** Initial GPTQ "base" benchmark showed 83.7 tok/s greedy — suspiciously close to the LoRA-merged GPTQ result (84.5 tok/s). Re-running with corrected paths gave 34.0 tok/s.

**Root cause:** `ptq_config.yaml` `checkpoint_dir` for GPTQ was updated to point to the LoRA-merged quantized folder during the merge experiments. The "base" eval script used this config path, silently loading the LoRA-merged model instead of base Mistral.

**Fix:** Verified by re-running with explicit base model path. Updated config and results table with correct numbers.

**Takeaway:** Config files are mutable state. When switching between base and fine-tuned evaluation, validate which checkpoint is actually loaded — log the model path at load time. A single wrong `checkpoint_dir` can silently corrupt all downstream benchmarks.

---

## 9. AWQ .device attribute error on AutoAWQForCausalLM

**Observed:** `inference_utils.py` line 102 calls `.to(model.device)` on inputs. AWQ's `AutoAWQForCausalLM` wrapper doesn't expose a `.device` attribute like HuggingFace models do.

**Root cause:** `AutoAWQForCausalLM` wraps the actual `model` object. The HF model lives at `model.model`, not `model` directly. The `.device` attribute exists on the inner model, not the wrapper.

**Fix:** Use `next(model.parameters()).device` or unwrap via `model.model.device`. This is framework-agnostic and works for both HF and AWQ models.

**Takeaway:** Third-party quantization wrappers (AutoAWQ, AutoGPTQ) don't always expose the same API as HuggingFace `AutoModelForCausalLM`. Always check the wrapper's interface — don't assume duck typing works.

---

## 10. Throughput (tok/s) not comparable across base vs fine-tuned models

**Observed:** LoRA-merged PTQ INT8 showed 109.5 tok/s vs base PTQ INT8 at 40.1 tok/s — a 2.7× gap on the same A100 GPU, same quantization, same benchmark script.

**Expected:** Same quantization method + same hardware = similar throughput.

**Root cause:** The throughput formula is `generated_tokens / total_time` (including prefill). Fine-tuned models generate concise 30-60 token bullet summaries (early EOS). Base model generates verbose 200+ token paragraphs (hits max_new_tokens=256). Different token counts change how the fixed prefill cost is amortized, and longer generations may trigger `no_repeat_ngram_size` penalties that slow decode.

**Fix:** Not a bug — this is the correct measurement of real-world throughput. For hardware-level decode speed comparisons, one would need `min_new_tokens = max_new_tokens` to force identical generation lengths. But that's artificial.

**Takeaway:** When benchmarking quantization methods, compare within the same model family (base vs base, or LoRA vs LoRA). Cross-family tok/s comparisons are confounded by generation behavior differences.

---

## 11. PEFT `PEFT_TYPE_TO_MODEL_MAPPING` attribute missing

**Observed:** `ImportError` / `AttributeError` when loading PEFT models with certain peft versions: `peft.peft_model` has no attribute `PEFT_TYPE_TO_MODEL_MAPPING`.

**Root cause:** PEFT library version mismatch. Some versions removed or relocated this mapping.

**Fix:** Added shim at import time:

```python
if not hasattr(peft.peft_model, 'PEFT_TYPE_TO_MODEL_MAPPING'):
    peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING = {}
```

**Takeaway:** Pin library versions in requirements.txt. When using multiple quantization libraries (bitsandbytes, auto_gptq, autoawq, peft, optimum), version conflicts are inevitable. Add defensive shims for known breakages.

---

## 12. NF4 quantized model faster than FP16 for autoregressive decode

**Observed:** LoRA-merged + PTQ NF4 ran at 285.2 tok/s greedy — 61% faster than the same LoRA-merged model at FP16 (177.2 tok/s). Same model weights, same prompts, same A100 GPU.

**Expected:** Quantization adds dequantization overhead → should be slower.

**Root cause:** Autoregressive decoding (batch_size=1, one token at a time) is **memory-bandwidth-bound**, not compute-bound. Each decode step reads the entire weight matrix from HBM but only does a single vector-matrix multiply. NF4 weights are 3.25× smaller than FP16 → 3.25× fewer bytes read per step → less time waiting on HBM bandwidth. The dequantization ALU cost is fully hidden behind the memory transfer on A100 (which has high compute:bandwidth ratio).

**Why INT8 doesn't show the same speedup:** LLM.int8() uses mixed-precision decomposition — outlier features are computed in FP16, rest in INT8. The decomposition/scatter logic adds compute overhead that isn't memory-bound, negating the bandwidth savings.

**Fix:** Not a bug — this is correct behavior. NF4 is genuinely faster for single-request autoregressive serving. The "quantization = slower" intuition only holds for compute-bound workloads (large batches, prefill phase).

**Takeaway:** For single-request serving, 4-bit quantization wins on both VRAM and latency. The tradeoff only becomes real at high batch sizes where the workload becomes compute-bound and dequantization overhead is no longer hidden.
