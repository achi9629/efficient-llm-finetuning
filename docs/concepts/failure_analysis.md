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
