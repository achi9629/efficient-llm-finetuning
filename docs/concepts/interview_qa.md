# Interview Q&A — Efficient LLM Fine-Tuning Project

## The "Why" Narrative

"Most teams grab PEFT, call get_peft_model(), train with default rank=16, and ship. They never ask: is rank=16 right for my data? Is QLoRA actually cheaper than LoRA here? What's the real throughput cost of 4-bit inference?

I built this project to answer those questions with numbers — not intuition. I implemented LoRA from scratch to understand what PEFT abstracts away, then ran controlled experiments to measure the actual tradeoffs between adaptation methods and quantization strategies on a single-GPU budget."

---

## Key Findings (with numbers)

### Finding 1: Adaptation is not optional for task-specific quality

- Base Mistral-7B: 0.139 ROUGE-L
- LoRA (r=16, ~0.15% params): 0.183 ROUGE-L → +4.4 pt gain
- "LoRA doesn't teach new facts. It teaches the model your task's output distribution. That's why even r=2 works — the task-specific subspace is tiny."

### Finding 2: Rank is a dataset-size knob, not a quality knob

- r=2: 0.201 → r=64: 0.180 (monotonic decrease)
- "Rank controls capacity. If your dataset doesn't have enough signal to fill that capacity, the extra parameters fit noise. Same reason you don't use a 10-layer MLP for linear regression."

### Finding 3a: QLoRA saves both VRAM and training memory at a throughput cost

- During inference: Base/LoRA use ~14.5 GB (FP16), QLoRA uses ~5.83 GB (4-bit) — 60% VRAM reduction
- Greedy throughput: ~60% drop (70 vs 174 tok/s)
- Beam search: ~3x drop (10.1 vs 29.7 tok/s)
- "QLoRA is sold as 'same quality, less memory.' In practice, you're trading latency for memory. If your deployment is latency-bound, QLoRA is the wrong choice."

### Finding 3b: Training fit is an activation problem, not a weight problem

- LoRA FP16: 56.15 GB peak → QLoRA 4-bit: 45.46 GB → QLoRA + grad ckpt: 9.99 GB
- "Everyone thinks QLoRA fits on 24GB because the base model is 4-bit. It doesn't — activations still blow up. You need gradient checkpointing to actually fit. That's 82% VRAM reduction but 40% slower. The right question isn't 'does it fit?' — it's 'what's the cheapest way to make it fit?'"

### Finding 4: Scratch validates understanding, PEFT validates correctness

- Scratch LoRA: 0.183 vs PEFT LoRA: 0.191 (0.8 pt gap)
- "I use PEFT in production. But having built from scratch, I can debug it when PEFT does something unexpected."

### Finding 5: Training-time vs inference-time quantization are orthogonal

- QLoRA = training budget. PTQ/GPTQ = serving cost. You often want both.
- "When someone says 'we used quantization,' I ask: training-time or inference-time? They solve different bottlenecks."

---

## Common Interview Questions + Answers

### "Tell me about a project"

"I was investigating the real cost-quality tradeoffs of LLM adaptation on a single GPU. The surprising finding was that standard choices — rank=16, QLoRA for 'efficiency' — were suboptimal. Rank=2 outperformed rank=64 by 2 ROUGE-L points because our data was small enough that higher capacity overfits. QLoRA delivered real inference VRAM savings (5.83 vs 14.5 GB — 60% reduction) but at a 60% throughput penalty. I built a decision framework mapping deployment constraints to method choices."

### "Why not just use PEFT?"

"I do — I validated against it. But implementing from scratch means I can explain why alpha/r scaling matters, debug merge behavior, and make informed choices about target layers. When something breaks in production, I'm not stuck with a black box."

### "Why LoRA works?"

"Fine-tuning weight updates have low intrinsic rank. LoRA exploits this: instead of updating full W (d×k), you learn two small matrices B (d×r) and A (r×k) where r << min(d,k). For Mistral-7B with r=16, that's 131K params per layer vs 16.7M — 128x fewer. At inference, BA merges into W for zero overhead."

### "Why QLoRA needs double quantization?"

"QLoRA quantizes base weights to NF4 (4-bit NormalFloat). The quantization constants themselves take memory. Double quantization quantizes those constants too — FP32 constants → FP8 — saving ~0.4GB on a 7B model. It's quantizing the quantization metadata."

### "Why does quantization hurt accuracy?"

"You're reducing precision of weight representations. Information is lost. The key question is which weights matter most. Naive PTQ treats all weights equally — lossy. GPTQ uses calibration data to minimize reconstruction error on important weights. AWQ goes further by considering activation magnitudes. The more you know about what the model actually uses, the less you lose."

### "When would you choose LoRA vs QLoRA?"

"Depends on the bottleneck. If training VRAM is the constraint (fitting a 7B model on 16GB GPU), QLoRA's 4-bit base is necessary. If you have enough VRAM, LoRA gives better throughput (~2.5x faster) and simplifies the pipeline. On small datasets, QLoRA's implicit regularization from quantization can help — we saw this in our results. But I wouldn't generalize that claim."

### "What would you do differently?"

"Three things: (1) Run on a larger dataset to see where rank ordering reverses. (2) Add per-layer sensitivity analysis to find which transformer blocks break under INT4. (3) Test on Hindi where tokenizer fertility (~2-3x tokens/word) amplifies quantization degradation."

### "How does this connect to production?"

"LoRA adapters are ~few MB. You can serve multiple task-specific adapters sharing one base model — multi-tenant serving via vLLM or LoRAX. Separately, GPTQ/GGUF reduces per-request VRAM for more concurrent requests. And since adapters are separate files, you can hot-swap them without server restart."

### "How did you fit Mistral-7B training on a single 24GB GPU?"

"Three-layer approach: (1) 4-bit NF4 base via bitsandbytes — reduces model weights from 13.5GB to ~3.5GB. (2) Gradient checkpointing — recomputes activations instead of storing them, cutting activation memory by ~60%. (3) Gradient accumulation (4 steps) — keeps micro-batch size small. Without all three, peak VRAM was 45GB. With all three: 10GB. The tradeoff is 40% slower training — recomputing activations isn't free."

### "Why is QLoRA slower than LoRA if it uses less memory?"

"Two reasons: (1) Every forward/backward step dequantizes INT4 weights to BF16 before matmul — that's overhead per step. We measured 17% throughput drop (2,427 vs 2,847 tok/s). (2) When you add gradient checkpointing to fit on 24GB, you recompute activations on the backward pass — another ~30% slowdown. Total: QLoRA+GC is 40% slower than LoRA, but it's the only option that actually fits."

---

## Tricky Follow-ups

### "How does QLoRA reduce inference VRAM vs LoRA?"

"QLoRA loads the base model in 4-bit during inference, using ~5.83 GB vs ~14.5 GB for FP16 LoRA — a 60% reduction. The adapters are merged into the quantized model, keeping the 4-bit representation. The tradeoff is ~60% lower greedy throughput (70 vs 174 tok/s) due to dequantization overhead per step."

### "Why is beam search so much slower under QLoRA?"

"Beam search maintains multiple candidate sequences, multiplying KV cache and attention computation. Under quantized matmul (dequantize → compute → requantize per step), this overhead compounds. Greedy decoding does one forward pass per token; beam search does num_beams forward passes."

### "Is ROUGE-L the right metric here?"

"No single metric is sufficient. ROUGE-L measures longest common subsequence — it captures fluency and recall but misses semantic equivalence (paraphrases score poorly) and factual correctness. I'd add BERTScore for semantic similarity and human evaluation for factual accuracy. For cross-lingual work, chrF++ is better than ROUGE because it's character-based and handles morphologically rich languages."

### "How would you decide rank for a new task?"

"Start with a small sweep: {4, 8, 16, 32}. Plot quality vs rank. If quality plateaus or drops, you've found the ceiling. On small data (<10k samples), start lower. On large instruction-tuning datasets, r=32-64 is common. The key is: sweep, don't guess."

---

## Project Positioning & Strategy Questions

### "What's the ROI of this project? Why should we care?"

"This project answers a question every ML team faces: given a fixed GPU budget, what's the cheapest way to adapt an LLM without killing quality or latency? I measured it instead of guessing. The decision framework I built maps real deployment constraints — memory-bound vs latency-bound vs no-training-budget — to concrete method recommendations backed by numbers."

### "Why this project over something else?"

"Fine-tuning and quantization are the two most common production LLM operations. Every team does them, but most cargo-cult their choices (rank=16, QLoRA by default). I wanted to build the judgment to make those choices intentionally. That's more valuable than knowing one more framework."

### "What makes this project different from a tutorial?"

"Three things: (1) I implemented LoRA from scratch and validated against PEFT — I don't just call APIs. (2) I ran controlled experiments with fixed variables — same data, same model, same eval — so comparisons are fair. (3) My conclusions include caveats about generalizability. I don't claim QLoRA is always better; I explain when and why it was better in this specific setting."

### "Walk me through a design decision you made"

"Choosing which layers to target with LoRA. The original paper targets q_proj and v_proj only. I targeted all four attention projections (q, k, v, o). Why? GQA in Mistral means k_proj and v_proj are smaller (4096×1024 vs 4096×4096), so adding them costs little. And empirically, covering all attention weights gave better quality than the minimal set. I didn't add MLP layers — that doubles parameters with diminishing returns on summarization."

### "How would you scale this to a team setting?"

"Three changes: (1) Config-driven experiments — all hyperparameters in YAML, not hardcoded. Already done. (2) Standardized eval pipeline — same metrics, same test set, same reporting format. Already done. (3) Artifact tracking — experiment names encode model, dataset, timestamp. Already done. A new team member can run a sweep, compare against baselines, and make a decision without reading my code."

### "Your QLoRA without checkpointing used 45GB. That's more than FP16 LoRA's 56GB?"

"No, 45 < 56. QLoRA's 4-bit base saves ~10GB on weights. But both exceed 24GB because activations dominate — they scale with batch_size × seq_len × hidden_dim, not model precision. The insight is that model compression alone doesn't solve the training memory problem; activation memory management (checkpointing, micro-batching) is what unlocks single-GPU training."

---

## System Design Questions

### "Design a system to serve fine-tuned models to 100 customers with different tasks"

"One base model in GPU memory. Per-customer LoRA adapters stored on disk (~5-10 MB each). On request: load adapter, apply to base model logits, generate response, unload. This is what vLLM's LoRA support and LoRAX do. Key constraints: adapter loading latency (<100ms), memory for active adapters, eviction policy for inactive adapters. GPTQ-quantize the base model to fit more concurrent requests per GPU."

### "You have 8GB VRAM. How do you fine-tune a 7B model?"

"QLoRA. 4-bit base (~3.5 GB) + LoRA adapters in FP16 (~few MB) + optimizer states + activations. Use gradient checkpointing to trade compute for memory on activations. Use paged optimizers (bitsandbytes) to offload optimizer states to CPU. With all three, 7B fits in 8GB for training. For inference, GGUF Q4_K_M runs in ~4GB."

### "Your model quality dropped after quantization. How do you debug it?"

"Step 1: Which quantization method? PTQ is lossy by construction — check if GPTQ with calibration data recovers quality. Step 2: Which layers broke? Run per-layer sensitivity — quantize one layer at a time, measure quality delta. Attention layers in early/late blocks are usually most sensitive. Step 3: If specific layers are the problem, use mixed precision — keep sensitive layers in FP16, quantize the rest. Step 4: If PTQ can't recover, QAT on the sensitive layers only — don't retrain the whole model."

### "How would you evaluate if a fine-tuned model is production-ready?"

"Four gates: (1) Quality gate — ROUGE-L within 2 pts of best checkpoint on held-out test set. (2) Latency gate — tok/s p50 within 20% of FP16 baseline. (3) Memory gate — peak VRAM under deployment budget. (4) Robustness gate — no quality collapse on long inputs (2k+ tokens) or edge cases. All four must pass. If quality passes but latency fails, the model isn't ready — it's a different kind of broken."

### "LoRA vs full fine-tuning — when do you choose which?"

"LoRA when: (1) you need multiple task-specific models sharing one base (multi-tenant), (2) training VRAM is limited, (3) dataset is small-to-medium (<100k samples), (4) you want fast iteration. Full FT when: (1) you're training a foundation model or doing major domain adaptation, (2) you have the compute budget, (3) quality ceiling from LoRA isn't sufficient. For most production use cases — task adaptation, instruction tuning — LoRA is the right default."

### "Design a training pipeline for a team that fine-tunes 7B models on 24GB GPUs"

"Three-tier memory strategy: (1) 4-bit base model via bitsandbytes (~3.5GB). (2) Gradient checkpointing to cap activation memory (~6GB instead of ~40GB). (3) Gradient accumulation to keep micro-batch small while maintaining effective batch size. Our measured peak was 9.99GB — leaves headroom for longer sequences or larger ranks. The cost is 40% slower wall-clock time (12.9 vs 7.8 min). For a team, I'd add: config-driven sweeps (YAML), automated _train_perf.json logging per run, and a comparison script that reads all reports and builds the cost table automatically."

### "Your training used 56GB for LoRA. How would you reduce it without changing the method?"

"Three levers, in order of impact: (1) Gradient checkpointing — saves ~80% activation memory by recomputing on backward pass. We went from 56GB to ~10GB. (2) Reduce micro-batch size and increase gradient accumulation — same effective batch, less peak memory. (3) Mixed-precision training (bf16) — already enabled, but if using fp32, switching halves activation memory. (4) CPU offloading via DeepSpeed ZeRO-Offload — moves optimizer states to RAM. I'd use (1) first because it's one flag with the biggest impact."

### "You need to fine-tune 10 different LoRA adapters. How do you schedule training on 4 GPUs?"

"Each adapter trains independently — no inter-GPU communication needed. Simple job queue: 10 jobs, 4 workers. Each job: load 4-bit base + grad checkpointing → fits 24GB GPU → train → save adapter + _train_perf.json. At 13 min/job, all 10 finish in ~35 min (ceil(10/4) × 13). No need for distributed training here — LoRA adapters share no state. I'd use a SLURM array job or a simple Python multiprocessing pool. If training time matters more than GPU cost, you can skip grad checkpointing on 80GB A100s and finish in ~8 min/job."
