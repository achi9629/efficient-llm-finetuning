# QLoRA Rank Sweep Results (Day 6)

## Setup

- Model: Mistral-7B-v0.3
- Dataset: CNN/DailyMail
- Method: QLoRA with Scratch LoRA (4-bit base + LoRA adapters)
- Learning rate: 0.0002
- Sweep: rank/alpha = {2/4, 4/8, 8/16, 16/32, 32/64, 64/128}

## Results (Score, Memory, Latency)

| Rank | Alpha | ROUGE-L | ROUGE-Lsum | Greedy tok/s p50 | Beam tok/s p50 | Peak VRAM (GB) | TTFT avg (ms) |
|------|------:|--------:|-----------:|-----------------:|---------------:|---------------:|--------------:|
| 2    | 4     | 0.2013  | 0.2630     | 130.17           | 15.54          | 14.53          | 15.56         |
| 4    | 8     | 0.2008  | 0.2617     | 124.04           | 15.70          | 14.53          | 15.72         |
| 8    | 16    | 0.1912  | 0.2532     | 127.76           | 15.49          | 14.53          | 15.69         |
| 16   | 32    | 0.1923  | 0.2526     | 126.52           | 15.34          | 14.53          | 16.06         |
| 32   | 64    | 0.1892  | 0.2540     | 129.09           | 15.43          | 14.53          | 16.09         |
| 64   | 128   | 0.1802  | 0.2429     | 126.30           | 15.19          | 14.53          | 16.22         |

## Best QLoRA Model Selection

Selected run:

- qlora_r2_a4_lr0.0002_mistral7bv03_cnndaily_20260423_164506

Why this model was selected:

1. Highest quality in the sweep (best ROUGE-L and ROUGE-Lsum).
2. Best greedy throughput among all ranks.
3. Same peak VRAM as other ranks in this setup, so higher ranks do not provide a memory advantage.
4. Lowest TTFT in the sweep, giving the best responsiveness.

## Practical Interpretation

- In this training regime, lower-rank adapters performed better than higher-rank variants.
- Increasing rank beyond 4 did not improve quality and mostly added complexity without memory benefit.
- Since VRAM is constant across these runs, quality + latency become the deciding factors, and both favor rank 2.

## Finalization

The Day 6 final QLoRA candidate checkpoint is:

- qlora_r2_a4_lr0.0002

For the Day 6 four-way comparison table (Base vs Scratch LoRA vs PEFT LoRA vs QLoRA) w ewill use this model as final QLora model
