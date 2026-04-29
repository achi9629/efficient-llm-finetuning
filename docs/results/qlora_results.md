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
| 2    | 4     | 0.2013  | 0.2630     | 69.79            | 10.11          | 5.88           | 13.50         |
| 4    | 8     | 0.2008  | 0.2617     | 66.04            | 7.71           | 5.88           | 12.98         |
| 8    | 16    | 0.1912  | 0.2532     | 67.20            | 9.92           | 5.88           | 11.64         |
| 16   | 32    | 0.1923  | 0.2526     | 68.00            | 9.97           | 5.88           | 11.85         |
| 32   | 64    | 0.1892  | 0.2540     | 68.63            | 9.97           | 5.88           | 12.48         |
| 64   | 128   | 0.1802  | 0.2429     | 69.66            | 8.89           | 5.88           | 11.86         |

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
