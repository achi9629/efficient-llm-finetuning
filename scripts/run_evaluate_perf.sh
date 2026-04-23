#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running base model inference ==="
# CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.evaluation.evaluate_perf \
#     --adapter_type lora_scratch
CUDA_VISIBLE_DEVICES=2 python -m src.finetuning.evaluation.evaluate_perf \
    --r 64 \
    --alpha 128 \
    --lr 0.0002 \
    --adapter_type qlora
echo "=== Done ==="