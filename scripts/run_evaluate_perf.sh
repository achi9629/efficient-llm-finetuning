#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running model inference ==="
CUDA_VISIBLE_DEVICES=3 python -m src.finetuning.evaluation.evaluate_perf \
    --r 32 \
    --alpha 64 \
    --lr 0.0002 \
    --adapter_type gptq_int4
echo "=== Done ==="