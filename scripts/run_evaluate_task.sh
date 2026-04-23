#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root
export HF_HUB_DISABLE_XET=1
echo "=== Running base model inference ==="
# CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.evaluation.evaluate_task \
#     --adapter_type lora_scratch
CUDA_VISIBLE_DEVICES=3 python -m src.finetuning.evaluation.evaluate_task \
    --r 64 \
    --alpha 128 \
    --lr 0.0002 \
    --adapter_type qlora

echo "=== Done ==="