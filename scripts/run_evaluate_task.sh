#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root
export HF_HUB_DISABLE_XET=1
echo "=== Running model inference ==="
CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.evaluation.evaluate_task \
    --r 64 \
    --alpha 128 \
    --lr 0.0002 \
    --adapter_type lora_peft

echo "=== Done ==="