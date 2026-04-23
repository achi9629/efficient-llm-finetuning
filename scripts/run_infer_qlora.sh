#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running QLora model inference ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.inference.infer_qlora \
    --adapter_path outputs/checkpoints/qlora_r64_a128_lr0.0002/lora_adapter.pt \

echo "=== Done ==="