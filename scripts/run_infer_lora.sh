#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running Lora from Scratch model inference ==="
CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.inference.infer_lora \
    --adapter_path outputs/checkpoints/lora/lora_adapter.pt \
    --adapter_type scratch

echo "=== Done ==="