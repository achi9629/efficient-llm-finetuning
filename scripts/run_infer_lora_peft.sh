#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running Lora peft model inference ==="
CUDA_VISIBLE_DEVICES=3 python -m src.finetuning.inference.infer_lora \
    --adapter_path outputs/checkpoints/lora_peft/ \
    --adapter_type peft

echo "=== Done ==="