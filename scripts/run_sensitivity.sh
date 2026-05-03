#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting Per Layer Sensitivity Analysis ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.quantization.sensitivity \
    --adapter_path outputs/checkpoints/lora_peft_r2_a4_lr0.0002_20260501_101659/

echo "=== Done ==="