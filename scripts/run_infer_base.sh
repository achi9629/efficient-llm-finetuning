#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running base model inference ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.inference.infer_base \
    --model_path outputs/checkpoints/qat_block3_module2_4bits_steps250_20260506_183257 \
    --adapter_type qat

echo "=== Done ==="