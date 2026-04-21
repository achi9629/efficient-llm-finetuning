#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting QLoRA Training ==="
CUDA_VISIBLE_DEVICES=3 python -m src.finetuning.training.train_qlora
# torchrun --nproc_per_node=4 -m src.finetuning.training.train_lora

echo "=== QLoRA Training Complete ==="