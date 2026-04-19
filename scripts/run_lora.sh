#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting LoRA Training ==="
CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.training.train_lora
# torchrun --nproc_per_node=4 -m src.finetuning.training.train_lora

echo "=== LoRA Training Complete ==="