#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting PEFT LoRA Training ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.training.train_lora_peft
# torchrun --nproc_per_node=4 -m src.finetuning.training.train_lora_peft

echo "=== PEFT LoRA Training Complete ==="