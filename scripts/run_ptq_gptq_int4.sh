#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting GPTQ Inference ==="
CUDA_VISIBLE_DEVICES=2 python -m src.finetuning.quantization.ptq_pipeline \
    --ptq_mode gptq_int4 \
    --adapter_type lora_peft \
    --adapter_path outputs/checkpoints/lora_peft_r2_a4_lr0.0002_20260501_101659/ \
    --lora_saving_path assets/models/lora_peft_merged_r2_mistral_7b_v0.3

echo "=== Done ==="