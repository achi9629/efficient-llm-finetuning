#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting GPTQ Inference ==="
CUDA_VISIBLE_DEVICES=0 python -m src.finetuning.quantization.ptq_pipeline \
    --ptq_mode gptq_int4 \
    --adapter_type qat_lora_peft \
    --adapter_path outputs/checkpoints/qat_block3_module2_4bits_steps250_20260506_183257/ \
    --lora_saving_path assets/models/GPTQ_qat_block3_module2_4bits_steps250_lora_peft_merged_r2_mistral_7b_v0.3

echo "=== Done ==="