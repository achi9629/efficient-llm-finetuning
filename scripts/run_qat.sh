#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting QAT Fine-tuning ==="
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore::FutureWarning" python -m src.finetuning.quantization.qat_pipeline \
    --lora_model_path assets/models/lora_peft_merged_r2_mistral_7b_v0.3

echo "=== Fine-tuning Complete ==="