#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting GPTQ Inference ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.quantization.ptq_pipeline \
    --ptq_mode gptq_int4

echo "=== Done ==="