#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting PTQ Inference ==="
CUDA_VISIBLE_DEVICES=2 python -m src.finetuning.quantization.ptq_pipeline \
    --ptq_mode nf4

echo "=== Done ==="