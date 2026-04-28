#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Starting PTQ Inference ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.quantization.ptq_pipeline \
    --ptq_mode int8

echo "=== Done ==="