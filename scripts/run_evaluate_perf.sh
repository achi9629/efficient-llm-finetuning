#!/bin/bash

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running base model inference ==="
CUDA_VISIBLE_DEVICES=1 python -m src.finetuning.evaluation.evaluate_perf

echo "=== Done ==="