#!/bin/bash
# scripts/run_baseline.sh — Run base model inference + evaluation

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running base model inference ==="
python -m src.finetuning.inference.infer_base

echo "=== Done ==="