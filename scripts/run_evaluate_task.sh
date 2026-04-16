#!/bin/bash
# scripts/run_evaluate_task.sh - Run evaluation for a task.

set -e # exit on error

cd "$(dirname "$0")/.." # change to project root

echo "=== Running base model inference ==="
python -m src.finetuning.evaluation.evaluate_task

echo "=== Done ==="