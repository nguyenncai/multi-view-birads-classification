#!/bin/bash

# --- Usage Example ---
# bash run_eval_vindr.sh /path/to/your/checkpoint/best_model_f1.pth
# --------------------

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/fit02/miniconda3/bin/python}"

# Requires checkpoint as argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_checkpoint.pth>"
    exit 1
fi

CHECKPOINT=$1
OUTPUT_DIR=$(dirname "$CHECKPOINT")/evaluation_vindr

# VinDr Dataset Paths
CSV_PATH="/home/fit02/nguyen_workspace/breast-level_annotations.csv"
IMG_DIR="/home/fit02/nguyen_workspace/vindr/cleaned_images"

echo "========================================="
echo " EVALUATING DUAL-BRANCH ON VINDR DATASET"
echo "========================================="
echo "Dataset Type: vindr (split: test)"
echo "CSV Path   : $CSV_PATH"
echo "Image Dir  : $IMG_DIR"
echo "Checkpoint : $CHECKPOINT"
echo "Output Dir : $OUTPUT_DIR"
echo "========================================="

echo "1) Standard Evaluation"
"$PYTHON_BIN" evaluate_dual_branch.py \
    --dataset_type vindr \
    --csv_path "$CSV_PATH" \
    --img_dir "$IMG_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 2

echo ""
echo "========================================="
echo "2) TTA Evaluation (Test-Time Augmentation)"
"$PYTHON_BIN" evaluate_dual_branch.py \
    --dataset_type vindr \
    --csv_path "$CSV_PATH" \
    --img_dir "$IMG_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 2 \
    --use_tta

echo ""
echo "Done!"
