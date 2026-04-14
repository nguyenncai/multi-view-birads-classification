#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/fit02/miniconda3/bin/python}"
CSV_PATH="/home/fit02/nguyen_workspace/breast-level_annotations.csv"
IMG_DIR="/home/fit02/nguyen_workspace/vindr/cleaned_images"
OUTPUT_DIR="./runs/dual_branch_ensemble_dinov2_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT=""

echo "========================================="
echo "DUAL-BRANCH ENSEMBLE - DINOv2 v3"
echo "========================================="
echo "Architecture: CC Branch + MLO Branch"
echo "Backbone: ViT-Large DINOv2 (304M params × 2)"
echo "Image size: 518 (native DINOv2 resolution)"
echo "4 views: R_CC, R_MLO, L_CC, L_MLO"
echo "========================================="
echo "Paper-based optimizations (2024-2025):"
echo "  [P0] Dual-Branch Ensemble (MDPI 2025)"
echo "       → CC & MLO branches process independently"
echo "       → Average predictions = ensemble"
echo "  [P1] Layer-wise LR decay (NVIDIA TAO)"
echo "  [P2] Heavy augmentation (±12° rot, ±8% trans)"
echo "  [P3] All v2 improvements retained"
echo "========================================="
echo "Expected improvement: +10-15% F1 vs v2"
echo "  v1 best: F1=0.529"
echo "  v2 expected: F1=0.57-0.61"
echo "  v3 target: F1=0.62-0.68"
echo "========================================="

"$PYTHON_BIN" train_dual_branch.py \
    --csv_path "$CSV_PATH" \
    --img_dir "$IMG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --img_size 518 \
    --backbone_name vit_large_patch14_dinov2 \
    --classifier_dropout 0.35 \
    --pretrained \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --epochs 35 \
    --lr 2e-5 \
    --backbone_lr_multiplier 0.2 \
    --layerwise_lr_decay 0.9 \
    --weight_decay 5e-5 \
    --num_workers 0 \
    --warmup_epochs 3 \
    --warmup_start_factor 0.2 \
    --early_stopping 12 \
    --focal_weight 0.15 \
    --focal_gamma 1.5 \
    --label_smoothing 0.05 \
    --ordinal_weight 0.20 \
    --augmentation_profile heavy \
    --normalization_profile imagenet \
    --preprocess_profile mammodino \
    --sampler sqrt_weighted \
    --class_weights_method inverse_sqrt \
    --use_ema \
    --ema_decay 0.9995 \
    --use_grad_checkpointing

echo ""
echo "Training completed!"
echo "Results: $OUTPUT_DIR/training_results.csv"
echo "Best F1 model: $OUTPUT_DIR/best_model_f1.pth"
echo "Best AUC model: $OUTPUT_DIR/best_model_auc.pth"
echo ""

# TTA Evaluation (if best F1 model exists)
if [ -f "$OUTPUT_DIR/best_model_f1.pth" ]; then
    CHECKPOINT="$OUTPUT_DIR/best_model_f1.pth"
    echo "========================================="
    echo "Running TTA Evaluation..."
    echo "========================================="

    # Note: TTA script needs model_dual_branch import, use standalone for now
    echo "TTA evaluation not available for dual-branch model yet"
    echo "Run evaluate_tta.py manually with --model_type dual_branch"
fi

echo "All done!"
