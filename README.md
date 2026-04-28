# Dual-Branch Ensemble for Multi-View BI-RADS Classification

📌 **[Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1SaGaN1Sbwpe0E43wEQmERqZ7qMJDsrw6?usp=drive_link)**
A deep learning model for **5-class BI-RADS mammography classification** (BI-RADS 1–5) using 4-view input (R_CC, R_MLO, L_CC, L_MLO) from the VinDr-Mammo dataset.

## Architecture

```
  R_CC + L_CC ──► CC Branch (ViT-Large DINOv2) ──► Gated Attention ──► CC Heads
                                                                          │
                                                               Average Ensemble ──► BI-RADS prediction
                                                                          │
  R_MLO + L_MLO ─► MLO Branch (ViT-Large DINOv2) ─► Gated Attention ─► MLO Heads
```

- **Backbone**: ViT-Large DINOv2 (304M params × 2 branches)
- **Input**: 518×518 px (native DINOv2 resolution)
- **Total params**: ~631M
- **Output**: BI-RADS right breast + BI-RADS left breast (5 classes each)

> Inspired by *"Innovative Multi-View Strategies for AI-Assisted Breast Cancer Screening"* (MDPI 2025)

## Results

| Metric | Score |
|--------|-------|
| F1 Macro (avg) | **0.582** |
| AUC Macro OVR | **0.842** |

## Project Structure

```
├── run_dual_branch_ensemble_v3.sh   # Entrypoint — chạy training với config đầy đủ
├── train_dual_branch.py             # Training loop (AMP, EMA, gradient accumulation)
├── model_dual_branch.py             # Kiến trúc Dual-Branch Ensemble
├── dataset.py                       # Dataset 4-view VinDr-Mammo, patient-level grouping
├── losses_fixed.py                  # BalancedLoss (Focal + CE) + OrdinalEMD Loss
├── annotation_schema.py             # Normalize VinDr CSV schema (hỗ trợ nhiều format)
├── evaluate_tta.py                  # Test-Time Augmentation evaluation
├── plot_training_results.py         # Vẽ biểu đồ kết quả training
├── requirements.txt                 # Dependencies
└── DOCUMENTATION_v3.md              # Tài liệu chi tiết kiến trúc
```

### File descriptions

| File | Description |
|------|-------------|
| **`run_dual_branch_ensemble_v3.sh`** | Shell script entrypoint. Thiết lập CUDA, gọi `train_dual_branch.py` với toàn bộ hyperparameters đã tuned (batch_size=1, grad_accum=8, lr=2e-5, 35 epochs, heavy augmentation). |
| **`train_dual_branch.py`** | Training loop chính: tạo model, optimizer (AdamW, 2 LR groups), scheduler (warmup → cosine), AMP + GradScaler, EMA (decay=0.9995), early stopping. Lưu best model theo F1 và AUC. |
| **`model_dual_branch.py`** | Định nghĩa `DualBranchEnsemble` — 2 ViewBranch (CC và MLO) với backbone DINOv2 riêng biệt, Gated Attention Fusion cho mỗi branch, 4 classification heads (right/left × CC/MLO), ensemble bằng average logits. |
| **`dataset.py`** | `VinDrMammographyDataset` — group ảnh theo patient (4 views/patient), hỗ trợ heavy/light augmentation, ImageNet/MammoClip normalization, CLAHE preprocessing (mammodino profile). Missing views → zero tensor. |
| **`losses_fixed.py`** | `BalancedLoss`: mix Focal Loss (gamma tùy chỉnh) + CE (label smoothing, class weights). `OrdinalEMDLoss`: phạt dự đoán lệch xa nhãn thật dựa trên CDF, tận dụng thứ tự BI-RADS 1 < 2 < 3 < 4 < 5. |
| **`annotation_schema.py`** | `load_vindr_annotations()` — normalize 2 loại CSV schema của VinDr (cleaned_label.csv và breast-level_annotations.csv) về format chung: patient_id, image_id, laterality, view, BIRADS, split. |
| **`evaluate_tta.py`** | Đánh giá model với Test-Time Augmentation (original + horizontal flip + vertical flip), average predictions → cải thiện accuracy. |

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
# Core: torch>=2.0, timm, torchvision, scikit-learn, pandas, tensorboard, tqdm
```

### Training

```bash
# Cần ~6GB VRAM trống
nvidia-smi

# Chạy training
bash run_dual_branch_ensemble_v3.sh
```

### Monitor

```bash
# Xem kết quả real-time
tail -f runs/dual_branch_ensemble_dinov2_*/training_results.csv

# TensorBoard
tensorboard --logdir runs/dual_branch_ensemble_dinov2_*/tensorboard
```

## Key Training Settings

| Setting | Value | Note |
|---------|-------|------|
| Backbone | `vit_large_patch14_dinov2` | Pretrained ImageNet-22K |
| Batch size | 1 (effective 8) | Gradient accumulation × 8 |
| Learning rate | 2e-5 (head), 4e-6 (backbone) | Backbone LR × 0.2 |
| Scheduler | 3-epoch warmup → cosine | eta_min = 2e-7 |
| Augmentation | Heavy (±12° rotation, ±8% translate, shear) | + horizontal flip |
| Loss | BalancedLoss + OrdinalEMD (weight=0.2) | Class weights: inverse_sqrt |
| EMA | decay = 0.9995 | Validation dùng EMA model |
| Gradient checkpointing | Enabled | Tiết kiệm ~40% VRAM |

## References

1. MDPI 2025 — [Multi-View Strategies for AI-Assisted Breast Cancer Screening](https://www.mdpi.com/2313-433X/11/8/247)
2. ArXiv 2024 — [Deep BI-RADS Network for Improved Cancer Detection](https://arxiv.org/html/2411.10894v1)
3. DINOv2 — [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
4. NVIDIA TAO — DINOv2 Fine-tuning Best Practices (Layer-wise LR decay)
