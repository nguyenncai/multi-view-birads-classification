#!/usr/bin/env python3
"""
Evaluation script for HV Dataset using Dual Branch Ensemble Model.
Supports both standard evaluation and Test-Time Augmentation (TTA).
"""
import argparse
import os
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_hv import HVMammographyDataset
from dataset import VinDrMammographyDataset
from model_dual_branch import create_dualbranch_ensemble_model


def predict_with_tta_tensor(model, images, use_tta, device):
    """
    Run prediction. If use_tta is True, applies horizontal/vertical flips 
    directly on tensors and averages the probabilities.
    """
    model.eval()
    
    with torch.no_grad():
        if not use_tta:
            outputs, _ = model(images)
            probs_right = torch.softmax(outputs['birads_right'], dim=1).cpu().numpy()
            probs_left = torch.softmax(outputs['birads_left'], dim=1).cpu().numpy()
            return probs_right, probs_left
            
        # TTA: Original, Horizontal Flip, Vertical Flip
        all_probs_right = []
        all_probs_left = []
        
        # 1. Original
        outputs, _ = model(images)
        all_probs_right.append(torch.softmax(outputs['birads_right'], dim=1).cpu().numpy())
        all_probs_left.append(torch.softmax(outputs['birads_left'], dim=1).cpu().numpy())
        
        # 2. Horizontal Flip (flip along W dimension / dim 3 of B,C,H,W)
        images_hf = {k: torch.flip(v, dims=[3]) for k, v in images.items()}
        outputs_hf, _ = model(images_hf)
        all_probs_right.append(torch.softmax(outputs_hf['birads_right'], dim=1).cpu().numpy())
        all_probs_left.append(torch.softmax(outputs_hf['birads_left'], dim=1).cpu().numpy())
        
        # 3. Vertical Flip (flip along H dimension / dim 2 of B,C,H,W)
        images_vf = {k: torch.flip(v, dims=[2]) for k, v in images.items()}
        outputs_vf, _ = model(images_vf)
        all_probs_right.append(torch.softmax(outputs_vf['birads_right'], dim=1).cpu().numpy())
        all_probs_left.append(torch.softmax(outputs_vf['birads_left'], dim=1).cpu().numpy())
        
        # Average probability across the 3 TTA variations
        avg_probs_right = np.mean(all_probs_right, axis=0)
        avg_probs_left = np.mean(all_probs_left, axis=0)
        
        return avg_probs_right, avg_probs_left


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 1. Initialize Model
    print(f'Initializing Dual Branch model with backbone: {args.backbone_name}')
    model = create_dualbranch_ensemble_model(
        backbone_name=args.backbone_name,
        img_size=args.img_size,
        pretrained=False,
        dropout=args.classifier_dropout,
    ).to(device)

    # 2. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    print(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Checkpoint Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Checkpoint Val F1: {checkpoint.get('val_f1_macro', 'N/A')}")
    elif isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint and checkpoint.get('ema_state_dict'):
        state_dict = checkpoint['ema_state_dict']
        print("  Using EMA state dict from checkpoint.")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print('Checkpoint loaded successfully!')

    # 3. Initialize Dataset and DataLoader
    print(f'\nLoading {args.dataset_type.upper()} Dataset ({args.split} split)...')
    if args.dataset_type == 'hv':
        dataset = HVMammographyDataset(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            split=args.split,
            val_size=args.val_size,
            random_state=args.random_state,
            img_size=args.img_size,
            augmentation_profile='light', # Evaluatin always uses light (no agressive augmentations)
            normalization_profile=args.normalization_profile,
            preprocess_profile=args.preprocess_profile,
        )
    else:
        dataset = VinDrMammographyDataset(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            split=args.split,
            img_size=args.img_size,
            augmentation_profile='light',
            normalization_profile=args.normalization_profile,
            preprocess_profile=args.preprocess_profile,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    
    print(f'Total patients to evaluate: {len(dataset)}')
    if args.use_tta:
        print('Test-Time Augmentation (TTA) is ENABLED (Original, H-Flip, V-Flip).')
    else:
        print('Test-Time Augmentation (TTA) is DISABLED.')

    # 4. Evaluation Loop
    all_right_labels, all_right_preds, all_right_probs = [], [], []
    all_left_labels, all_left_preds, all_left_probs = [], [], []

    for batch in tqdm(dataloader, desc='Evaluating'):
        images = {k: v.to(device) for k, v in batch['images'].items()}
        labels = {k: v for k, v in batch['labels'].items()} # keep labels on cpu

        # Get probabilities (with or without TTA)
        probs_right, probs_left = predict_with_tta_tensor(model, images, args.use_tta, device)

        right_labels = labels['right_birads'].numpy()
        left_labels = labels['left_birads'].numpy()
        
        right_preds = np.argmax(probs_right, axis=1)
        left_preds = np.argmax(probs_left, axis=1)

        # Filter out ignored indices (-1)
        right_mask = right_labels != -1
        left_mask = left_labels != -1
        
        all_right_labels.extend(right_labels[right_mask])
        all_right_preds.extend(right_preds[right_mask])
        all_right_probs.extend(probs_right[right_mask])
        
        all_left_labels.extend(left_labels[left_mask])
        all_left_preds.extend(left_preds[left_mask])
        all_left_probs.extend(probs_left[left_mask])

    # 5. Compute Metrics
    class_ids = np.arange(5)
    target_names = ['BIRADS 1', 'BIRADS 2', 'BIRADS 3', 'BIRADS 4', 'BIRADS 5']
    
    f1_right = f1_score(all_right_labels, all_right_preds, labels=class_ids, average='macro', zero_division=0)
    f1_left = f1_score(all_left_labels, all_left_preds, labels=class_ids, average='macro', zero_division=0)
    
    # Combine right and left for overall metrics
    combined_labels = np.array(list(all_right_labels) + list(all_left_labels))
    combined_preds = np.array(list(all_right_preds) + list(all_left_preds))
    combined_probs = np.array(list(all_right_probs) + list(all_left_probs))

    f1_avg = f1_score(combined_labels, combined_preds, labels=class_ids, average='macro', zero_division=0)

    # AUC Macro OVR
    auc_macro = 0.0
    unique_labels = np.unique(combined_labels)
    if len(unique_labels) > 2:
        auc_macro = roc_auc_score(combined_labels, combined_probs[:, unique_labels],
                                  labels=unique_labels, multi_class='ovr', average='macro')
    elif len(unique_labels) == 2:
        positive_label = int(unique_labels[-1])
        auc_macro = roc_auc_score((combined_labels == positive_label).astype(np.int32),
                                 combined_probs[:, positive_label])

    # 6. Detailed Reports
    print(f'\n{"="*50}')
    print(f'          EVALUATION RESULTS ({args.dataset_type.upper()})          ')
    print(f'{"="*50}')
    print(f'F1 Score (Right Breasts) : {f1_right:.4f}')
    print(f'F1 Score (Left Breasts)  : {f1_left:.4f}')
    print(f'Overall F1 Macro         : {f1_avg:.4f}')
    print(f'Overall AUC Macro (OvR)  : {auc_macro:.4f}')
    print(f'{"-"*50}')
    
    print('\n*** OVERALL CLASSIFICATION REPORT ***')
    print(classification_report(combined_labels, combined_preds, labels=class_ids, target_names=target_names, zero_division=0))
    
    print('\n*** OVERALL CONFUSION MATRIX ***')
    cm = confusion_matrix(combined_labels, combined_preds, labels=class_ids)
    print(f"{'':>10} " + " ".join([f"P-{i+1:>2}" for i in range(5)]))
    for i, row in enumerate(cm):
        print(f"True {i+1:<4} | " + " ".join([f"{val:>4}" for val in row]))

    # 7. Save to File
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, 'evaluation_hv_results.json')
        
        results = {
            'metrics': {
                'f1_right': float(f1_right),
                'f1_left': float(f1_left),
                'f1_macro': float(f1_avg),
                'auc_macro': float(auc_macro)
            },
            'classification_report': classification_report(combined_labels, combined_preds, labels=class_ids, target_names=target_names, zero_division=0, output_dict=True),
            'confusion_matrix': cm.tolist(),
            'config': {
                'checkpoint': args.checkpoint,
                'use_tta': args.use_tta,
                'val_size': args.val_size,
                'split': args.split,
            }
        }
        
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nDetailed results saved to {out_file}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Dual Branch model on HV or VinDr Dataset")
    # Dataset args
    parser.add_argument('--dataset_type', type=str, default='hv', choices=['hv', 'vindr'], help='Dataset type')
    parser.add_argument('--csv_path', type=str, default='/home/fit02/nguyen_workspace/dataset_hv/cleaned_label.csv')
    parser.add_argument('--img_dir', type=str, default='/home/fit02/nguyen_workspace/dataset_hv')
    parser.add_argument('--split', type=str, default='test', choices=['training', 'test', 'all'])
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model_f1.pth')
    parser.add_argument('--backbone_name', type=str, default='vit_large_patch14_dinov2')
    parser.add_argument('--classifier_dropout', type=float, default=0.35)
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--normalization_profile', type=str, default='imagenet', choices=['imagenet', 'mammoclip'])
    parser.add_argument('--preprocess_profile', type=str, default='mammodino', choices=['none', 'mammoclip', 'mammodino'])
    
    # Execution args
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_tta', action='store_true', help='Enable Test Time Augmentation (flips)')
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save JSON results')
    
    args = parser.parse_args()
    main(args)
