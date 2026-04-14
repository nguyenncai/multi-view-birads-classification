#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) evaluation for multi-view BI-RADS model.

Applies multiple augmentations at inference time and averages predictions
to improve accuracy. Based on best practices from 2024-2025 papers.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VinDrMammographyDataset
from model_multiview_birads import create_multiview_birads_model

import torchvision.transforms as transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_tta_transforms(img_size):
    """Get list of TTA transforms: original + flips + rotations."""
    base = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    flip_h = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    flip_v = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return [base, flip_h, flip_v]


def predict_with_tta(model, images, tta_transforms, device):
    """Run TTA prediction by averaging over multiple augmentations."""
    model.eval()
    all_probs_right = []
    all_probs_left = []

    with torch.no_grad():
        for transform in tta_transforms:
            # Apply transform to each view
            tta_images = {}
            for view_name, img_tensor in images.items():
                # Convert tensor back to PIL, apply transform
                img_pil = transforms.functional.to_pil_image(img_tensor.cpu())
                img_t = transform(img_pil).unsqueeze(0).to(device)
                tta_images[view_name] = img_t

            outputs, _ = model(tta_images)
            probs_right = torch.softmax(outputs['birads_right'], dim=1).cpu().numpy()
            probs_left = torch.softmax(outputs['birads_left'], dim=1).cpu().numpy()
            all_probs_right.append(probs_right)
            all_probs_left.append(probs_left)

    # Average predictions
    avg_probs_right = np.mean(all_probs_right, axis=0)
    avg_probs_left = np.mean(all_probs_left, axis=0)
    return avg_probs_right, avg_probs_left


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    model = create_multiview_birads_model(
        backbone_name=args.backbone_name,
        img_size=args.img_size,
        pretrained=False,
        dropout=args.classifier_dropout,
        shared_backbone=(args.model_type == 'shared'),
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Try different state dict keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_state_dict' in checkpoint and checkpoint.get('ema_state_dict') is not None:
        state_dict = checkpoint['ema_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded checkpoint from {args.checkpoint}')

    # Load dataset
    dataset = VinDrMammographyDataset(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.split,
        img_size=args.img_size,
        augmentation_profile='light',  # No train augmentations for eval
        normalization_profile='imagenet',
        preprocess_profile=args.preprocess_profile,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # TTA transforms
    tta_transforms = get_tta_transforms(args.img_size)
    print(f'Using {len(tta_transforms)} TTA transforms: original, horizontal flip, vertical flip')

    # Evaluate
    all_right_labels, all_right_preds, all_right_probs = [], [], []
    all_left_labels, all_left_preds, all_left_probs = [], [], []

    for batch in tqdm(dataloader, desc='TTA Evaluation'):
        images = {k: v.to(device) for k, v in batch['images'].items()}
        labels = {k: v for k, v in batch['labels'].items()}

        # TTA prediction
        probs_right, probs_left = predict_with_tta(model, images, tta_transforms, device)

        right_labels = labels['right_birads'].numpy()
        left_labels = labels['left_birads'].numpy()
        right_preds = np.argmax(probs_right, axis=1)
        left_preds = np.argmax(probs_left, axis=1)

        right_mask = right_labels != -1
        left_mask = left_labels != -1
        all_right_labels.extend(right_labels[right_mask])
        all_right_preds.extend(right_preds[right_mask])
        all_right_probs.extend(probs_right[right_mask])
        all_left_labels.extend(left_labels[left_mask])
        all_left_preds.extend(left_preds[left_mask])
        all_left_probs.extend(probs_left[left_mask])

    # Compute metrics
    class_ids = np.arange(5)
    f1_right = f1_score(all_right_labels, all_right_preds, labels=class_ids, average='macro', zero_division=0)
    f1_left = f1_score(all_left_labels, all_left_preds, labels=class_ids, average='macro', zero_division=0)
    f1_avg = (f1_right + f1_left) / 2

    combined_labels = np.array(list(all_right_labels) + list(all_left_labels))
    combined_preds = np.array(list(all_right_preds) + list(all_left_preds))
    combined_probs = np.array(list(all_right_probs) + list(all_left_probs))

    auc_macro = 0.0
    unique_labels = np.unique(combined_labels)
    if len(unique_labels) > 2:
        auc_macro = roc_auc_score(combined_labels, combined_probs[:, unique_labels],
                                  labels=unique_labels, multi_class='ovr', average='macro')
    elif len(unique_labels) == 2:
        positive_label = int(unique_labels[-1])
        auc_macro = roc_auc_score((combined_labels == positive_label).astype(np.int32),
                                 combined_probs[:, positive_label])

    print(f'\n===== TTA Evaluation Results =====')
    print(f'F1 Right:  {f1_right:.4f}')
    print(f'F1 Left:   {f1_left:.4f}')
    print(f'F1 Avg:    {f1_avg:.4f}')
    print(f'AUC Macro: {auc_macro:.4f}')

    # Save results
    if args.output_file:
        results = {
            'f1_right': f1_right,
            'f1_left': f1_left,
            'f1_avg': f1_avg,
            'auc_macro': auc_macro,
            'checkpoint': args.checkpoint,
            'tta_transforms': len(tta_transforms),
        }
        with open(args.output_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {args.output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--csv_path', type=str, default='/home/fit02/nguyen_workspace/breast-level_annotations.csv')
    parser.add_argument('--img_dir', type=str, default='/home/fit02/nguyen_workspace/vindr/cleaned_images')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--model_type', type=str, default='shared', choices=['shared', 'viewspec'])
    parser.add_argument('--backbone_name', type=str, default='vit_large_patch14_dinov2')
    parser.add_argument('--classifier_dropout', type=float, default=0.35)
    parser.add_argument('--preprocess_profile', type=str, default='mammodino', choices=['none', 'mammoclip', 'mammodino'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_file', type=str, default=None, help='Output JSON file for results')
    main(parser.parse_args())
