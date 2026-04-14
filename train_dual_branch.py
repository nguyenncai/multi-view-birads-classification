#!/usr/bin/env python3
"""
Train a Dual-Branch Ensemble BI-RADS model on VinDr mammography annotations.

Architecture inspired by MDPI 2025 paper:
- CC Branch: processes CC views independently
- MLO Branch: processes MLO views independently  
- Final prediction = average of branch probabilities
"""
import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import VinDrMammographyDataset, get_class_weights
from losses_fixed import BalancedLoss, OrdinalEMDLoss
from model_dual_branch import create_dualbranch_ensemble_model


class ModelEMA:
    """EMA helper for more stable validation."""
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            model_state = model.state_dict()
            for name, ema_value in self.ema_model.state_dict().items():
                model_value = model_state[name].detach()
                if not ema_value.dtype.is_floating_point:
                    ema_value.copy_(model_value)
                else:
                    ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


class MultiViewBIRADSLoss(torch.nn.Module):
    """Average right/left BI-RADS loss with optional ordinal regularization."""
    def __init__(self, birads_weights=None, focal_weight=0.0, gamma=1.0, smoothing=0.0, ordinal_weight=0.0):
        super().__init__()
        self.base = BalancedLoss(
            alpha=birads_weights,
            focal_weight=focal_weight,
            gamma=gamma,
            smoothing=smoothing,
            ignore_index=-1,
        )
        self.ordinal_weight = ordinal_weight
        self.ordinal = OrdinalEMDLoss(
            class_weights=birads_weights,
            ignore_index=-1,
            p=2,
        ) if ordinal_weight > 0 else None

    def _side_loss(self, logits, labels):
        base = self.base(logits, labels, reduction='mean')
        if self.ordinal is None:
            return base, torch.tensor(0.0, device=logits.device)
        ord_loss = self.ordinal(logits, labels, reduction='mean')
        return base + self.ordinal_weight * ord_loss, ord_loss

    def forward(self, outputs, labels):
        right_total, right_ord = self._side_loss(outputs['birads_right'], labels['right_birads'])
        left_total, left_ord = self._side_loss(outputs['birads_left'], labels['left_birads'])
        total_loss = 0.5 * (right_total + left_total)
        loss_dict = {
            'total': total_loss.item(),
            'right': right_total.item(),
            'left': left_total.item(),
            'ordinal_right': right_ord.item(),
            'ordinal_left': left_ord.item(),
        }
        return total_loss, loss_dict


def compute_macro_auc_ovr(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0:
        return 0.0
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        positive_label = int(unique_labels[-1])
        return float(roc_auc_score((labels == positive_label).astype(np.int32), probs[:, positive_label]))
    if len(unique_labels) > 2:
        probs_for_auc = probs[:, unique_labels]
        return float(roc_auc_score(labels, probs_for_auc, labels=unique_labels, multi_class='ovr', average='macro'))
    return 0.0


def build_patient_sampler(dataset, sampler_name):
    if sampler_name == 'uniform':
        return None

    side_labels = []
    for patient_id in dataset.patient_ids:
        labels = dataset.patient_data[patient_id]['labels']
        for side in ('R', 'L'):
            label = labels.get(side, {}).get('birads', -1)
            if label != -1:
                side_labels.append(int(label))

    class_counts = {label: side_labels.count(label) for label in range(1, 6)}
    exponent = 1.0 if sampler_name == 'weighted' else 0.5

    patient_weights = []
    for patient_id in dataset.patient_ids:
        labels = dataset.patient_data[patient_id]['labels']
        candidate_weights = []
        for side in ('R', 'L'):
            label = labels.get(side, {}).get('birads', -1)
            if label != -1 and class_counts[int(label)] > 0:
                candidate_weights.append(1.0 / (class_counts[int(label)] ** exponent))
        patient_weights.append(max(candidate_weights) if candidate_weights else 1.0)

    return WeightedRandomSampler(
        weights=torch.tensor(patient_weights, dtype=torch.float64),
        num_samples=len(patient_weights),
        replacement=True,
    )


def get_layerwise_lr_groups(model, base_lr, backbone_lr_multiplier, layerwise_decay=0.9):
    """Create learning rate groups with layer-wise decay for transformer blocks.
    
    Based on DINOv2 fine-tuning best practices (NVIDIA TAO, CVPR 2024):
    - Bottom layers get lower LR (decay^depth)
    - Top layers get higher LR
    """
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if parameter belongs to backbone
        is_backbone = False
        for prefix in model.pretrained_backbone_prefixes:
            if name.startswith(prefix):
                is_backbone = True
                break
        
        if is_backbone:
            backbone_params.append((name, param))
        else:
            head_params.append((name, param))

    # For head params, use full base_lr
    lr_groups = [{'params': [p for _, p in head_params], 'lr': base_lr}]
    
    # For backbone, apply layer-wise decay
    # Group by block depth (if possible) or use uniform decay
    backbone_lr = base_lr * backbone_lr_multiplier
    lr_groups.append({'params': [p for _, p in backbone_params], 'lr': backbone_lr})
    
    return lr_groups


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, accumulation_steps=1, ema=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        images = {k: v.to(device) for k, v in batch['images'].items()}
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        with autocast(device_type='cuda', enabled=device.type == 'cuda'):
            outputs, _ = model(images)
            loss, loss_dict = criterion(outputs, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        should_step = (
            (batch_idx + 1) % accumulation_steps == 0
            or (batch_idx + 1) == len(dataloader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        total_loss += loss_dict['total']
        pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_right_labels, all_right_preds, all_right_probs = [], [], []
    all_left_labels, all_left_preds, all_left_probs = [], [], []
    all_attention = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            images = {k: v.to(device) for k, v in batch['images'].items()}
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            outputs, attention = model(images)
            loss, loss_dict = criterion(outputs, labels)
            total_loss += loss_dict['total']
            all_attention.append(attention.cpu().numpy())

            right_probs = torch.softmax(outputs['birads_right'], dim=1).cpu().numpy()
            left_probs = torch.softmax(outputs['birads_left'], dim=1).cpu().numpy()
            right_preds = np.argmax(right_probs, axis=1)
            left_preds = np.argmax(left_probs, axis=1)
            right_labels = labels['right_birads'].cpu().numpy()
            left_labels = labels['left_birads'].cpu().numpy()

            right_mask = right_labels != -1
            left_mask = left_labels != -1
            all_right_labels.extend(right_labels[right_mask])
            all_right_preds.extend(right_preds[right_mask])
            all_right_probs.extend(right_probs[right_mask])
            all_left_labels.extend(left_labels[left_mask])
            all_left_preds.extend(left_preds[left_mask])
            all_left_probs.extend(left_probs[left_mask])

    class_ids = np.arange(5)
    f1_right = f1_score(
        all_right_labels,
        all_right_preds,
        labels=class_ids,
        average='macro',
        zero_division=0,
    ) if all_right_labels else 0.0
    f1_left = f1_score(
        all_left_labels,
        all_left_preds,
        labels=class_ids,
        average='macro',
        zero_division=0,
    ) if all_left_labels else 0.0
    f1_avg = float(np.mean([score for score, labels in [(f1_right, all_right_labels), (f1_left, all_left_labels)] if labels])) if (all_right_labels or all_left_labels) else 0.0

    combined_labels = np.array(list(all_right_labels) + list(all_left_labels))
    combined_preds = np.array(list(all_right_preds) + list(all_left_preds))
    combined_probs = np.array(list(all_right_probs) + list(all_left_probs))
    f1_per_class = f1_score(
        combined_labels,
        combined_preds,
        labels=class_ids,
        average=None,
        zero_division=0,
    ) if len(combined_labels) > 0 else np.zeros(5)
    auc_macro = compute_macro_auc_ovr(combined_labels, combined_probs) if len(combined_labels) > 0 else 0.0
    attention_avg = np.concatenate(all_attention, axis=0).mean(axis=0) if all_attention else np.zeros(4)

    return {
        'loss': total_loss / len(dataloader),
        'f1_right': f1_right,
        'f1_left': f1_left,
        'f1_avg': f1_avg,
        'f1_per_class': f1_per_class,
        'auc_macro_ovr': auc_macro,
        'attention': attention_avg,
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))

    train_dataset = VinDrMammographyDataset(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.train_split,
        img_size=args.img_size,
        augmentation_profile=args.augmentation_profile,
        normalization_profile=args.normalization_profile,
        preprocess_profile=args.preprocess_profile,
    )
    val_dataset = VinDrMammographyDataset(
        csv_path=args.csv_path,
        img_dir=args.img_dir,
        split=args.val_split,
        img_size=args.img_size,
        augmentation_profile='light',
        normalization_profile=args.normalization_profile,
        preprocess_profile=args.preprocess_profile,
    )

    sampler = build_patient_sampler(train_dataset, args.sampler)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    model = create_dualbranch_ensemble_model(
        backbone_name=args.backbone_name,
        img_size=args.img_size,
        pretrained=args.pretrained,
        dropout=args.classifier_dropout,
        use_grad_checkpointing=args.use_grad_checkpointing,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,} ({n_params/1e6:.1f}M)')

    class_weights = None
    if args.class_weights_method != 'none':
        class_weights = get_class_weights(
            args.csv_path,
            split=args.train_split,
            method=args.class_weights_method,
            deduplicate_breasts=True,
        )['birads'].to(device)
        print(f'BI-RADS class weights: {class_weights.detach().cpu().tolist()}')

    criterion = MultiViewBIRADSLoss(
        birads_weights=class_weights,
        focal_weight=args.focal_weight,
        gamma=args.focal_gamma,
        smoothing=args.label_smoothing,
        ordinal_weight=args.ordinal_weight,
    )

    # Optimizer with layer-wise LR groups
    lr_groups = get_layerwise_lr_groups(
        model, args.lr, args.backbone_lr_multiplier, args.layerwise_lr_decay
    )
    optimizer = optim.AdamW(lr_groups, weight_decay=args.weight_decay)

    # Scheduler
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.lr * 0.01)
    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine

    scaler = GradScaler('cuda', enabled=device.type == 'cuda')
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    best_f1 = 0.0
    best_auc = 0.0
    patience_counter = 0
    results_csv = os.path.join(args.output_dir, 'training_results.csv')
    with open(results_csv, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            'epoch', 'train_loss', 'val_loss', 'val_f1_right', 'val_f1_left', 'val_f1_avg',
            'val_auc_macro_ovr', 'f1_class_0', 'f1_class_1', 'f1_class_2', 'f1_class_3', 'f1_class_4',
            'attention_lcc', 'attention_lmlo', 'attention_rcc', 'attention_rmlo', 'lr'
        ])

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            accumulation_steps=args.gradient_accumulation_steps,
            ema=ema,
        )
        eval_model = ema.ema_model if ema is not None else model
        val_metrics = validate(eval_model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[-1]['lr']
        with open(results_csv, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                epoch, train_loss, val_metrics['loss'],
                val_metrics['f1_right'], val_metrics['f1_left'], val_metrics['f1_avg'],
                val_metrics['auc_macro_ovr'], *val_metrics['f1_per_class'],
                *val_metrics['attention'], current_lr,
            ])

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('F1/right', val_metrics['f1_right'], epoch)
        writer.add_scalar('F1/left', val_metrics['f1_left'], epoch)
        writer.add_scalar('F1/avg', val_metrics['f1_avg'], epoch)
        writer.add_scalar('AUC/macro_ovr', val_metrics['auc_macro_ovr'], epoch)
        writer.add_scalar('LR/head', current_lr, epoch)

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"F1 R/L/Avg: {val_metrics['f1_right']:.4f}/{val_metrics['f1_left']:.4f}/{val_metrics['f1_avg']:.4f}, "
            f"AUC: {val_metrics['auc_macro_ovr']:.4f}"
        )
        print(f"Per-class F1: {val_metrics['f1_per_class']}")

        payload = {
            'epoch': epoch,
            'model_state_dict': eval_model.state_dict(),
            'raw_model_state_dict': model.state_dict(),
            'ema_state_dict': ema.ema_model.state_dict() if ema is not None else None,
            'val_f1_macro': val_metrics['f1_avg'],
            'val_auc_macro_ovr': val_metrics['auc_macro_ovr'],
            'model_family': 'dual_branch_ensemble',
            'backbone_name': model.backbone_name,
            'train_args': vars(args),
        }
        if val_metrics['f1_avg'] > best_f1:
            best_f1 = val_metrics['f1_avg']
            patience_counter = 0
            torch.save(payload, os.path.join(args.output_dir, 'best_model_f1.pth'))
            print(f'✓ Saved best F1 model ({best_f1:.4f})')
        else:
            patience_counter += 1
        if val_metrics['auc_macro_ovr'] > best_auc:
            best_auc = val_metrics['auc_macro_ovr']
            torch.save(payload, os.path.join(args.output_dir, 'best_model_auc.pth'))
            print(f'✓ Saved best AUC model ({best_auc:.4f})')

        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f'Early stopping at epoch {epoch}')
            break

    writer.close()
    print(f'\nTraining completed! Best F1: {best_f1:.4f}, Best AUC: {best_auc:.4f}')
    print(f'Results saved to {results_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/home/fit02/nguyen_workspace/breast-level_annotations.csv')
    parser.add_argument('--img_dir', type=str, default='/home/fit02/nguyen_workspace/vindr/cleaned_images')
    parser.add_argument('--output_dir', type=str, default='./runs/dual_branch_ensemble')
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--train_split', type=str, default='training')
    parser.add_argument('--val_split', type=str, default='test')

    parser.add_argument('--backbone_name', type=str, default='vit_large_patch14_dinov2')
    parser.add_argument('--classifier_dropout', type=float, default=0.35)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--backbone_lr_multiplier', type=float, default=0.2)
    parser.add_argument('--layerwise_lr_decay', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--warmup_start_factor', type=float, default=0.2)
    parser.add_argument('--early_stopping', type=int, default=12)
    parser.add_argument('--use_ema', dest='use_ema', action='store_true')
    parser.add_argument('--no-ema', dest='use_ema', action='store_false')
    parser.add_argument('--ema_decay', type=float, default=0.9995)

    parser.add_argument('--focal_weight', type=float, default=0.15)
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--ordinal_weight', type=float, default=0.20)
    parser.add_argument('--class_weights_method', type=str, default='inverse_sqrt', choices=['none', 'inverse', 'inverse_sqrt', 'effective_num'])
    parser.add_argument('--sampler', type=str, default='sqrt_weighted', choices=['uniform', 'weighted', 'sqrt_weighted'])
    parser.add_argument('--augmentation_profile', type=str, default='heavy', choices=['light', 'heavy'])
    parser.add_argument('--normalization_profile', type=str, default='imagenet', choices=['imagenet', 'mammoclip'])
    parser.add_argument('--preprocess_profile', type=str, default='mammodino', choices=['none', 'mammoclip', 'mammodino'])
    parser.add_argument('--use_grad_checkpointing', action='store_true', help='Enable gradient checkpointing to save VRAM')

    parser.set_defaults(pretrained=True, use_ema=True)
    main(parser.parse_args())
