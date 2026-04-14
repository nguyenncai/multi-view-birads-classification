"""
Fixed Loss Functions - Giải quyết model collapse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossFix(nn.Module):
    """
    Focal Loss với gamma thấp hơn để tránh model collapse
    """
    def __init__(self, alpha=None, gamma=1.0, reduction='mean', ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma  # Giảm từ 2.0 → 1.0
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets, reduction=None):
        reduction = self.reduction if reduction is None else reduction
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            if reduction == 'none':
                return torch.zeros_like(targets, dtype=inputs.dtype, device=inputs.device)
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        full_targets = targets
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if reduction == 'none':
            full_loss = torch.zeros_like(full_targets, dtype=focal_loss.dtype, device=inputs.device)
            full_loss[valid_mask] = focal_loss
            return full_loss
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BalancedLoss(nn.Module):
    """
    Balanced loss: Mix của CE và Focal Loss
    Tránh model collapse bằng cách không quá aggressive
    """
    def __init__(self, alpha=None, focal_weight=0.5, gamma=1.0, smoothing=0.05, ignore_index=-1):
        super().__init__()
        self.focal_loss = FocalLossFix(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.alpha = alpha
        self.focal_weight = focal_weight  # 0.5 = 50% focal, 50% CE
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, pred, target, reduction='mean'):
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            if reduction == 'none':
                return torch.zeros_like(target, dtype=pred.dtype, device=pred.device)
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Focal loss
        loss_focal = self.focal_loss(pred, target, reduction='none')

        # Cross entropy with label smoothing. When class weights are provided,
        # apply them here too; otherwise the CE branch would ignore imbalance.
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        alpha = self.alpha
        if alpha is not None and alpha.device != pred_valid.device:
            alpha = alpha.to(pred_valid.device)
        loss_ce = F.cross_entropy(
            pred_valid,
            target_valid,
            weight=alpha,
            reduction='none',
            label_smoothing=self.smoothing
        )
        full_loss_ce = torch.zeros_like(target, dtype=loss_ce.dtype, device=pred.device)
        full_loss_ce[valid_mask] = loss_ce

        # Balanced combination
        total_loss = self.focal_weight * loss_focal + (1 - self.focal_weight) * full_loss_ce

        if reduction == 'none':
            return total_loss
        if reduction == 'sum':
            return total_loss[valid_mask].sum()
        return total_loss[valid_mask].mean()


class OrdinalEMDLoss(nn.Module):
    """Ordinal EMD loss on cumulative class probabilities."""
    def __init__(self, class_weights=None, ignore_index=-1, p=2):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.p = p

    def forward(self, pred, target, reduction='mean'):
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            if reduction == 'none':
                return torch.zeros_like(target, dtype=pred.dtype, device=pred.device)
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        probs = torch.softmax(pred_valid, dim=1)
        target_onehot = F.one_hot(target_valid, num_classes=pred_valid.shape[1]).to(probs.dtype)
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(target_onehot, dim=1)
        loss = torch.abs(pred_cdf - target_cdf) ** self.p
        loss = loss.mean(dim=1)

        if self.class_weights is not None:
            weights = self.class_weights.to(pred_valid.device).gather(0, target_valid)
            loss = loss * weights

        if reduction == 'none':
            full_loss = torch.zeros_like(target, dtype=loss.dtype, device=pred.device)
            full_loss[valid_mask] = loss
            return full_loss
        if reduction == 'sum':
            return loss.sum()
        return loss.mean()


class BIRADSLossMedium(nn.Module):
    """Medium loss for medium model"""
    def __init__(
        self,
        birads_weights=None,
        focal_weight=0.5,
        gamma=1.0,
        smoothing=0.05,
        ordinal_weight=0.0,
        ordinal_power=2,
    ):
        super().__init__()
        self.criterion = BalancedLoss(
            alpha=birads_weights,
            focal_weight=focal_weight,
            gamma=gamma,
            smoothing=smoothing,
            ignore_index=-1
        )
        self.ordinal_weight = ordinal_weight
        self.ordinal_criterion = None
        if ordinal_weight > 0:
            self.ordinal_criterion = OrdinalEMDLoss(
                class_weights=birads_weights,
                ignore_index=-1,
                p=ordinal_power,
            )
    
    def forward(self, outputs, batch):
        labels = batch['labels']
        birads_labels = labels['birads']

        per_sample_loss = self.criterion(outputs['birads'], birads_labels, reduction='none')
        valid_mask = (birads_labels != -1)

        if 'birads2' in labels and 'mixup_lam' in labels:
            mixup_mask = batch.get('is_mixup', False)
            if isinstance(mixup_mask, bool):
                mixup_mask = torch.full_like(birads_labels, mixup_mask, dtype=torch.bool)
            elif isinstance(mixup_mask, torch.Tensor):
                mixup_mask = mixup_mask.to(birads_labels.device).bool()
            else:
                mixup_mask = torch.zeros_like(birads_labels, dtype=torch.bool)

            secondary_labels = labels['birads2']
            secondary_loss = self.criterion(outputs['birads'], secondary_labels, reduction='none')
            lam = labels['mixup_lam'].to(outputs['birads'].device).clamp(0.0, 1.0)

            mixup_mask = mixup_mask & (secondary_labels != -1)
            mixed_loss = lam * per_sample_loss + (1.0 - lam) * secondary_loss
            per_sample_loss = torch.where(mixup_mask, mixed_loss, per_sample_loss)

        if valid_mask.any():
            base_loss = per_sample_loss[valid_mask].mean()
        else:
            base_loss = torch.tensor(0.0, device=outputs['birads'].device, requires_grad=True)

        ordinal_loss = torch.tensor(0.0, device=outputs['birads'].device)
        if self.ordinal_criterion is not None:
            ordinal_loss = self.ordinal_criterion(outputs['birads'], birads_labels, reduction='mean')

        loss = base_loss + self.ordinal_weight * ordinal_loss

        loss_dict = {
            'total': loss.item(),
            'birads': base_loss.item(),
            'ordinal': float(ordinal_loss.item()),
        }
        
        return loss, loss_dict


if __name__ == '__main__':
    # Test loss
    batch_size = 8
    num_classes = 5
    
    outputs = {'birads': torch.randn(batch_size, num_classes)}
    labels = {'birads': torch.randint(0, num_classes, (batch_size,))}
    batch = {'labels': labels, 'is_mixup': False}
    
    weights = torch.tensor([1.0, 1.5, 2.5, 3.0, 4.0])
    criterion = BIRADSLossMedium(birads_weights=weights, focal_weight=0.5, gamma=1.0)
    
    loss, loss_dict = criterion(outputs, batch)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
