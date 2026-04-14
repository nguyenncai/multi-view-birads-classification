"""
Advanced Loss Functions for BI-RADS Classification
Including Focal Loss, Label Smoothing, and Mixup-aware losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-1):
        """
        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (default 2.0)
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Label to ignore
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class labels
        Returns:
            loss: scalar or [B] depending on reduction
        """
        # Filter out ignore_index
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probability of true class
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing
    Helps prevent overconfidence and improves generalization
    """
    def __init__(self, smoothing=0.1, ignore_index=-1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 to 1.0)
            ignore_index: Label to ignore
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C] logits
            target: [B] class labels
        Returns:
            loss: scalar
        """
        # Filter out ignore_index
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        n_class = pred.size(1)
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        
        # Compute loss
        log_prob = F.log_softmax(pred, dim=1)
        loss = (-smooth_one_hot * log_prob).sum(dim=1).mean()
        
        return loss


class MixupLoss(nn.Module):
    """Loss function for mixup augmentation"""
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion
    
    def forward(self, pred, y1, y2, lam):
        """
        Args:
            pred: [B, C] predictions
            y1, y2: [B] two mixed labels
            lam: [B] or scalar, mixing coefficient
        Returns:
            loss: scalar
        """
        if isinstance(lam, torch.Tensor):
            lam = lam.mean().item()
        
        loss1 = self.base_criterion(pred, y1)
        loss2 = self.base_criterion(pred, y2)
        
        return lam * loss1 + (1 - lam) * loss2


class CombinedLoss(nn.Module):
    """
    Combined loss with Focal Loss, Label Smoothing, and Deep Supervision
    """
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, 
                 aux_weight=0.3, ignore_index=-1):
        """
        Args:
            alpha: Class weights for focal loss
            gamma: Focal loss gamma parameter
            smoothing: Label smoothing factor
            aux_weight: Weight for auxiliary loss (deep supervision)
            ignore_index: Label to ignore
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.smooth_ce = LabelSmoothingCrossEntropy(smoothing=smoothing, ignore_index=ignore_index)
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
    
    def forward(self, outputs, labels, use_mixup=False):
        """
        Args:
            outputs: dict with 'birads' and optionally 'birads_aux'
            labels: dict with 'birads' and optionally mixup labels
            use_mixup: bool, whether batch uses mixup
        Returns:
            total_loss: scalar
            loss_dict: dict with loss components
        """
        birads_labels = labels['birads']
        valid_birads = (birads_labels != self.ignore_index)
        
        if not valid_birads.any():
            return torch.tensor(0.0, device=outputs['birads'].device, requires_grad=True), {}
        
        # Main loss
        if use_mixup and 'birads2' in labels:
            # Mixup loss
            y1 = labels['birads'][valid_birads]
            y2 = labels['birads2'][valid_birads]
            lam = labels['mixup_lam'][valid_birads]
            
            focal_criterion = self.focal_loss
            mixup_focal = MixupLoss(focal_criterion)
            loss_main = mixup_focal(
                outputs['birads'][valid_birads], 
                y1, y2, lam
            )
        else:
            # Regular loss: Focal + Label Smoothing
            loss_focal = self.focal_loss(
                outputs['birads'][valid_birads],
                birads_labels[valid_birads]
            )
            loss_smooth = self.smooth_ce(
                outputs['birads'][valid_birads],
                birads_labels[valid_birads]
            )
            loss_main = 0.7 * loss_focal + 0.3 * loss_smooth
        
        # Auxiliary loss (deep supervision)
        loss_aux = torch.tensor(0.0, device=outputs['birads'].device)
        if 'birads_aux' in outputs:
            if use_mixup and 'birads2' in labels:
                mixup_aux = MixupLoss(self.focal_loss)
                loss_aux = mixup_aux(
                    outputs['birads_aux'][valid_birads],
                    y1, y2, lam
                )
            else:
                loss_aux = self.focal_loss(
                    outputs['birads_aux'][valid_birads],
                    birads_labels[valid_birads]
                )
        
        # Total loss
        total_loss = loss_main + self.aux_weight * loss_aux
        
        loss_dict = {
            'total': total_loss.item(),
            'main': loss_main.item(),
            'aux': loss_aux.item() if isinstance(loss_aux, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict


class BIRADSLossHeavy(nn.Module):
    """Heavy loss wrapper for BI-RADS task"""
    def __init__(self, birads_weights=None, gamma=2.0, smoothing=0.1, aux_weight=0.3):
        super().__init__()
        self.criterion = CombinedLoss(
            alpha=birads_weights,
            gamma=gamma,
            smoothing=smoothing,
            aux_weight=aux_weight,
            ignore_index=-1
        )
    
    def forward(self, outputs, batch):
        """
        Args:
            outputs: dict with model outputs
            batch: batch dict with 'labels' and 'is_mixup'
        Returns:
            loss: scalar
            loss_dict: dict with loss components
        """
        labels = batch['labels']
        is_mixup = batch.get('is_mixup', False)
        
        # Check if any sample in batch is mixup
        if isinstance(is_mixup, torch.Tensor):
            use_mixup = is_mixup.any().item()
        elif isinstance(is_mixup, bool):
            use_mixup = is_mixup
        else:
            use_mixup = False
        
        loss, loss_dict = self.criterion(outputs, labels, use_mixup=use_mixup)
        return loss, loss_dict


if __name__ == '__main__':
    # Test losses
    batch_size = 8
    num_classes = 5
    
    # Dummy data
    outputs = {
        'birads': torch.randn(batch_size, num_classes),
        'birads_aux': torch.randn(batch_size, num_classes)
    }
    labels = {
        'birads': torch.randint(0, num_classes, (batch_size,))
    }
    batch = {
        'labels': labels,
        'is_mixup': False
    }
    
    # Test combined loss
    weights = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
    criterion = BIRADSLossHeavy(birads_weights=weights)
    
    loss, loss_dict = criterion(outputs, batch)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
