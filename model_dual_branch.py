#!/usr/bin/env python3
"""
Dual-Branch Ensemble model for Multi-View BI-RADS classification.

Architecture inspired by MDPI 2025 paper (https://www.mdpi.com/2313-433X/11/8/247):
- CC branch: processes CC views independently
- MLO branch: processes MLO views independently
- Final prediction = average of branch probabilities

Expected improvement: +5-15% F1 vs single model
"""
import torch
import torch.nn as nn
import timm


class GatedAttentionFusion(nn.Module):
    """Gated attention fusion for 2 views."""
    def __init__(self, feature_dim, reduction_ratio=4):
        super().__init__()
        hidden_dim = max(feature_dim, (2 * feature_dim) // reduction_ratio)
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, v1, v2):
        v_global = torch.cat([v1, v2], dim=1)
        alpha = self.attention_mlp(v_global)
        v_fused = torch.cat([
            v1 * alpha[:, 0:1],
            v2 * alpha[:, 1:2],
        ], dim=1)
        return v_fused, alpha


class ViewBranch(nn.Module):
    """Single branch processing 2 views (e.g., CC or MLO) for both sides."""
    def __init__(self, backbone_name, img_size, pretrained, feature_dim, dropout=0.35, use_grad_checkpointing=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        # Enable gradient checkpointing to save VRAM
        if use_grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        self.fusion = GatedAttentionFusion(feature_dim)

    def _encode(self, image):
        features = self.backbone(image)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim > 2:
            features = features[:, 0]
        return features

    def forward(self, left_view, right_view):
        v_left = self._encode(left_view)
        v_right = self._encode(right_view)
        v_fused, alpha = self.fusion(v_left, v_right)
        # Concatenate: left features + right features + fused
        return torch.cat([v_left, v_right, v_fused], dim=1), alpha


class DualBranchEnsemble(nn.Module):
    """Dual-Branch Ensemble for 4-view BI-RADS classification.
    
    Architecture:
    - CC Branch: R_CC + L_CC → features_cc
    - MLO Branch: R_MLO + L_MLO → features_mlo
    - Right Head: combines R_CC + R_MLO features
    - Left Head: combines L_CC + L_MLO features
    - Final = average(CC branch logits, MLO branch logits)
    """
    def __init__(
        self,
        backbone_name='vit_base_patch14_dinov2',
        img_size=384,
        pretrained=True,
        dropout=0.35,
        use_grad_checkpointing=False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained_backbone_prefixes = ('cc_branch', 'mlo_branch')

        # Create a shared backbone to save memory
        shared_backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        feature_dim = self._infer_feature_dim(shared_backbone)
        del shared_backbone

        # CC Branch
        self.cc_branch = ViewBranch(backbone_name, img_size, pretrained, feature_dim, use_grad_checkpointing=use_grad_checkpointing)
        # MLO Branch
        self.mlo_branch = ViewBranch(backbone_name, img_size, pretrained, feature_dim, use_grad_checkpointing=use_grad_checkpointing)

        # Side-specific heads
        head_input_dim = feature_dim * 4  # left + right + fused (2*feature_dim)
        hidden_dim = max(512, feature_dim)
        
        # CC-side heads
        self.right_head_cc = self._make_head(head_input_dim, 5, hidden_dim, dropout)
        self.left_head_cc = self._make_head(head_input_dim, 5, hidden_dim, dropout)
        
        # MLO-side heads
        self.right_head_mlo = self._make_head(head_input_dim, 5, hidden_dim, dropout)
        self.left_head_mlo = self._make_head(head_input_dim, 5, hidden_dim, dropout)

    def _infer_feature_dim(self, backbone):
        feature_dim = getattr(backbone, 'num_features', None)
        if feature_dim is None:
            feature_dim = getattr(backbone, 'embed_dim', None)
        if feature_dim is None:
            raise RuntimeError(f'Could not infer feature dim for backbone: {self.backbone_name}')
        return feature_dim

    def _make_head(self, input_dim, num_classes, hidden_dim, dropout):
        mid_dim = 256
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(mid_dim, num_classes),
        )

    def forward(self, images):
        # CC branch: R_CC + L_CC
        feats_cc, alpha_cc = self.cc_branch(images['R_CC'], images['L_CC'])
        logits_cc_right = self.right_head_cc(feats_cc)
        logits_cc_left = self.left_head_cc(feats_cc)

        # MLO branch: R_MLO + L_MLO
        feats_mlo, alpha_mlo = self.mlo_branch(images['R_MLO'], images['L_MLO'])
        logits_mlo_right = self.right_head_mlo(feats_mlo)
        logits_mlo_left = self.left_head_mlo(feats_mlo)

        # Ensemble: average of CC and MLO branch logits
        logits_right = 0.5 * (logits_cc_right + logits_mlo_right)
        logits_left = 0.5 * (logits_cc_left + logits_mlo_left)

        outputs = {
            'birads_right': logits_right,
            'birads_left': logits_left,
        }
        attention = torch.cat([alpha_cc, alpha_mlo], dim=1)
        return outputs, attention


def create_dualbranch_ensemble_model(
    backbone_name='vit_large_patch14_dinov2',
    img_size=518,
    pretrained=True,
    dropout=0.35,
    use_grad_checkpointing=False,
):
    return DualBranchEnsemble(
        backbone_name=backbone_name,
        img_size=img_size,
        pretrained=pretrained,
        dropout=dropout,
        use_grad_checkpointing=use_grad_checkpointing,
    )
