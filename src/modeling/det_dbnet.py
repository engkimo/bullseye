import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    """Feature Pyramid Network for DBNet++."""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        
        # Lateral connections
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) 
            for _ in range(len(in_channels))
        ])
        
        # Extra layers for FPN output
        self.extra_convs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels // 4) for _ in range(len(in_channels))
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Build laterals
        laterals = [
            lateral(features[i]) for i, lateral in enumerate(self.laterals)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # FPN convolutions
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals))
        ]
        
        # Upsample and concatenate
        target_size = fpn_outs[0].shape[2:]
        upsampled = []
        
        for i, out in enumerate(fpn_outs):
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            out = self.extra_convs[i](out)
            upsampled.append(out)
        
        # Concatenate all features
        return torch.cat(upsampled, dim=1)


class DifferentiableBinarization(nn.Module):
    """Differentiable Binarization module for DBNet++."""
    
    def __init__(self, in_channels: int, k: int = 50):
        super().__init__()
        self.k = k
        
        # Probability branch
        self.prob_conv = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Sigmoid()
        )
        
        # Threshold branch
        self.thresh_conv = nn.Sequential(
            ConvBNReLU(in_channels, 64),
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Dict[str, torch.Tensor]:
        # Probability map
        prob_map = self.prob_conv(x)
        
        if training:
            # Threshold map
            thresh_map = self.thresh_conv(x)
            
            # Differentiable binarization
            binary_map = self._db(prob_map, thresh_map)
            
            return {
                'probability': prob_map,
                'threshold': thresh_map,
                'binary': binary_map
            }
        else:
            return {
                'probability': prob_map,
                'binary': prob_map
            }
    
    def _db(self, prob_map: torch.Tensor, thresh_map: torch.Tensor) -> torch.Tensor:
        """Differentiable binarization function."""
        return torch.reciprocal(1 + torch.exp(-self.k * (prob_map - thresh_map)))


class AdaptiveScaleFeatureFusion(nn.Module):
    """Adaptive Scale Feature Fusion module for DBNet++."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Multi-scale convolutions
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]
        ])
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 3, out_channels * 3, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_conv = ConvBNReLU(out_channels * 3, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale features
        feats = [conv(x) for conv in self.scale_convs]
        concat_feats = torch.cat(feats, dim=1)
        
        # Attention weights
        attn = self.attention(concat_feats)
        
        # Weighted fusion
        fused = concat_feats * attn
        
        # Output
        return self.out_conv(fused)


class DBNetPP(nn.Module):
    """DBNet++ for text detection."""
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
        # Backbone
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=False)
            in_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=False)
            in_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # FPN
        self.fpn = FPN(in_channels, out_channels=256)
        
        # Adaptive Scale Feature Fusion
        fpn_channels = 256 // 4 * len(in_channels)
        self.asff = AdaptiveScaleFeatureFusion(fpn_channels, 256)
        
        # Detection head
        self.head = DifferentiableBinarization(256)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract features at different scales
        features = []
        feat = x
        
        # Manually extract features at different stages
        for i, layer in enumerate(self.backbone):
            feat = layer(feat)
            if i in [4, 5, 6, 7]:  # Conv2, Conv3, Conv4, Conv5
                features.append(feat)
        
        # FPN
        fpn_out = self.fpn(features)
        
        # ASFF
        fused = self.asff(fpn_out)
        
        # Detection head
        outputs = self.head(fused, training=self.training)
        
        return outputs
    
    def load_weights(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)


class DBNetLoss(nn.Module):
    """Loss function for DBNet++."""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 10.0,
                 ohem_ratio: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ohem_ratio = ohem_ratio
    
    def forward(self, preds: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Probability loss (BCE)
        prob_loss = self._balanced_bce_loss(
            preds['probability'], 
            targets['gt_prob'],
            targets['gt_mask']
        )
        
        # Threshold loss (L1)
        thresh_loss = self._l1_loss(
            preds['threshold'],
            targets['gt_thresh'],
            targets['thresh_mask']
        )
        
        # Binary loss (Dice)
        binary_loss = self._dice_loss(
            preds['binary'],
            targets['gt_prob'],
            targets['gt_mask']
        )
        
        # Total loss
        total_loss = prob_loss + self.alpha * thresh_loss + self.beta * binary_loss
        
        return {
            'total': total_loss,
            'prob': prob_loss,
            'thresh': thresh_loss,
            'binary': binary_loss
        }
    
    def _balanced_bce_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                          mask: torch.Tensor) -> torch.Tensor:
        """Balanced Binary Cross Entropy loss."""
        positive = (target * mask).byte()
        negative = ((1 - target) * mask).byte()
        
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), 
                           int(positive_count * self.ohem_ratio))
        
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        
        # OHEM for negative samples
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
        
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / \
                      (positive_count + negative_count + 1e-6)
        
        return balance_loss
    
    def _l1_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
        """L1 loss for threshold map."""
        loss = torch.abs(pred - target) * mask
        return loss.sum() / (mask.sum() + 1e-6)
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """Dice loss for binary map."""
        pred = pred * mask
        target = target * mask
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = 2 * intersection / (union + 1e-6)
        return 1 - dice