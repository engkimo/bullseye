import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class Conv(nn.Module):
    """Standard convolution with batch norm and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: Optional[int] = None, groups: int = 1,
                 activation: bool = True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                            padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """CSP Bottleneck."""
    
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True,
                 groups: int = 1, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, 
                 shortcut: bool = True, groups: int = 1, expansion: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1.0) 
              for _ in range(n)]
        )
    
    def forward(self, x):
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1)
        self.m = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class YOLOHead(nn.Module):
    """YOLO detection head for text detection."""
    
    def __init__(self, in_channels: List[int], num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes  # word, line
        self.reg_max = 16
        self.num_outputs = num_classes + self.reg_max * 4
        
        # Detection heads for different scales
        self.heads = nn.ModuleList([
            nn.Sequential(
                Conv(ch, ch, 3),
                Conv(ch, ch, 3),
                nn.Conv2d(ch, self.num_outputs, 1)
            ) for ch in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, feat in enumerate(features):
            out = self.heads[i](feat)
            outputs.append(out)
        return outputs


class YOLOTextDetector(nn.Module):
    """YOLO-based text detector for word and line detection."""
    
    def __init__(self, 
                 model_size: str = 's',
                 num_classes: int = 2,
                 device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        self.num_classes = num_classes
        
        # Model configurations
        configs = {
            'n': {'depth': 0.33, 'width': 0.25},
            's': {'depth': 0.33, 'width': 0.50},
            'm': {'depth': 0.67, 'width': 0.75},
        }
        
        config = configs[model_size]
        depth_multiple = config['depth']
        width_multiple = config['width']
        # Derive backbone feature channels (P3, P4, P5) from width multiple
        c3 = int(256 * width_multiple)
        c4 = int(512 * width_multiple)
        c5 = int(1024 * width_multiple)
        channels = [c3, c4, c5]
        
        # Build backbone
        self.backbone = self._build_backbone(width_multiple, depth_multiple)
        
        # Build neck
        self.neck = self._build_neck(channels)
        
        # Build head
        self.head = YOLOHead(channels, num_classes)
        # Mirror key attributes for decoding convenience
        self.reg_max = self.head.reg_max
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator()
        
        self.to(device)
    
    def _build_backbone(self, width_multiple: float, depth_multiple: float) -> nn.Sequential:
        """Build YOLO backbone."""
        layers = []
        
        # Initial conv
        ch = int(64 * width_multiple)
        layers.append(Conv(3, ch, 6, 2))
        
        # P2
        ch = int(128 * width_multiple)
        layers.append(Conv(int(64 * width_multiple), ch, 3, 2))
        layers.append(C3(ch, ch, int(3 * depth_multiple)))
        
        # P3
        ch = int(256 * width_multiple)
        layers.append(Conv(int(128 * width_multiple), ch, 3, 2))
        layers.append(C3(ch, ch, int(6 * depth_multiple)))
        
        # P4
        ch = int(512 * width_multiple)
        layers.append(Conv(int(256 * width_multiple), ch, 3, 2))
        layers.append(C3(ch, ch, int(9 * depth_multiple)))
        
        # P5
        ch = int(1024 * width_multiple)
        layers.append(Conv(int(512 * width_multiple), ch, 3, 2))
        layers.append(C3(ch, ch, int(3 * depth_multiple)))
        layers.append(SPPF(ch, ch, 5))
        
        return nn.Sequential(*layers)
    
    def _build_neck(self, channels: List[int]) -> nn.Module:
        """Build FPN neck."""
        return FPNNeck(channels)
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # Backbone
        features = []
        feat = x
        
        for i, layer in enumerate(self.backbone):
            feat = layer(feat)
            # Collect P3, P4, P5 features
            if i in [3, 5, 9]:  # Indices for P3, P4, P5
                features.append(feat)
        
        # Neck
        features = self.neck(features)
        
        # Head
        outputs = self.head(features)
        
        # Decode predictions
        if not self.training:
            return self._decode_predictions(outputs, x.shape)
        
        return outputs
    
    def _decode_predictions(self, outputs: List[torch.Tensor], 
                          input_shape: torch.Size) -> List[Dict[str, torch.Tensor]]:
        """Decode YOLO outputs to bounding boxes."""
        batch_size = input_shape[0]
        input_h, input_w = input_shape[2:]
        
        all_detections = []
        
        for output in outputs:
            b, c, h, w = output.shape
            
            # Reshape output
            output = output.permute(0, 2, 3, 1).reshape(b, h * w, c)
            
            # Split into components
            box_reg = output[..., :self.reg_max * 4]
            cls_scores = output[..., self.reg_max * 4:]
            
            # Get grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=self.device),
                torch.arange(w, device=self.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            
            # Decode boxes
            stride = input_h / h
            boxes = self._decode_boxes(box_reg, grid, stride)
            
            # Get class predictions
            cls_probs = torch.sigmoid(cls_scores)
            cls_ids = torch.argmax(cls_probs, dim=-1)
            cls_confs = torch.max(cls_probs, dim=-1)[0]
            
            # Create detections
            for b_idx in range(batch_size):
                detections = []
                for i in range(h * w):
                    if cls_confs[b_idx, i] > 0.25:  # Confidence threshold
                        det = {
                            'bbox': boxes[b_idx, i].tolist(),
                            'confidence': cls_confs[b_idx, i].item(),
                            'class': 'word' if cls_ids[b_idx, i] == 0 else 'line'
                        }
                        detections.append(det)
                all_detections.append(detections)
        
        return all_detections
    
    def _decode_boxes(self, box_reg: torch.Tensor, grid: torch.Tensor, 
                     stride: float) -> torch.Tensor:
        """Decode bounding box regression."""
        # Simple center-based decoding
        b, hw, c = box_reg.shape
        box_reg = box_reg.reshape(b, hw, 4, self.reg_max)
        
        # Softmax over reg_max dimension
        box_reg = F.softmax(box_reg, dim=-1)
        
        # Expected value
        reg_range = torch.arange(self.reg_max, device=self.device)
        box_reg = torch.sum(box_reg * reg_range, dim=-1)
        
        # Decode to xyxy format
        grid = grid.unsqueeze(0).expand(b, -1, -1)
        
        x1 = (grid[..., 0] - box_reg[..., 0]) * stride
        y1 = (grid[..., 1] - box_reg[..., 1]) * stride
        x2 = (grid[..., 0] + box_reg[..., 2]) * stride
        y2 = (grid[..., 1] + box_reg[..., 3]) * stride
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        return boxes
    
    def load_weights(self, path: str):
        """Load pretrained weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)


class FPNNeck(nn.Module):
    """Feature Pyramid Network neck for YOLO."""
    
    def __init__(self, channels: List[int]):
        super().__init__()
        
        # Lateral convolutions
        self.lateral_conv5 = Conv(channels[2], channels[1], 1)
        self.lateral_conv4 = Conv(channels[1], channels[0], 1)
        
        # FPN convolutions
        self.fpn_conv5 = C3(channels[1] * 2, channels[1], 3, shortcut=False)
        self.fpn_conv4 = C3(channels[0] * 2, channels[0], 3, shortcut=False)
        
        # PAN convolutions
        self.pan_conv4 = Conv(channels[0], channels[0], 3, 2)
        self.pan_conv5 = Conv(channels[1], channels[1], 3, 2)
        
        # Concat channels: down3 (ch0) + f4 (ch1) -> ch0+ch1
        self.pan_c3_4 = C3(channels[0] + channels[1], channels[1], 3, shortcut=False)
        # Concat channels: down4 (ch1) + p5 (ch2) -> ch1+ch2
        self.pan_c3_5 = C3(channels[1] + channels[2], channels[2], 3, shortcut=False)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features
        
        # FPN top-down pathway
        up5 = self.lateral_conv5(p5)
        up5 = F.interpolate(up5, size=p4.shape[2:], mode='nearest')
        f4 = torch.cat([up5, p4], dim=1)
        f4 = self.fpn_conv5(f4)
        
        up4 = self.lateral_conv4(f4)
        up4 = F.interpolate(up4, size=p3.shape[2:], mode='nearest')
        f3 = torch.cat([up4, p3], dim=1)
        f3 = self.fpn_conv4(f3)
        
        # PAN bottom-up pathway
        down3 = self.pan_conv4(f3)
        f4_2 = torch.cat([down3, f4], dim=1)
        f4_2 = self.pan_c3_4(f4_2)
        
        down4 = self.pan_conv5(f4_2)
        f5_2 = torch.cat([down4, p5], dim=1)
        f5_2 = self.pan_c3_5(f5_2)
        
        return [f3, f4_2, f5_2]


class AnchorGenerator:
    """Anchor generator for YOLO text detection."""
    
    def __init__(self):
        # Text-specific anchors (width, height)
        self.anchors = {
            'word': [(30, 15), (50, 20), (80, 25)],
            'line': [(100, 20), (200, 30), (300, 40)]
        }
    
    def generate(self, feature_size: Tuple[int, int], stride: int) -> torch.Tensor:
        """Generate anchors for given feature size."""
        # Implementation for anchor generation
        pass
