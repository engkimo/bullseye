import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class YOLOLayoutDetector(nn.Module):
    """YOLO-based layout detection model."""
    
    def __init__(self,
                 num_classes: int = 13,  # DocLayNet classes
                 model_size: str = 'm',
                 device: str = 'cuda'):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        
        # Use similar architecture as text detection YOLO
        # but adapted for layout detection
        from .det_yolo import Conv, C3, SPPF, FPNNeck
        
        # Model configurations
        configs = {
            's': {'depth': 0.33, 'width': 0.50},
            'm': {'depth': 0.67, 'width': 0.75},
            'l': {'depth': 1.0, 'width': 1.0},
        }
        
        config = configs[model_size]
        self.depth_multiple = config['depth']
        self.width_multiple = config['width']
        # Channels will be inferred in _build_model()
        self.channels = []
        
        # Build model
        self._build_model()
        
        self.to(device)
    
    def _build_model(self):
        """Build YOLO model for layout detection."""
        # Import components
        from .det_yolo import Conv, C3, SPPF
        
        # Backbone
        ch = int(64 * self.width_multiple)
        self.stem = Conv(3, ch, 6, 2)
        
        # P2
        ch = int(128 * self.width_multiple)
        self.layer1 = nn.Sequential(
            Conv(int(64 * self.width_multiple), ch, 3, 2),
            C3(ch, ch, int(3 * self.depth_multiple))
        )
        
        # P3
        ch = int(256 * self.width_multiple)
        self.layer2 = nn.Sequential(
            Conv(int(128 * self.width_multiple), ch, 3, 2),
            C3(ch, ch, int(6 * self.depth_multiple))
        )

        # P4
        ch = int(512 * self.width_multiple)
        self.layer3 = nn.Sequential(
            Conv(int(256 * self.width_multiple), ch, 3, 2),
            C3(ch, ch, int(9 * self.depth_multiple))
        )

        # P5
        ch = int(1024 * self.width_multiple)
        self.layer4 = nn.Sequential(
            Conv(int(512 * self.width_multiple), ch, 3, 2),
            C3(ch, ch, int(3 * self.depth_multiple)),
            SPPF(ch, ch, 5)
        )

        # Infer channels for detection head from backbone outputs
        p3_ch = int(256 * self.width_multiple)
        p4_ch = int(512 * self.width_multiple)
        p5_ch = int(1024 * self.width_multiple)
        self.channels = [p3_ch, p4_ch, p5_ch]

        # Detection heads
        self.detect = LayoutDetectionHead(self.channels, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # Backbone forward
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        
        # Multi-scale features
        features = [p3, p4, p5]
        
        # Detection
        if self.training:
            # return raw predictions per scale for loss
            return self.detect(features, return_raw=True)
        else:
            detections = self.detect(features, return_raw=False)
            return detections
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model weights."""
        super().load_state_dict(state_dict, strict=strict)


class LayoutDetectionHead(nn.Module):
    """Detection head for layout elements."""
    
    def __init__(self, in_channels: List[int], num_classes: int):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = 16
        
        # Detection layers for each scale
        self.detect_layers = nn.ModuleList()
        for ch in in_channels:
            self.detect_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, self.num_classes + 4 * self.reg_max, 1)
                )
            )
    
    def forward(self, features: List[torch.Tensor], return_raw: bool = False):
        outputs = []
        
        for i, (feat, detect_layer) in enumerate(zip(features, self.detect_layers)):
            pred = detect_layer(feat)  # (B, C, H, W)
            b, c, h, w = pred.shape
            if return_raw:
                # keep (B, H, W, C)
                outputs.append({'pred': pred.permute(0, 2, 3, 1).contiguous(),
                                'hw': (h, w)})
            else:
                # Split predictions
                pr = pred.permute(0, 2, 3, 1).reshape(b, h * w, c)
                bbox_pred = pr[..., :4 * self.reg_max]
                cls_pred = pr[..., 4 * self.reg_max:]
                
                # Process for each image in batch
                batch_outputs = []
                for batch_idx in range(b):
                    detections = self._decode_predictions(
                        bbox_pred[batch_idx],
                        cls_pred[batch_idx],
                        h, w,
                        feat.shape[2:]
                    )
                    batch_outputs.append(detections)
                outputs.append(batch_outputs)
        
        if return_raw:
            return outputs  # list of levels with raw pred and hw
        
        # Merge detections from all scales
        merged_outputs = []
        for batch_idx in range(len(outputs[0])):
            all_detections = []
            for scale_output in outputs:
                all_detections.extend(scale_output[batch_idx])
            
            # Apply NMS
            final_detections = self._nms(all_detections, iou_threshold=0.45)
            merged_outputs.append(final_detections)
        
        return merged_outputs
    
    def _decode_predictions(self, bbox_pred, cls_pred, grid_h, grid_w, 
                          feat_size) -> List[Dict[str, torch.Tensor]]:
        """Decode predictions to bounding boxes."""
        detections = []
        
        # Get grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=bbox_pred.device),
            torch.arange(grid_w, device=bbox_pred.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        
        # Decode bounding boxes
        bbox_pred = bbox_pred.reshape(-1, 4, self.reg_max)
        bbox_pred = F.softmax(bbox_pred, dim=-1)
        
        reg_range = torch.arange(self.reg_max, device=bbox_pred.device)
        bbox_pred = torch.sum(bbox_pred * reg_range, dim=-1)
        
        # Calculate stride
        stride_h = feat_size[0] / grid_h
        stride_w = feat_size[1] / grid_w
        
        # Convert to xyxy format
        x1 = (grid[:, 0] - bbox_pred[:, 0]) * stride_w
        y1 = (grid[:, 1] - bbox_pred[:, 1]) * stride_h
        x2 = (grid[:, 0] + bbox_pred[:, 2]) * stride_w
        y2 = (grid[:, 1] + bbox_pred[:, 3]) * stride_h
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # Get class predictions
        cls_scores = torch.sigmoid(cls_pred)
        max_scores, class_ids = torch.max(cls_scores, dim=-1)
        
        # Filter by confidence
        conf_threshold = 0.25
        mask = max_scores > conf_threshold
        
        boxes = boxes[mask]
        scores = max_scores[mask]
        classes = class_ids[mask]
        
        # Convert to list of dicts
        for i in range(len(boxes)):
            detections.append({
                'bbox': boxes[i].tolist(),
                'confidence': scores[i].item(),
                'class': classes[i].item()
            })
        
        return detections
    
    def _nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Apply non-maximum suppression."""
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        # Apply NMS per class
        keep = []
        for cls, class_dets in by_class.items():
            # Sort by confidence
            class_dets = sorted(class_dets, key=lambda x: x['confidence'], reverse=True)
            
            # NMS
            i = 0
            while i < len(class_dets):
                keep.append(class_dets[i])
                
                # Remove overlapping detections
                j = i + 1
                while j < len(class_dets):
                    iou = self._compute_iou(class_dets[i]['bbox'], class_dets[j]['bbox'])
                    if iou > iou_threshold:
                        class_dets.pop(j)
                    else:
                        j += 1
                i += 1
        
        return keep
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
