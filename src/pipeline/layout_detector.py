import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import cv2
from pathlib import Path
import logging

"""
Avoid importing heavy backends (e.g., DETR/torchvision) at module import time.
We lazily import the selected backend inside __init__ to prevent optional
dependencies from breaking environments that only need YOLO.
"""


logger = logging.getLogger(__name__)


class LayoutDetector:
    """Unified layout detection interface for YOLO and DETR models."""
    
    # DocLayNet compatible labels
    LABELS = [
        'paragraph',
        'title', 
        'section_heading',
        'caption',
        'figure',
        'table',
        'list',
        'footnote',
        'page_number',
        'header',
        'footer',
        'equation',
        'reference'
    ]
    
    def __init__(self,
                 model_type: str = 'yolo',
                 weights_path: Optional[Path] = None,
                 device: str = 'cuda'):
        
        self.model_type = model_type
        self.device = device
        self.num_classes = len(self.LABELS)
        
        # Initialize model
        if model_type == 'yolo':
            # Lazy import to avoid pulling in torchvision when not needed
            from ..modeling.layout_yolo import YOLOLayoutDetector
            self.model = YOLOLayoutDetector(
                num_classes=self.num_classes,
                model_size='m',  # medium for better accuracy
                device=device
            )
        elif model_type == 'detr':
            try:
                # Lazy import; DETR path can require torchvision custom ops
                from ..modeling.layout_detr import DETRLayoutDetector
                self.model = DETRLayoutDetector(
                    num_classes=self.num_classes,
                    hidden_dim=256,
                    nheads=8,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    device=device
                )
            except Exception as e:
                logger.error("Failed to initialize DETR layout detector: %s", e)
                raise
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if provided
        if weights_path and weights_path.exists():
            logger.info(f"Loading weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("No weights loaded, using random initialization")
        
        self.model.eval()
    
    def detect(self, image: np.ndarray, 
               conf_thresh: float = 0.5,
               nms_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """Detect layout elements in image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            conf_thresh: Confidence threshold
            nms_thresh: NMS IoU threshold
            
        Returns:
            List of detection results, each containing:
                - bbox: [x1, y1, x2, y2]
                - label: Layout element type
                - confidence: Detection confidence
        """
        
        if self.model_type == 'yolo':
            return self._detect_yolo(image, conf_thresh, nms_thresh)
        else:
            return self._detect_detr(image, conf_thresh)
    
    def _detect_yolo(self, image: np.ndarray,
                     conf_thresh: float,
                     nms_thresh: float) -> List[Dict[str, Any]]:
        """YOLO detection."""
        
        # Preprocess
        img_tensor, scale_x, scale_y = self._preprocess_yolo(image)
        
        with torch.no_grad():
            # Forward pass
            predictions = self.model(img_tensor)
            
            # Post-process
            results = []
            for pred in predictions[0]:  # First batch
                if pred['confidence'] < conf_thresh:
                    continue
                
                # Scale coordinates back
                x1 = pred['bbox'][0] * scale_x
                y1 = pred['bbox'][1] * scale_y
                x2 = pred['bbox'][2] * scale_x
                y2 = pred['bbox'][3] * scale_y
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': self.LABELS[pred['class']],
                    'confidence': pred['confidence']
                })
            
            # Apply NMS
            results = self._nms(results, nms_thresh)
            
            return results
    
    def _detect_detr(self, image: np.ndarray,
                     conf_thresh: float) -> List[Dict[str, Any]]:
        """DETR detection."""
        
        # Preprocess
        img_tensor, scale = self._preprocess_detr(image)
        orig_h, orig_w = image.shape[:2]
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(img_tensor)
            
            # Get predictions
            logits = outputs['logits'][0]  # [num_queries, num_classes]
            boxes = outputs['boxes'][0]    # [num_queries, 4] in cxcywh format
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Filter by confidence
            results = []
            for i in range(logits.shape[0]):
                # Get max probability and class
                max_prob, class_id = probs[i].max(dim=0)
                
                # Skip background class and low confidence
                if class_id == self.num_classes or max_prob < conf_thresh:
                    continue
                
                # Convert box from cxcywh to xyxy
                cx, cy, w, h = boxes[i]
                x1 = (cx - w/2) * orig_w
                y1 = (cy - h/2) * orig_h
                x2 = (cx + w/2) * orig_w
                y2 = (cy + h/2) * orig_h
                
                results.append({
                    'bbox': [x1.item(), y1.item(), x2.item(), y2.item()],
                    'label': self.LABELS[class_id],
                    'confidence': max_prob.item()
                })
            
            return results
    
    def _preprocess_yolo(self, image: np.ndarray) -> tuple:
        """Preprocess image for YOLO."""
        orig_h, orig_w = image.shape[:2]
        target_size = 640
        
        # Resize
        resized = cv2.resize(image, (target_size, target_size))
        
        # Calculate scale factors
        scale_x = orig_w / target_size
        scale_y = orig_h / target_size
        
        # Convert to tensor
        img_tensor = torch.from_numpy(resized).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor / 255.0
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, scale_x, scale_y
    
    def _preprocess_detr(self, image: np.ndarray) -> tuple:
        """Preprocess image for DETR."""
        # DETR expects normalized images
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Resize to max dimension 800
        h, w = image.shape[:2]
        max_size = 800
        scale = min(max_size / h, max_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Normalize
        normalized = (resized / 255.0 - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(normalized).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, scale
    
    def _nms(self, detections: List[Dict], iou_thresh: float) -> List[Dict]:
        """Non-maximum suppression."""
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            label = det['label']
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(det)
        
        # Apply NMS per class
        results = []
        for label, dets in by_class.items():
            # Sort by confidence
            dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            for i, det1 in enumerate(dets):
                suppress = False
                for det2 in keep:
                    if self._bbox_iou(det1['bbox'], det2['bbox']) > iou_thresh:
                        suppress = True
                        break
                if not suppress:
                    keep.append(det1)
            
            results.extend(keep)
        
        return results
    
    def _bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def visualize(self, image: np.ndarray, 
                  detections: List[Dict[str, Any]]) -> np.ndarray:
        """Visualize layout detections on image."""
        vis_img = image.copy()
        
        # Color map for different labels
        colors = {
            'paragraph': (0, 255, 0),
            'title': (255, 0, 0),
            'section_heading': (255, 128, 0),
            'caption': (128, 0, 255),
            'figure': (0, 255, 255),
            'table': (255, 0, 255),
            'list': (128, 255, 0),
            'footnote': (128, 128, 128),
            'page_number': (64, 64, 64),
            'header': (192, 192, 192),
            'footer': (192, 192, 192),
            'equation': (0, 128, 255),
            'reference': (255, 128, 128)
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            label = det['label']
            conf = det['confidence']
            
            # Get color
            color = colors.get(label, (255, 255, 255))
            
            # Draw bbox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label}: {conf:.2f}"
            cv2.putText(vis_img, text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_img
