import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import cv2
from pathlib import Path
import logging

"""
Lazily import detection backends to avoid pulling heavy optional deps
at module import time (e.g., torchvision for ResNet backbones).
"""


logger = logging.getLogger(__name__)


class TextDetector:
    """Unified text detection interface for DBNet++ and YOLO models."""
    
    def __init__(self, 
                 model_type: str = 'dbnet',
                 weights_path: Optional[Path] = None,
                 device: str = 'cuda',
                 lite_mode: bool = False):
        
        self.model_type = model_type
        self.device = device
        self.lite_mode = lite_mode
        
        # Initialize model
        if model_type == 'dbnet':
            from ..modeling.det_dbnet import DBNetPP  # lazy import
            self.model = DBNetPP(
                backbone='resnet18' if lite_mode else 'resnet50',
                device=device
            )
        elif model_type == 'yolo':
            from ..modeling.det_yolo import YOLOTextDetector  # lazy import
            self.model = YOLOTextDetector(
                model_size='n' if lite_mode else 's',
                device=device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if provided
        if weights_path and weights_path.exists():
            logger.info(f"Loading weights from {weights_path}")
            self.model.load_weights(weights_path)
        else:
            logger.warning("No weights loaded, using random initialization (CV fallback will be used)")
            self._use_cv_fallback = True
        
        self.model.eval()
    
    def detect(self, image: np.ndarray, **kwargs) -> List[Dict[str, Any]]:
        """Detect text regions in image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            
        Returns:
            List of detection results, each containing:
                - polygon: List of points [[x1,y1], [x2,y2], ...]
                - confidence: Detection confidence score
                - type: 'word' or 'line'
        """
        
        # Fallback to classical CV if weights are not available
        if getattr(self, '_use_cv_fallback', False):
            return self._detect_cv2(image, **kwargs)
        if self.model_type == 'dbnet':
            return self._detect_dbnet(image, **kwargs)
        else:
            return self._detect_yolo(image, **kwargs)
    
    def _detect_dbnet(self, image: np.ndarray, 
                      box_thresh: float = 0.7,
                      max_candidates: int = 1000,
                      unclip_ratio: float = 1.5) -> List[Dict[str, Any]]:
        """DBNet++ detection."""
        
        # Preprocess
        img_tensor, scale = self._preprocess_dbnet(image)
        
        with torch.no_grad():
            # Forward pass
            preds = self.model(img_tensor)
            
            # Get probability map
            prob_map = preds['binary'].squeeze(0).squeeze(0).cpu().numpy()
            
            # Threshold and find contours
            binary_map = (prob_map > box_thresh).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_map, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Process contours
            results = []
            for contour in contours[:max_candidates]:
                if cv2.contourArea(contour) < 10:
                    continue
                
                # Get bounding box
                poly = cv2.minAreaRect(contour)
                poly = cv2.boxPoints(poly)
                poly = np.array(poly, dtype=np.float32)
                
                # Unclip
                poly = self._unclip(poly, unclip_ratio)
                
                # Scale back to original size
                poly = poly / scale
                
                # Get confidence
                mask = np.zeros_like(prob_map)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                confidence = np.mean(prob_map[mask > 0])
                
                results.append({
                    'polygon': poly.tolist(),
                    'confidence': float(confidence),
                    'type': 'line'
                })
            
            return results
    
    def _detect_yolo(self, image: np.ndarray,
                     conf_thresh: float = 0.25,
                     iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        """YOLO detection."""
        
        # Preprocess
        img_tensor = self._preprocess_yolo(image)
        
        with torch.no_grad():
            # Forward pass
            detections = self.model(img_tensor)
            # Flatten nested outputs (e.g., per-scale/per-batch) into a single list
            if detections and isinstance(detections[0], list):
                flat = []
                for item in detections:
                    if isinstance(item, list):
                        flat.extend(item)
                    else:
                        flat.append(item)
                detections = flat
            
            # Post-process
            results = []
            for det in detections:
                if det['confidence'] < conf_thresh:
                    continue
                
                # Convert bbox to polygon
                x1, y1, x2, y2 = det['bbox']
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                
                results.append({
                    'polygon': polygon,
                    'confidence': det['confidence'],
                    'type': det['class']  # 'word' or 'line'
                })
            
            # NMS
            results = self._nms(results, iou_thresh)
            
            return results
    
    def _preprocess_dbnet(self, image: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Preprocess image for DBNet."""
        h, w = image.shape[:2]
        
        # Resize to multiple of 32
        target_size = 640 if self.lite_mode else 1024
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Normalize
        img_tensor = torch.from_numpy(resized).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, scale
    
    def _preprocess_yolo(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for YOLO."""
        # Resize
        target_size = 416 if self.lite_mode else 640
        resized = cv2.resize(image, (target_size, target_size))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(resized).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor / 255.0
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def _unclip(self, poly: np.ndarray, ratio: float) -> np.ndarray:
        """Unclip polygon by ratio."""
        poly = cv2.minAreaRect(poly.astype(np.int32))
        poly = cv2.boxPoints(poly)
        
        # Calculate area
        area = cv2.contourArea(poly)
        length = cv2.arcLength(poly, True)
        distance = area * ratio / length
        
        # Offset polygon
        offset_poly = []
        for i in range(len(poly)):
            point1 = poly[i]
            point2 = poly[(i + 1) % len(poly)]
            
            # Calculate normal
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm == 0:
                continue
            
            dx = dx / norm * distance
            dy = dy / norm * distance
            
            # Offset points
            offset_poly.append([
                point1[0] - dy,
                point1[1] + dx
            ])
        
        return np.array(offset_poly, dtype=np.float32)
    
    def _nms(self, detections: List[Dict], iou_thresh: float) -> List[Dict]:
        """Non-maximum suppression."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, det1 in enumerate(detections):
            suppress = False
            for det2 in keep:
                if self._polygon_iou(det1['polygon'], det2['polygon']) > iou_thresh:
                    suppress = True
                    break
            if not suppress:
                keep.append(det1)
        
        return keep
    
    def _polygon_iou(self, poly1: List, poly2: List) -> float:
        """Calculate IoU between two polygons."""
        # Convert to numpy arrays
        poly1 = np.array(poly1, dtype=np.float32)
        poly2 = np.array(poly2, dtype=np.float32)
        
        # Create masks
        x_min = min(poly1[:, 0].min(), poly2[:, 0].min())
        y_min = min(poly1[:, 1].min(), poly2[:, 1].min())
        x_max = max(poly1[:, 0].max(), poly2[:, 0].max())
        y_max = max(poly1[:, 1].max(), poly2[:, 1].max())
        
        w = int(x_max - x_min + 1)
        h = int(y_max - y_min + 1)
        
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)
        
        # Adjust coordinates
        poly1_adj = poly1 - [x_min, y_min]
        poly2_adj = poly2 - [x_min, y_min]
        
        cv2.fillPoly(mask1, [poly1_adj.astype(np.int32)], 1)
        cv2.fillPoly(mask2, [poly2_adj.astype(np.int32)], 1)
        
        # Calculate IoU
        intersection = np.sum(mask1 * mask2)
        union = np.sum(mask1) + np.sum(mask2) - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _detect_cv2(self, image: np.ndarray,
                     min_area: int = 80,
                     max_area_ratio: float = 0.5,
                     morph_kernel_w: int = 25,
                     morph_kernel_h: int = 3) -> List[Dict[str, Any]]:
        """Classical text region detection as a fallback.

        - Convert to gray -> adaptive threshold
        - Morphological close to connect characters
        - Find contours and filter by area/shape
        """
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Normalize contrast
        gray = cv2.equalizeHist(gray)
        # Adaptive threshold to handle varying background
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
        # Morphological closing to connect text lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_w, morph_kernel_h))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        max_area = max_area_ratio * (h * w)
        boxes: List[Dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter too tall or too wide regions heuristically
            if bh < 8 or bw < 8:
                continue
            # Expand slightly
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            boxes.append({'polygon': polygon, 'confidence': 0.5, 'type': 'line'})
        # Apply simple NMS
        boxes = self._nms(boxes, iou_thresh=0.3)
        return boxes
