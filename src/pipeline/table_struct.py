import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import cv2
from pathlib import Path
import logging

from ..modeling.table_tatr import TATR


logger = logging.getLogger(__name__)


class TableRecognizer:
    """Table structure recognition using TATR (Table Transformer) approach."""
    
    def __init__(self,
                 weights_path: Optional[Path] = None,
                 device: str = 'cuda',
                 text_recognizer: Optional[object] = None,
                 ocr_conf_threshold: float = 0.0,
                 blank_low_confidence: bool = True):
        
        self.device = device
        
        # Initialize TATR model
        self.model = TATR(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048
        ).to(device)
        
        # Load weights if provided
        if weights_path and weights_path.exists():
            logger.info(f"Loading weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("No weights loaded, using random initialization")
        
        self.model.eval()
        
        # Table structure classes
        self.structure_classes = [
            'table',
            'table_column',
            'table_row', 
            'table_column_header',
            'table_projected_row_header',
            'table_spanning_cell'
        ]
        # Optional text recognizer (ABINet/SATRN)
        self.text_recognizer = text_recognizer
        self.ocr_conf_threshold = float(ocr_conf_threshold)
        self.blank_low_confidence = bool(blank_low_confidence)
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect tables and recognize their structure.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            
        Returns:
            List of table results, each containing:
                - bbox: Table bounding box [x1, y1, x2, y2]
                - cells: List of cell information
                - html: HTML representation
                - markdown: Markdown representation
        """
        
        # First detect table regions
        table_regions = self._detect_tables(image)
        
        results = []
        for table_bbox in table_regions:
            # Crop table region
            x1, y1, x2, y2 = [int(x) for x in table_bbox]
            table_img = image[y1:y2, x1:x2]
            
            # Recognize table structure
            structure = self._recognize_structure(table_img)
            
            # Extract cells
            cells = self._extract_cells(table_img, structure)

            # OCR each cell if recognizer available
            if self.text_recognizer is not None:
                for cell in cells:
                    cx1, cy1, cx2, cy2 = [int(v) for v in cell['bbox']]
                    # clamp
                    cx1 = max(0, min(cx1, table_img.shape[1] - 1))
                    cy1 = max(0, min(cy1, table_img.shape[0] - 1))
                    cx2 = max(cx1 + 1, min(cx2, table_img.shape[1]))
                    cy2 = max(cy1 + 1, min(cy2, table_img.shape[0]))
                    crop = table_img[cy1:cy2, cx1:cx2]
                    try:
                        text, conf = self.text_recognizer.recognize(crop)
                        cell['confidence'] = float(conf)
                        if conf >= self.ocr_conf_threshold:
                            cell['text'] = text
                        else:
                            # mark low confidence
                            cell['low_confidence'] = True
                            if self.blank_low_confidence:
                                cell['text'] = ''
                    except Exception:
                        # keep existing text placeholder
                        cell['low_confidence'] = True
            
            # Generate representations
            html = self._generate_html(cells)
            markdown = self._generate_markdown(cells)
            
            results.append({
                'bbox': table_bbox,
                'cells': cells,
                'html': html,
                'markdown': markdown
            })
        
        return results
    
    def _detect_tables(self, image: np.ndarray) -> List[List[float]]:
        """Detect table regions in image."""
        # Preprocess
        img_tensor = self._preprocess(image)
        
        with torch.no_grad():
            # Forward pass for table detection
            outputs = self.model(img_tensor, task='detection')
            
            # Post-process
            boxes = outputs['boxes'][0]  # [num_queries, 4]
            scores = outputs['scores'][0]  # [num_queries]
            labels = outputs['labels'][0]  # [num_queries]
            
            # Filter tables
            table_boxes = []
            for i in range(len(scores)):
                if scores[i] > 0.5 and labels[i] == 0:  # 'table' class
                    box = self._cxcywh_to_xyxy(boxes[i], image.shape)
                    table_boxes.append(box)
            
            return table_boxes
    
    def _recognize_structure(self, table_img: np.ndarray) -> Dict[str, Any]:
        """Recognize table structure (rows, columns, cells)."""
        # Preprocess
        img_tensor = self._preprocess(table_img)
        
        with torch.no_grad():
            # Forward pass for structure recognition
            outputs = self.model(img_tensor, task='structure')
            
            # Post-process
            boxes = outputs['boxes'][0]
            scores = outputs['scores'][0]
            labels = outputs['labels'][0]
            
            # Group by structure type
            structure = {
                'columns': [],
                'rows': [],
                'cells': [],
                'headers': []
            }
            
            for i in range(len(scores)):
                if scores[i] < 0.5:
                    continue
                
                box = self._cxcywh_to_xyxy(boxes[i], table_img.shape)
                class_name = self.structure_classes[labels[i]]
                
                if class_name == 'table_column':
                    structure['columns'].append(box)
                elif class_name == 'table_row':
                    structure['rows'].append(box)
                elif class_name in ['table_column_header', 'table_projected_row_header']:
                    structure['headers'].append({
                        'bbox': box,
                        'type': class_name
                    })
                elif class_name == 'table_spanning_cell':
                    structure['cells'].append({
                        'bbox': box,
                        'spanning': True
                    })
            
            # Sort rows and columns
            structure['columns'].sort(key=lambda x: x[0])  # Sort by x1
            structure['rows'].sort(key=lambda x: x[1])     # Sort by y1
            
            return structure
    
    def _extract_cells(self, table_img: np.ndarray, 
                      structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract individual cells from table structure."""
        cells = []
        
        # Get row and column boundaries
        col_boundaries = self._get_boundaries(structure['columns'], axis=0)
        row_boundaries = self._get_boundaries(structure['rows'], axis=1)
        
        # Create cell grid
        for row_idx, (y1, y2) in enumerate(row_boundaries):
            for col_idx, (x1, x2) in enumerate(col_boundaries):
                # Check if this is a header
                is_header = False
                header_type = None
                for header in structure['headers']:
                    if self._bbox_overlap([x1, y1, x2, y2], header['bbox']) > 0.5:
                        is_header = True
                        header_type = header['type']
                        break
                
                # Placeholder text (OCR may overwrite later)
                text = f"Cell({row_idx},{col_idx})"

                # Default spans
                row_span = 1
                col_span = 1
                # Estimate spans from spanning elements if provided
                for sp in structure.get('cells', []):
                    if sp.get('spanning'):
                        if self._bbox_overlap([x1, y1, x2, y2], sp['bbox']) > 0.5:
                            row_span = max(row_span, self._count_spanned(sp['bbox'], row_boundaries, axis=1))
                            col_span = max(col_span, self._count_spanned(sp['bbox'], col_boundaries, axis=0))

                cells.append({
                    'row': row_idx,
                    'col': col_idx,
                    'bbox': [x1, y1, x2, y2],
                    'text': text,
                    'row_span': row_span,
                    'col_span': col_span,
                    'is_header': is_header,
                    'header_type': header_type
                })
        
        return cells
    
    def _get_boundaries(self, boxes: List[List[float]], axis: int) -> List[Tuple[float, float]]:
        """Get row/column boundaries from boxes."""
        if not boxes:
            return []
        
        # Extract start and end coordinates
        coords = []
        for box in boxes:
            if axis == 0:  # columns (x-axis)
                coords.extend([box[0], box[2]])
            else:  # rows (y-axis)
                coords.extend([box[1], box[3]])
        
        # Sort and remove duplicates
        coords = sorted(list(set(coords)))
        
        # Create boundaries
        boundaries = []
        for i in range(len(coords) - 1):
            boundaries.append((coords[i], coords[i + 1]))
        
        return boundaries
    
    def _generate_html(self, cells: List[Dict[str, Any]]) -> str:
        """Generate HTML representation of table."""
        if not cells:
            return "<table></table>"
        
        # Get dimensions
        max_row = max(cell['row'] for cell in cells)
        max_col = max(cell['col'] for cell in cells)
        
        # Create grid
        grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            grid[cell['row']][cell['col']] = cell
        
        # Generate HTML
        html = ["<table>"]
        
        for row_idx, row in enumerate(grid):
            html.append("  <tr>")
            for col_idx, cell in enumerate(row):
                if cell:
                    tag = "th" if cell['is_header'] else "td"
                    rs = f" rowspan=\"{cell.get('row_span',1)}\"" if cell.get('row_span',1) > 1 else ""
                    cs = f" colspan=\"{cell.get('col_span',1)}\"" if cell.get('col_span',1) > 1 else ""
                    cls = " class=\"low-confidence\"" if cell.get('low_confidence') else ""
                    conf = cell.get('confidence')
                    data_conf = f" data-confidence=\"{conf:.2f}\"" if isinstance(conf, float) else ""
                    text = cell.get('text', '')
                    html.append(f"    <{tag}{rs}{cs}{cls}{data_conf}>{text}</{tag}>")
                else:
                    html.append("    <td></td>")
            html.append("  </tr>")
        
        html.append("</table>")
        
        return "\n".join(html)
    
    def _generate_markdown(self, cells: List[Dict[str, Any]]) -> str:
        """Generate Markdown representation of table."""
        if not cells:
            return ""
        
        # Get dimensions
        max_row = max(cell['row'] for cell in cells)
        max_col = max(cell['col'] for cell in cells)
        
        # Create grid
        grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for cell in cells:
            grid[cell['row']][cell['col']] = cell
        
        # Generate Markdown
        lines = []
        
        for row_idx, row in enumerate(grid):
            # Create row
            row_texts = []
            for cell in row:
                text = cell['text'] if cell else ""
                row_texts.append(text)
            
            lines.append("| " + " | ".join(row_texts) + " |")
            
            # Add separator after header row
            if row_idx == 0 or (row_idx == 0 and any(c and c['is_header'] for c in row)):
                separator = "|" + "|".join([" --- " for _ in range(max_col + 1)]) + "|"
                lines.append(separator)
        
        return "\n".join(lines)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for TATR."""
        # Resize to fixed size
        target_size = (800, 800)
        resized = cv2.resize(image, target_size)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (resized / 255.0 - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(normalized).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor
    
    def _cxcywh_to_xyxy(self, box: torch.Tensor, img_shape: tuple) -> List[float]:
        """Convert box from cxcywh to xyxy format."""
        h, w = img_shape[:2]
        cx, cy, bw, bh = box
        
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        
        return [x1.item(), y1.item(), x2.item(), y2.item()]
    
    def _bbox_overlap(self, box1: List[float], box2: List[float]) -> float:
        """Calculate overlap ratio between two boxes."""
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
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        
        if area1 == 0:
            return 0.0
        
        return intersection / area1

    def _count_spanned(self, bbox: List[float], boundaries: List[Tuple[float, float]], axis: int) -> int:
        """Count how many boundary intervals are covered by bbox along given axis."""
        if not boundaries:
            return 1
        x1, y1, x2, y2 = bbox
        covered = 0
        for start, end in boundaries:
            if axis == 0:  # columns x-axis
                inter_start = max(start, x1)
                inter_end = min(end, x2)
            else:  # rows y-axis
                inter_start = max(start, y1)
                inter_end = min(end, y2)
            if inter_end > inter_start:
                covered += 1
        return max(1, covered)
