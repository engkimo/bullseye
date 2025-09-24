import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from torchvision.models import resnet34


class TATR(nn.Module):
    """Table Transformer (TATR) for table structure recognition."""
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 num_queries: int = 125,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        
        # CNN backbone (ResNet-34 for efficiency)
        self.backbone = nn.Sequential(
            *list(resnet34(pretrained=False).children())[:-2]
        )
        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Object queries for table elements
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Structure classes: 
        # 0: table, 1: column, 2: row, 3: column_header, 
        # 4: row_header, 5: spanning_cell
        self.num_structure_classes = 6
        
        # Detection heads
        self.class_head = nn.Linear(d_model, self.num_structure_classes)
        self.bbox_head = MLP(d_model, d_model, 4, 3)
        
        # Cell association head
        self.cell_row_head = nn.Linear(d_model, 20)  # Max 20 rows
        self.cell_col_head = nn.Linear(d_model, 20)  # Max 20 columns
    
    def forward(self, x: torch.Tensor, task: str = 'structure') -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            task: 'detection' for table detection, 'structure' for structure recognition
        """
        
        # Extract CNN features
        features = self.backbone(x)
        features = self.input_proj(features)
        
        b, c, h, w = features.shape
        
        # Add positional encoding
        pos = self.pos_encoding(features)
        
        # Flatten for transformer
        features_flat = features.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        pos_flat = pos.flatten(2).permute(0, 2, 1)
        
        # Prepare decoder queries
        query_embed = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        
        # Transformer forward
        memory = self.transformer.encoder(features_flat + pos_flat)
        hs = self.transformer.decoder(query_embed, memory)
        
        # Output heads
        outputs_class = self.class_head(hs)
        outputs_coord = self.bbox_head(hs).sigmoid()
        
        out = {
            'logits': outputs_class,
            'boxes': outputs_coord,
            'labels': torch.argmax(outputs_class, dim=-1),
            'scores': F.softmax(outputs_class, dim=-1).max(dim=-1)[0]
        }
        
        if task == 'structure':
            # Additional outputs for structure recognition
            out['row_ids'] = self.cell_row_head(hs)
            out['col_ids'] = self.cell_col_head(hs)
        
        return out


class PositionalEncoding2D(nn.Module):
    """Dynamic 2D sinusoidal positional encoding (no fixed grid size).

    d_model must be divisible by 4. Produces (B, d_model, H, W) encodings that
    combine sine/cosine terms for Y and X axes.
    """

    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        d = self.d_model
        d_quarter = d // 4

        # Frequencies
        div = torch.exp(
            torch.arange(0, d_quarter, dtype=torch.float32, device=device)
            * (-(math.log(10000.0) / max(1, d_quarter)))
        )  # (d_quarter,)

        # Positions
        pos_y = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)  # (h,1)
        pos_x = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(1)  # (w,1)

        # Y encodings: (h, d_quarter) -> (d_quarter, h, w)
        y = pos_y * div.unsqueeze(0)
        y_sin = torch.sin(y).transpose(0, 1).unsqueeze(2).expand(-1, h, w)
        y_cos = torch.cos(y).transpose(0, 1).unsqueeze(2).expand(-1, h, w)

        # X encodings: (w, d_quarter) -> (d_quarter, h, w)
        x_enc = pos_x * div.unsqueeze(0)
        x_sin = torch.sin(x_enc).transpose(0, 1).unsqueeze(1).expand(-1, h, w)
        x_cos = torch.cos(x_enc).transpose(0, 1).unsqueeze(1).expand(-1, h, w)

        pe = torch.zeros(d, h, w, device=device, dtype=torch.float32)
        pe[0:d_quarter] = y_sin
        pe[d_quarter:2 * d_quarter] = y_cos
        pe[2 * d_quarter:3 * d_quarter] = x_sin
        pe[3 * d_quarter:4 * d_quarter] = x_cos

        return pe.unsqueeze(0).expand(b, -1, -1, -1)


class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TableStructureDecoder:
    """Decode TATR outputs to table structure."""
    
    def __init__(self, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        
        self.structure_labels = [
            'table', 'table_column', 'table_row', 
            'table_column_header', 'table_projected_row_header', 
            'table_spanning_cell'
        ]
    
    def decode(self, outputs: Dict[str, torch.Tensor]) -> List[Dict]:
        """Decode model outputs to table structure."""
        
        logits = outputs['logits']
        boxes = outputs['boxes']
        scores = outputs['scores']
        
        batch_size = logits.shape[0]
        results = []
        
        for b in range(batch_size):
            # Filter by confidence
            mask = scores[b] > self.conf_threshold
            
            filtered_boxes = boxes[b][mask]
            filtered_labels = outputs['labels'][b][mask]
            filtered_scores = scores[b][mask]
            
            # Get row/column assignments if available
            row_ids = None
            col_ids = None
            if 'row_ids' in outputs:
                row_ids = torch.argmax(outputs['row_ids'][b][mask], dim=-1)
                col_ids = torch.argmax(outputs['col_ids'][b][mask], dim=-1)
            
            # Build structure
            structure = {
                'columns': [],
                'rows': [],
                'cells': [],
                'headers': []
            }
            
            for i in range(len(filtered_boxes)):
                label = self.structure_labels[filtered_labels[i]]
                box = filtered_boxes[i].tolist()
                score = filtered_scores[i].item()
                
                element = {
                    'bbox': box,
                    'label': label,
                    'score': score
                }
                
                if row_ids is not None:
                    element['row'] = row_ids[i].item()
                    element['col'] = col_ids[i].item()
                
                # Categorize by type
                if label == 'table_column':
                    structure['columns'].append(element)
                elif label == 'table_row':
                    structure['rows'].append(element)
                elif 'header' in label:
                    structure['headers'].append(element)
                else:
                    structure['cells'].append(element)
            
            results.append(structure)
        
        return results
    
    def construct_table(self, structure: Dict) -> Dict:
        """Construct table from decoded structure."""
        
        # Sort columns and rows
        columns = sorted(structure['columns'], key=lambda x: x['bbox'][0])
        rows = sorted(structure['rows'], key=lambda x: x['bbox'][1])
        
        # Create grid
        num_rows = len(rows)
        num_cols = len(columns)
        
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Assign cells to grid positions
        for cell in structure['cells']:
            # Find row and column indices
            row_idx = self._find_position(cell['bbox'], rows, axis=1)
            col_idx = self._find_position(cell['bbox'], columns, axis=0)
            
            if row_idx is not None and col_idx is not None:
                grid[row_idx][col_idx] = cell
        
        # Handle headers
        for header in structure['headers']:
            if header['label'] == 'table_column_header':
                col_idx = self._find_position(header['bbox'], columns, axis=0)
                if col_idx is not None:
                    # Place in first row
                    if 0 < num_rows:
                        grid[0][col_idx] = header
        
        return {
            'grid': grid,
            'num_rows': num_rows,
            'num_cols': num_cols
        }
    
    def _find_position(self, cell_bbox: List[float], 
                      elements: List[Dict], axis: int) -> Optional[int]:
        """Find position of cell in row/column list."""
        
        cell_center = (cell_bbox[axis] + cell_bbox[axis + 2]) / 2
        
        for i, element in enumerate(elements):
            elem_bbox = element['bbox']
            if elem_bbox[axis] <= cell_center <= elem_bbox[axis + 2]:
                return i
        
        return None
