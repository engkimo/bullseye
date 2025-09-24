import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math
from torchvision.models import resnet50
from torchvision.ops import box_convert


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embeddings for DETR."""
    
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, 
                 normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Create position embeddings
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1] + eps) * self.scale
            x_embed = x_embed / (x_embed[-1] + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        
        pos = torch.cat((pos_y[:, None, :].expand(-1, w, -1), 
                        pos_x[None, :, :].expand(h, -1, -1)), dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        
        return pos


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for DETR."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        q = k = src + pos
        src2 = self.self_attn(q, k, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer for DETR."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        # Self attention
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.multihead_attn(
            query=tgt + query_pos,
            key=memory + pos,
            value=memory
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DETRLayoutDetector(nn.Module):
    """DETR-based layout detection model."""
    
    def __init__(self,
                 num_classes: int = 13,
                 hidden_dim: int = 256,
                 nheads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 num_queries: int = 100,
                 device: str = 'cuda'):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.device = device
        
        # CNN backbone (ResNet-50)
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=False).children())[:-2]
        )
        self.conv_proj = nn.Conv2d(2048, hidden_dim, 1)
        
        # Positional encoding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        
        # Transformer
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, nheads)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nheads)
            for _ in range(num_decoder_layers)
        ])
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone
        features = self.backbone(x)
        
        # Project to hidden dimension
        features = self.conv_proj(features)
        
        # Positional encoding
        pos_embed = self.position_embedding(features)
        
        # Flatten for transformer
        b, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # (hw, b, c)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        
        # Transformer encoder
        memory = features
        for encoder_layer in self.encoder:
            memory = encoder_layer(memory, pos_embed)
        
        # Decoder queries
        query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, b, -1)
        tgt = torch.zeros_like(query_embed)
        
        # Transformer decoder
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, pos_embed, query_embed)
        
        # Output projections
        outputs_class = self.class_embed(tgt)
        outputs_coord = self.bbox_embed(tgt).sigmoid()
        
        out = {
            'logits': outputs_class.transpose(0, 1),  # (b, num_queries, num_classes+1)
            'boxes': outputs_coord.transpose(0, 1)     # (b, num_queries, 4)
        }
        
        return out
    
    def load_state_dict(self, state_dict, strict=True):
        """Load model weights."""
        super().load_state_dict(state_dict, strict=strict)


class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    
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


class HungarianMatcher(nn.Module):
    """Hungarian matching for DETR training."""
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """Perform the matching."""
        bs, num_queries = outputs["logits"].shape[:2]
        
        # Flatten to compute the cost matrices
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["boxes"].flatten(0, 1)
        
        # Concatenate target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute the classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(
            box_convert(out_bbox, "cxcywh", "xyxy"),
            box_convert(tgt_bbox, "cxcywh", "xyxy")
        )
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Hungarian matching
        indices = []
        for i, c in enumerate(C.split([len(v["boxes"]) for v in targets], -1)):
            from scipy.optimize import linear_sum_assignment
            indices.append(linear_sum_assignment(c[i]))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute generalized IoU between boxes."""
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / union
    
    # Enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area