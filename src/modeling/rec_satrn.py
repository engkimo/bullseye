import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class LocalityAwareFeedforward(nn.Module):
    """Locality-aware feedforward layer for SATRN."""
    
    def __init__(self, d_model: int, d_ff: int, kernel_size: int = 11):
        super().__init__()
        
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_ff, d_ff, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2,
            groups=d_ff
        )
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        return x


class AdaptivePositionalEncoding(nn.Module):
    """Adaptive 2D positional encoding for SATRN."""
    
    def __init__(self, d_model: int, max_h: int = 48, max_w: int = 160):
        super().__init__()
        
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        
        # Create 2D positional encodings
        pe_h = torch.zeros(max_h, d_model // 2)
        pe_w = torch.zeros(max_w, d_model // 2)
        
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                           (-math.log(10000.0) / (d_model // 2)))
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer('pe_h', pe_h)
        self.register_buffer('pe_w', pe_w)
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Add 2D positional encoding to input tensor."""
        # x shape: (batch, h*w, d_model)
        batch_size = x.size(0)
        
        # Get positional encodings
        pe_h = self.pe_h[:h].unsqueeze(1).expand(h, w, self.d_model // 2)
        pe_w = self.pe_w[:w].unsqueeze(0).expand(h, w, self.d_model // 2)
        
        # Concatenate h and w encodings
        pe = torch.cat([pe_h, pe_w], dim=-1)  # (h, w, d_model)
        pe = pe.reshape(h * w, self.d_model)
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + pe


class ShallowCNN(nn.Module):
    """Shallow CNN for feature extraction in SATRN."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 512):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
        self.output_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        b, c, h, w = x.shape
        return x, h, w


class SATRNEncoderLayer(nn.Module):
    """Modified transformer encoder layer for SATRN."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.feedforward = LocalityAwareFeedforward(d_model, dim_feedforward)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
    
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.feedforward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class SATRNDecoderLayer(nn.Module):
    """Modified transformer decoder layer for SATRN."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        self.feedforward = LocalityAwareFeedforward(d_model, dim_feedforward)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.feedforward(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class SATRN(nn.Module):
    """Self-Attention based Text Recognition Network."""
    
    def __init__(self,
                 num_classes: int,
                 max_len: int = 50,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_len = max_len
        self.d_model = d_model
        
        # CNN backbone
        self.cnn = ShallowCNN(in_channels=1, out_channels=d_model)
        
        # Positional encoding
        self.pos_encoder = AdaptivePositionalEncoding(d_model)
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            SATRNEncoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_encoder_layers)
        ])
        
        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            SATRNDecoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_decoder_layers)
        ])
        
        # Output embedding
        self.tgt_embedding = nn.Embedding(num_classes, d_model)
        self.output_proj = nn.Linear(d_model, num_classes)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, images: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # CNN feature extraction
        features, h, w = self.cnn(images)
        
        # Reshape to sequence
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # (B, H, W, C)
        features = features.reshape(batch_size, h * w, self.d_model)
        
        # Add positional encoding
        features = self.pos_encoder(features, h, w)
        
        # Transformer encoder
        memory = features.transpose(0, 1)  # (seq_len, batch, d_model)
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory)
        
        # Prepare decoder input
        if targets is not None:
            # Training mode with teacher forcing
            tgt = self.tgt_embedding(targets)
            tgt = self.pos_encoder(tgt, 1, targets.size(1))
            tgt = tgt.transpose(0, 1)
            
            # Create causal mask
            tgt_len = targets.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(images.device)
            
            # Transformer decoder
            output = tgt
            for decoder_layer in self.decoder_layers:
                output = decoder_layer(output, memory, tgt_mask=tgt_mask)
            
            output = output.transpose(0, 1)  # (batch, seq_len, d_model)
            
        else:
            # Inference mode - autoregressive generation
            output = self._generate(memory, batch_size)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return {'logits': logits}
    
    def _generate(self, memory: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Autoregressive generation for inference."""
        device = memory.device
        
        # Start with SOS token (assume index 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        output_features = []
        
        for step in range(self.max_len):
            # Embed current sequence
            tgt = self.tgt_embedding(generated)
            tgt = self.pos_encoder(tgt, 1, generated.size(1))
            tgt = tgt.transpose(0, 1)
            
            # Decoder forward
            output = tgt
            for decoder_layer in self.decoder_layers:
                output = decoder_layer(output, memory)
            
            output = output.transpose(0, 1)
            
            # Get last timestep
            last_output = output[:, -1:]
            output_features.append(last_output)
            
            # Predict next token
            logits = self.output_proj(last_output)
            next_token = torch.argmax(logits, dim=-1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS (assume index 2)
            if (next_token == 2).all():
                break
        
        # Stack all output features
        output = torch.cat(output_features, dim=1)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask