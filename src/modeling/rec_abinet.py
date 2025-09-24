import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer with dynamic length support."""
    
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        self.d_model = d_model
        # Precompute a reasonable default; will expand on-the-fly if needed.
        pe = self._build_pe(max_len)
        self.register_buffer('pe', pe)  # (1, max_len, d_model)

    def _build_pe(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(1):
            # build on the fly to required length (keep on same device)
            pe = self._build_pe(L).to(x.device)
        else:
            pe = self.pe[:, :L].to(x.device)
        return x + pe


class VisionEncoder(nn.Module):
    """Vision encoder for ABINet.

    Note: Projector is now a registered module (created in __init__),
    so its parameters are optimized correctly.
    """
    
    def __init__(self, in_channels: int = 1, hidden_dim: int = 512, seq_out: int = 50):
        super().__init__()
        
        # CNN backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Transformer encoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.hidden_dim = hidden_dim
        self.seq_out = int(seq_out)
        
        # Determine projection input dimension using a dummy pass (H=48, W=320)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 48, 320)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            b, c, h, w = x.shape
            proj_in = c * h
        
        if proj_in != self.hidden_dim:
            self.proj = nn.Linear(proj_in, self.hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        
        # Project to hidden dim if needed (registered module)
        x = self.proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = x.transpose(0, 1)  # (seq_len, batch, dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch, seq_len, dim)
        
        # Align sequence length to a fixed output length (max_len) for stable supervision
        if self.seq_out and x.size(1) != self.seq_out:
            # interpolate along sequence axis to seq_out steps
            x = x.transpose(1, 2)  # (B, D, T)
            x = torch.nn.functional.interpolate(x, size=self.seq_out, mode='linear', align_corners=False)
            x = x.transpose(1, 2)  # (B, seq_out, D)
        
        return x


class LanguageModel(nn.Module):
    """Language model for ABINet.

    memory_input_dim: dimension of encoder memory (vision features). If it
    differs from hidden_dim, a projection is applied via a registered module.
    """
    
    def __init__(self, num_classes: int, hidden_dim: int = 256, max_len: int = 50, memory_input_dim: int = 512):
        super().__init__()
        
        self.embedding = nn.Embedding(num_classes, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        # Memory projector (registered)
        if memory_input_dim != hidden_dim:
            self.memory_proj = nn.Linear(memory_input_dim, hidden_dim)
        else:
            self.memory_proj = nn.Identity()
    
    def forward(self, memory: torch.Tensor, 
                tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project encoder memory to decoder hidden dim if needed (registered)
        memory = self.memory_proj(memory)

        if tgt is None:
            # Inference mode - generate autoregressively
            return self._generate(memory)
        
        # Training mode
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # Create mask for autoregressive decoding
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
        tgt_mask = tgt_mask.to(tgt.device)
        
        # Transformer decoding
        tgt_embed = tgt_embed.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        output = self.transformer(tgt_embed, memory, tgt_mask=tgt_mask)
        output = output.transpose(0, 1)
        
        return output
    
    def _generate(self, memory: torch.Tensor) -> torch.Tensor:
        """Autoregressive generation for inference."""
        batch_size = memory.size(0)
        device = memory.device
        # Ensure memory matches decoder hidden dim (registered)
        memory = self.memory_proj(memory)
        
        # Start with SOS token (assume index 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long).to(device)
        
        for _ in range(self.max_len - 1):
            tgt_embed = self.embedding(generated)
            tgt_embed = self.pos_encoder(tgt_embed)
            
            tgt_embed = tgt_embed.transpose(0, 1)
            memory_t = memory.transpose(0, 1)
            
            output = self.transformer(tgt_embed, memory_t)
            output = output.transpose(0, 1)
            
            # Get last token prediction
            next_token = output[:, -1:]
            generated = torch.cat([generated, torch.zeros_like(next_token[:, :, 0:1], dtype=torch.long)], dim=1)
        
        return output


class FusionModule(nn.Module):
    """Fusion module for combining vision and language features."""
    
    def __init__(self, vision_dim: int = 512, language_dim: int = 256):
        super().__init__()
        
        # Project to same dimension
        self.vision_proj = nn.Linear(vision_dim, 512)
        self.language_proj = nn.Linear(language_dim, 512)
        
        # Multi-head attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(512, vision_dim)
    
    def forward(self, vision_feat: torch.Tensor, 
                language_feat: torch.Tensor) -> torch.Tensor:
        
        # Project features
        v_proj = self.vision_proj(vision_feat)
        l_proj = self.language_proj(language_feat)
        
        # Cross attention
        v_proj = v_proj.transpose(0, 1)
        l_proj = l_proj.transpose(0, 1)
        
        fused, _ = self.cross_attention(v_proj, l_proj, l_proj)
        fused = fused.transpose(0, 1)
        
        # Output projection
        output = self.output_proj(fused)
        
        return output


class ABINet(nn.Module):
    """ABINet: Autonomous, Bidirectional and Iterative Language Modeling."""
    
    def __init__(self, 
                 num_classes: int,
                 max_len: int = 50,
                 vision_dim: int = 512,
                 language_dim: int = 256,
                 num_iterations: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_len = max_len
        self.num_iterations = num_iterations
        
        # Vision model
        self.vision_encoder = VisionEncoder(in_channels=1, hidden_dim=vision_dim, seq_out=max_len)
        
        # Language model
        self.language_model = LanguageModel(
            num_classes=num_classes,
            hidden_dim=language_dim,
            max_len=max_len,
            memory_input_dim=vision_dim,
        )
        
        # Fusion module
        self.fusion = FusionModule(vision_dim, language_dim)
        
        # Output heads
        self.vision_head = nn.Linear(vision_dim, num_classes)
        self.language_head = nn.Linear(language_dim, num_classes)
        self.fusion_head = nn.Linear(vision_dim, num_classes)
    
    def forward(self, images: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                decoder_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Vision branch
        vision_feat = self.vision_encoder(images)
        vision_logits = self.vision_head(vision_feat)
        
        outputs = {'vision_logits': vision_logits}
        
        # Language modeling (iterative refinement)
        if self.training and targets is not None:
            # Training mode with teacher forcing
            lm_input = decoder_input if decoder_input is not None else targets
            language_feat = self.language_model(vision_feat, lm_input)
            language_logits = self.language_head(language_feat)
            outputs['language_logits'] = language_logits

            # Fusion
            fused_feat = self.fusion(vision_feat, language_feat)
            fusion_logits = self.fusion_head(fused_feat)
            outputs['fusion_logits'] = fusion_logits

            # Training: use language logits as principal output
            T = language_logits.size(1)
            outputs['logits'] = language_logits[:, :T]
            
        else:
            # Inference mode - greedy decode with language model only
            B = images.size(0)
            device = images.device
            # Start with SOS token index=1 (consistent with dataset encoding)
            tokens = torch.ones(B, 1, dtype=torch.long, device=device)
            logits_steps: list[torch.Tensor] = []
            for _ in range(self.max_len - 1):
                language_feat = self.language_model(vision_feat, tokens)
                language_logits = self.language_head(language_feat)  # (B, L, C)
                step_logits = language_logits[:, -1:, :]            # last step
                logits_steps.append(step_logits)
                next_token = torch.argmax(step_logits, dim=-1)      # (B,1)
                tokens = torch.cat([tokens, next_token], dim=1)
                # stop if all predicted EOS (index=2)
                if torch.all(next_token.squeeze(1) == 2):
                    break
            if logits_steps:
                outputs['logits'] = torch.cat(logits_steps, dim=1)
            else:
                outputs['logits'] = torch.zeros(B, 1, self.num_classes, device=device)

        return outputs

    @torch.no_grad()
    def greedy_tokens(self, images: torch.Tensor, max_len: int) -> torch.Tensor:
        """Greedy-generate token ids for scheduled sampling input.

        Returns tokens including SOS, length=max_len.
        """
        self.eval()
        with torch.no_grad():
            memory = self.vision_encoder(images)
            B = images.size(0)
            device = images.device
            tokens = torch.ones(B, 1, dtype=torch.long, device=device)
            for _ in range(max(1, max_len - 1)):
                language_feat = self.language_model(memory, tokens)
                language_logits = self.language_head(language_feat)
                step_logits = language_logits[:, -1:, :]
                next_token = torch.argmax(step_logits, dim=-1)
                tokens = torch.cat([tokens, next_token], dim=1)
        self.train()
        return tokens[:, :max_len]

    @torch.no_grad()
    def beam_tokens(self, images: torch.Tensor, beam_size: int = 3, length_penalty: float = 0.0):
        """Beam search decoding for a single image (batch=1).

        Returns
        -------
        tokens: torch.Tensor (1, L)
        score: float (average log-prob per token)
        """
        assert images.size(0) == 1, "beam_tokens currently supports batch=1"
        device = images.device
        self.eval()
        memory = self.vision_encoder(images)  # (1, T, D)
        # Each beam: (tokens_tensor [1,L], cum_logprob)
        beams = [(torch.ones(1, 1, dtype=torch.long, device=device), 0.0)]  # start with <sos>=1
        max_len = self.max_len
        for _ in range(max_len - 1):
            new_beams = []
            for tok, logp in beams:
                # If already ended with <eos>=2, keep as is
                if tok[0, -1].item() == 2:
                    new_beams.append((tok, logp))
                    continue
                language_feat = self.language_model(memory, tok)
                language_logits = self.language_head(language_feat)
                step_logits = language_logits[:, -1, :]  # (1, C)
                probs = torch.nn.functional.log_softmax(step_logits, dim=-1)  # (1, C)
                # Take top-k candidates
                topk = torch.topk(probs, k=min(beam_size, probs.size(1)), dim=-1)
                for i in range(topk.indices.size(1)):
                    nid = topk.indices[0, i].item()
                    nlogp = topk.values[0, i].item()
                    ntok = torch.cat([tok, torch.tensor([[nid]], dtype=torch.long, device=device)], dim=1)
                    new_beams.append((ntok, logp + nlogp))
            # Keep best beam_size beams (apply optional length penalty)
            def score_fn(b):
                t, lp = b
                L = t.size(1)
                denom = (L ** (1.0 + length_penalty)) if length_penalty > 0 else max(L, 1)
                return lp / denom
            new_beams.sort(key=score_fn, reverse=True)
            beams = new_beams[:beam_size]
            # Early stop if all beams ended
            if all(t[0, -1].item() == 2 for t, _ in beams):
                break
        best_tok, best_lp = beams[0]
        # average log-prob per token (ignore sos)
        L = max(best_tok.size(1) - 1, 1)
        avg_lp = best_lp / L
        self.train()
        return best_tok[:, :max_len], float(avg_lp)


class ABINetLoss(nn.Module):
    """Loss function for ABINet training with optional label smoothing."""
    
    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.0,
                 w_vision: float = 0.5, w_language: float = 0.5, w_fusion: float = 1.0,
                 ctc_weight: float = 0.0, blank_index: int = 0):
        super().__init__()
        try:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        except TypeError:
            # for older torch versions without label_smoothing
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.w_vision = float(w_vision)
        self.w_language = float(w_language)
        self.w_fusion = float(w_fusion)
        self.ctc_weight = float(ctc_weight)
        self.blank_index = int(blank_index)
        if self.ctc_weight > 0.0:
            self.ctc_loss_fn = nn.CTCLoss(blank=self.blank_index, reduction='mean', zero_infinity=True)
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        # Teacher forcing: decoder input is typically <sos> + tokens[:-1], and
        # the target to predict is tokens[1:] (including <eos>). Align losses accordingly.
        # If targets length is L, we can compute up to L-1 steps of next-token prediction.

        def _loss_for(logits: torch.Tensor, name: str) -> torch.Tensor:
            B, T, C = logits.shape
            # Use targets shifted by 1: predict next token (incl. <eos>)
            if targets.size(1) <= 1:
                # not enough length to compute meaningful loss
                return torch.tensor(0.0, device=logits.device)
            T_eff = min(T, targets.size(1) - 1)
            log = logits[:, :T_eff]
            tgt = targets[:, 1:1 + T_eff]
            return self.criterion(log.reshape(B * T_eff, C), tgt.reshape(B * T_eff))

        # Vision loss
        if 'vision_logits' in outputs:
            losses['vision_loss'] = _loss_for(outputs['vision_logits'], 'vision') * self.w_vision

        # Language loss
        if 'language_logits' in outputs:
            losses['language_loss'] = _loss_for(outputs['language_logits'], 'language') * self.w_language

        # Fusion loss
        if 'fusion_logits' in outputs:
            losses['fusion_loss'] = _loss_for(outputs['fusion_logits'], 'fusion') * self.w_fusion

        # Total loss
        if losses:
            total = None
            for v in losses.values():
                total = v if total is None else total + v
            losses['total'] = total
        else:
            losses['total'] = torch.tensor(0.0, device=targets.device)

        # Optional CTC loss on vision branch logits (alignment-free auxiliary)
        if self.ctc_weight > 0.0 and 'vision_logits' in outputs:
            v_log = outputs['vision_logits']  # (B, T, C)
            B, T, C = v_log.shape
            # log_probs: (T, B, C)
            log_probs = F.log_softmax(v_log, dim=-1).transpose(0, 1)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=targets.device)
            # Build target sequences per sample: strip <sos>(1), pad(0), stop at <eos>(2)
            target_list = []
            target_lengths = []
            for b in range(B):
                seq = targets[b].detach().tolist()
                out = []
                for idx in seq:
                    if idx == 1:  # <sos>
                        continue
                    if idx == 2:  # <eos>
                        break
                    if idx == 0:  # <pad>
                        break
                    out.append(int(idx))
                target_list.extend(out)
                target_lengths.append(len(out))
            if sum(target_lengths) == 0:
                ctc = torch.tensor(0.0, device=targets.device)
            else:
                flat_targets = torch.tensor(target_list, dtype=torch.long, device=targets.device)
                target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=targets.device)
                ctc = self.ctc_loss_fn(log_probs, flat_targets, input_lengths, target_lengths_t)
            losses['ctc_loss'] = ctc * self.ctc_weight
            losses['total'] = losses['total'] + losses['ctc_loss']

        return losses
