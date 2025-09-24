import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import os
import cv2
from pathlib import Path
import logging
import json
import unicodedata

from ..modeling.rec_abinet import ABINet
from ..modeling.rec_satrn import SATRN


logger = logging.getLogger(__name__)


class TextRecognizer:
    """Unified text recognition interface for ABINet and SATRN models."""
    
    def __init__(self,
                 model_type: str = 'abinet',
                 weights_path: Optional[Path] = None,
                 device: str = 'cuda',
                 vocab_path: Optional[Path] = None,
                 keep_unknown: Optional[bool] = None,
                 unk_placeholder: Optional[str] = None,
                 max_len: int = 50,
                 beam_size: Optional[int] = None):
        
        self.model_type = model_type
        self.device = device
        
        # Load vocabulary
        self.char_dict = self._load_vocabulary(vocab_path)
        self.num_classes = len(self.char_dict)
        
        # Initialize model
        if model_type == 'abinet':
            self.model = ABINet(
                num_classes=self.num_classes,
                max_len=int(max_len),
                vision_dim=512,
                language_dim=256
            ).to(device)
        elif model_type == 'satrn':
            self.model = SATRN(
                num_classes=self.num_classes,
                max_len=int(max_len),
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if provided
        if weights_path and weights_path.exists():
            try:
                logger.info(f"Loading weights from {weights_path}")
                checkpoint = torch.load(weights_path, map_location=device)
                state = None
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state = checkpoint['state_dict']
                elif isinstance(checkpoint, dict):
                    state = checkpoint
                if state is not None:
                    # Sanitize mismatched shapes (e.g., positional enc buffers when max_len changed)
                    model_sd = self.model.state_dict()
                    clean = {}
                    dropped = []
                    for k, v in state.items():
                        if k not in model_sd:
                            continue
                        if isinstance(v, torch.Tensor) and isinstance(model_sd[k], torch.Tensor):
                            if tuple(v.shape) != tuple(model_sd[k].shape):
                                dropped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                                continue
                        clean[k] = v
                    missing, unexpected = self.model.load_state_dict(clean, strict=False)
                    if dropped:
                        logger.warning(f"Dropped {len(dropped)} mismatched tensors (shape changes): {[d[0] for d in dropped[:5]]}...")
                    if missing or unexpected:
                        logger.warning(f"Loaded with missing={len(missing)} unexpected={len(unexpected)} keys")
                else:
                    logger.warning("Checkpoint format not recognized; skipping weights load")
            except Exception as e:
                logger.warning(f"Failed to load weights ({weights_path}): {e}. Using random initialization")
        else:
            logger.warning("No weights loaded, using random initialization")
        
        self.model.eval()
        
        # Special tokens
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        # Unknown handling controls (env overrides)
        if keep_unknown is None:
            keep_unknown = str(os.getenv('DOCJA_KEEP_UNKNOWN', '0')).lower() in ('1', 'true', 'yes')
        if unk_placeholder is None:
            unk_placeholder = os.getenv('DOCJA_UNK_PLACEHOLDER', '□')
        self.keep_unknown = bool(keep_unknown)
        self.unk_placeholder = unk_placeholder
        # Beam search size
        if beam_size is None:
            try:
                beam_size = int(os.getenv('DOCJA_REC_BEAM', '1'))
            except Exception:
                beam_size = 1
        self.beam_size = max(1, int(beam_size))
    
    def recognize(self, image: np.ndarray, 
                  vertical: bool = False) -> Tuple[str, float]:
        """Recognize text in image.
        
        Args:
            image: Cropped text region (H, W, 3) in RGB format
            vertical: Whether text is vertical
            
        Returns:
            text: Recognized text string
            confidence: Recognition confidence score
        """
        
        # Handle vertical text
        if vertical:
            image = np.rot90(image, k=-1)
        
        # Preprocess
        img_tensor = self._preprocess(image)
        
        with torch.no_grad():
            if self.model_type == 'abinet' and self.beam_size > 1:
                tokens, avg_lp = self.model.beam_tokens(img_tensor, beam_size=self.beam_size, length_penalty=0.0)
                pred_ids = tokens[0]
                text = self._decode(pred_ids)
                # convert average log-prob to pseudo-confidence (0..1)
                confidence = float(torch.exp(torch.tensor(avg_lp)).item())
            else:
                if self.model_type == 'abinet':
                    output = self.model(img_tensor)
                else:
                    output = self.model(img_tensor)
                logits = output['logits']
                confidences = F.softmax(logits, dim=-1)
                pred_ids = torch.argmax(logits, dim=-1)
                text = self._decode(pred_ids[0])
                confidence = self._calculate_confidence(confidences[0], pred_ids[0])
        
        # Post-process
        text = self._postprocess_text(text)
        
        return text, confidence
    
    def _load_vocabulary(self, vocab_path: Optional[Path]) -> dict:
        """Load character vocabulary.

        Priority:
          1) JSON file provided via vocab_path {char: idx}
          2) data/charset_ja.txt (one character per line)
          3) Built-in minimal default
        """
        # 1) explicit vocabulary file provided
        if vocab_path and Path(vocab_path).exists():
            # Try JSON first; if it fails, treat as one-character-per-line text
            try:
                with open(vocab_path, encoding='utf-8') as f:
                    obj = json.load(f)
                # obj can be {char: idx} or {idx: char} or [char, ...]
                if isinstance(obj, dict):
                    # Normalize mapping to {char: idx}
                    # detect if keys are chars or indices
                    if all(isinstance(k, str) and not k.isdigit() for k in obj.keys()):
                        char_dict = {str(k): int(v) for k, v in obj.items()}
                    else:
                        # idx->char mapping
                        tmp = {int(k): str(v) for k, v in obj.items()}
                        # build list ordered by index
                        max_idx = max(tmp.keys()) if tmp else -1
                        chars = ['<pad>', '<sos>', '<eos>', '<unk>']
                        # ensure capacity
                        arr = [''] * (max_idx + 1)
                        for i, ch in tmp.items():
                            if i < len(arr):
                                arr[i] = ch
                            else:
                                arr.extend([''] * (i - len(arr) + 1))
                                arr[i] = ch
                        # rebuild mapping from list (skip empties)
                        char_dict = {}
                        # assume arr already contains specials if provided; otherwise we will re-add
                        # rebuild with enumeration if needed
                        for idx, ch in enumerate(arr):
                            if ch:
                                char_dict[ch] = idx
                    self.idx_to_char = {idx: ch for ch, idx in char_dict.items()}
                    # Ensure special tokens exist
                    for i, sp in enumerate(['<pad>', '<sos>', '<eos>', '<unk>']):
                        if sp not in char_dict:
                            new_idx = len(char_dict)
                            char_dict[sp] = new_idx
                            self.idx_to_char[new_idx] = sp
                    return char_dict
                elif isinstance(obj, list):
                    chars = ['<pad>', '<sos>', '<eos>', '<unk>'] + [str(x) for x in obj if str(x).strip()]
                    char_dict = {ch: i for i, ch in enumerate(chars)}
                    self.idx_to_char = {i: ch for ch, i in char_dict.items()}
                    return char_dict
            except Exception:
                # Fallback to text (one char per line)
                try:
                    lines = Path(vocab_path).read_text(encoding='utf-8').splitlines()
                    chars = ['<pad>', '<sos>', '<eos>', '<unk>']
                    for line in lines:
                        line = line.rstrip('\n')
                        if not line or line.startswith('#'):
                            continue
                        chars.append(line)
                    char_dict = {ch: i for i, ch in enumerate(chars)}
                    self.idx_to_char = {i: ch for ch, i in char_dict.items()}
                    return char_dict
                except Exception as e:
                    logger.warning(f"Failed to parse vocabulary at {vocab_path}: {e}")
                    # fallthrough to default

        # 2) external charset file
        default_txt = Path('data/charset_ja.txt')
        if default_txt.exists():
            chars = ['<pad>', '<sos>', '<eos>', '<unk>']
            for line in default_txt.read_text(encoding='utf-8').splitlines():
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                chars.append(line)
            char_dict = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for ch, i in char_dict.items()}
            return char_dict

        # 3) fallback minimal default
        vocab = self._create_default_vocabulary()
        return vocab
    
    def _create_default_vocabulary(self) -> dict:
        """Create default Japanese vocabulary."""
        chars = ['<pad>', '<sos>', '<eos>', '<unk>']
        
        # ASCII
        for i in range(32, 127):
            chars.append(chr(i))
        
        # Hiragana (U+3040 - U+309F)
        for i in range(0x3040, 0x30A0):
            chars.append(chr(i))
        
        # Katakana (U+30A0 - U+30FF)
        for i in range(0x30A0, 0x3100):
            chars.append(chr(i))
        
        # Common Kanji (first 2000)
        common_kanji = [
            '日', '本', '人', '大', '年', '出', '中', '国', '生', '子',
            '分', '時', '上', '下', '月', '行', '見', '金', '長', '間',
            # ... add more common kanji
        ]
        chars.extend(common_kanji[:2000])
        
        # Numbers and symbols
        chars.extend(['０', '１', '２', '３', '４', '５', '６', '７', '８', '９'])
        chars.extend(['、', '。', '「', '」', '・', 'ー', '（', '）', '％', '円'])
        
        # Create char to index mapping
        char_dict = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in char_dict.items()}
        
        return char_dict
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for recognition.

        推論時も学習と同等の前処理を実施：
        - 自動クロップ（Otsu→外接矩形）
        - 縦長の自動回転
        - CLAHEによるコントラスト改善
        - レターボックス（48×320）
        """
        # Allow override by environment variables
        try:
            target_h = int(os.getenv('DOCJA_REC_TARGET_H', '48'))
        except Exception:
            target_h = 48
        try:
            target_w = int(os.getenv('DOCJA_REC_TARGET_W', '320'))
        except Exception:
            target_w = 320

        # 自動クロップ
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(bin_inv > 0)
        img_rgb = image
        if len(xs) > 0 and len(ys) > 0 and str(os.getenv('DOCJA_REC_AUTO_CROP', '1')) in ('1', 'true', 'yes'):
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            mh = int(0.05 * (y2 - y1 + 1))
            mw = int(0.05 * (x2 - x1 + 1))
            y1 = max(0, y1 - mh)
            y2 = min(image.shape[0] - 1, y2 + mh)
            x1 = max(0, x1 - mw)
            x2 = min(image.shape[1] - 1, x2 + mw)
            img_rgb = image[y1:y2 + 1, x1:x2 + 1]

        # 縦長なら回転
        h, w = img_rgb.shape[:2]
        if h > w * 1.5 and str(os.getenv('DOCJA_REC_AUTO_ROTATE', '1')) in ('1', 'true', 'yes'):
            img_rgb = np.rot90(img_rgb, k=-1)
            h, w = img_rgb.shape[:2]

        # CLAHE
        gray2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray2 = clahe.apply(gray2)

        # リサイズ + レターボックス
        ratio = min(target_w / w, target_h / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        resized = cv2.resize(gray2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255,))

        img_tensor = torch.from_numpy(padded).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        return img_tensor.to(self.device)
    
    def _decode(self, pred_ids: torch.Tensor) -> str:
        """Decode prediction IDs to text."""
        chars = []
        for idx in pred_ids:
            idx = idx.item()
            
            # Stop at EOS
            if idx == self.eos_idx:
                break
            
            # Skip PAD/SOS
            if idx in [self.pad_idx, self.sos_idx]:
                continue
            # UNK handling
            if idx == self.unk_idx:
                if self.keep_unknown:
                    chars.append(self.unk_placeholder)
                continue
            
            # Add character
            if idx < len(self.idx_to_char):
                chars.append(self.idx_to_char[idx])
        
        return ''.join(chars)
    
    def _calculate_confidence(self, confidences: torch.Tensor, 
                            pred_ids: torch.Tensor) -> float:
        """Calculate overall confidence score."""
        conf_scores = []
        
        for i, idx in enumerate(pred_ids):
            idx = idx.item()
            
            # Stop at EOS
            if idx == self.eos_idx:
                break
            
            # Skip PAD/SOS
            if idx in [self.pad_idx, self.sos_idx]:
                continue
            if idx == self.unk_idx:
                # count UNK if we keep placeholder, otherwise skip
                if not self.keep_unknown:
                    continue
            
            # Get confidence for this prediction
            conf = confidences[i, idx].item()
            conf_scores.append(conf)
        
        if not conf_scores:
            return 0.0
        
        # Return average confidence
        return sum(conf_scores) / len(conf_scores)

    def _postprocess_text(self, text: str) -> str:
        """Normalize recognized text.

        - Unicode NFKC正規化（全角半角の統一、互換文字の正規化）
        - 連続空白の単一化
        - 前後空白の除去
        """
        if not text:
            return text
        # Unicode normalization (NFKC)
        norm = unicodedata.normalize('NFKC', text)
        # Collapse whitespace
        norm = ' '.join(norm.split())
        return norm.strip()
    
    def _postprocess_text(self, text: str) -> str:
        """Post-process recognized text.

        - Unicode正規化(NFKC)
        - 連続空白の単一化
        - よくあるOCR誤りの置換
        - 句読点などの正規化
        - 連続重複文字の抑制
        """
        if not text:
            return text
        # Unicode normalize
        text = unicodedata.normalize('NFKC', text)
        # collapse whitespace
        text = ' '.join(text.split())
        # common errors
        text = self._fix_common_errors(text)
        # punctuation normalize
        text = self._normalize_punctuation(text)
        # remove duplicates
        text = self._remove_duplicates(text)
        return text.strip()
    
    def _remove_duplicates(self, text: str) -> str:
        """Remove consecutive duplicate characters."""
        if not text:
            return text
        
        result = [text[0]]
        for char in text[1:]:
            if char != result[-1]:
                result.append(char)
        
        return ''.join(result)
    
    def _fix_common_errors(self, text: str) -> str:
        """Fix common OCR errors."""
        replacements = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            'ｌ': '1', 'О': '0', 'о': '0',  # Common confusions
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        replacements = {
            '､': '、',
            '｡': '。',
            '｢': '「',
            '｣': '」',
            '･': '・',
            'ｰ': 'ー',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def recognize_batch(self, images: List[np.ndarray], 
                       vertical: bool = False) -> List[Tuple[str, float]]:
        """Batch recognition for efficiency."""
        results = []
        
        # Prepare batch
        batch_tensors = []
        for img in images:
            if vertical:
                img = np.rot90(img, k=-1)
            tensor = self._preprocess(img)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.cat(batch_tensors, dim=0)
        
        with torch.no_grad():
            if self.model_type == 'abinet':
                output = self.model(batch)
                logits = output['logits']
            else:  # SATRN
                output = self.model(batch)
                logits = output['logits']
            
            confidences = F.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode each result
        for i in range(len(images)):
            text = self._decode(pred_ids[i])
            conf = self._calculate_confidence(confidences[i], pred_ids[i])
            text = self._postprocess_text(text)
            results.append((text, conf))
        
        return results
