#!/usr/bin/env python3
import argparse
import json
import yaml
from pathlib import Path
import os
from typing import Dict, Any, List, Tuple
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from .pipeline.text_recognizer import TextRecognizer
from .modeling.rec_abinet import ABINetLoss
from .utils.metrics import cer as cer_metric


class LineImageDataset(Dataset):
    """行画像 + ラベルのデータセット（前処理強化版）

    - 自動クロップ（Otsu二値化→外接矩形）
    - 縦長判定で自動回転（90度）
    - CLAHEでコントラスト改善
    - 低前景率（ほぼ真っ白）をスキップ
    - デバッグ出力（環境変数 DOCJA_REC_DEBUG_DIR）
    """

    def __init__(
        self,
        images_dir: Path,
        labels_path: Path,
        char_dict: Dict[str, int],
        max_len: int = 50,
        target_h: int = 48,
        target_w: int = 320,
        min_fg_ratio: float = 0.003,
        auto_rotate: bool = True,
        auto_crop: bool = True,
        alt_images_dir: Path = None,
    ):
        self.images_dir = Path(images_dir)
        self.alt_images_dir = Path(alt_images_dir) if alt_images_dir else None
        self.labels = json.loads(Path(labels_path).read_text(encoding='utf-8'))
        self.samples = [(k, v) for k, v in self.labels.items()]
        self.char_dict = char_dict
        self.idx_pad = 0
        self.idx_sos = 1
        self.idx_eos = 2
        self.idx_unk = 3
        self.max_len = max_len
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.min_fg_ratio = float(min_fg_ratio)
        self.auto_rotate = bool(auto_rotate)
        self.auto_crop = bool(auto_crop)

        # Augmentation controls (shape-preserving, light-weight)
        self.augment = bool(os.getenv('DOCJA_REC_AUG_ENABLE', str(
            (yaml.safe_load(os.getenv('DOCJA_REC_AUG_CFG', 'false')) if False else False)
        )).lower() in ('1', 'true'))  # default off unless enabled via config below
        # Values can be overridden by config passed via environment JSON or attributes set by caller
        self.aug_rotate_deg = 0
        self.aug_blur_prob = 0.0
        self.aug_noise_prob = 0.0
        self.aug_noise_std = 3.0
        self.aug_erase_prob = 0.0

        # Optional debug dir
        self.debug_dir = os.getenv('DOCJA_REC_DEBUG_DIR')
        if self.debug_dir:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)

        # Reuse CLAHE instance to avoid per-sample init cost
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name, text = self.samples[idx]
        img_path = self.images_dir / name
        if not img_path.exists() and self.alt_images_dir is not None:
            alt_path = self.alt_images_dir / name
            if alt_path.exists():
                img_path = alt_path
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        img_t, ok = self._preprocess(img_np, debug_name=name)
        # フォールバック: 低前景サンプルは最大5回まで他のサンプルに置換
        tries = 0
        while not ok and tries < 5:
            tries += 1
            alt_idx = (idx + tries) % len(self.samples)
            alt_name, alt_text = self.samples[alt_idx]
            img = Image.open(self.images_dir / alt_name).convert('RGB')
            img_np = np.array(img)
            img_t, ok = self._preprocess(img_np, debug_name=alt_name)
            if ok:
                name, text = alt_name, alt_text
                break
        tgt = self._encode(text)
        return img_t, tgt

    def _preprocess(self, image: np.ndarray, debug_name: str = "") -> Tuple[torch.Tensor, bool]:
        img_rgb = image.copy()

        # 1) グレースケール + Otsuでテキストマスク
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2) 前景率（白地に黒文字想定の反転マスクの非ゼロ割合）
        fg_ratio = float(np.count_nonzero(bin_inv)) / float(bin_inv.size)

        # 3) クロップ（前景が少しでもある場合）
        if self.auto_crop and fg_ratio > 0:
            ys, xs = np.where(bin_inv > 0)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                # マージン（5%）
                mh = int(0.05 * (y2 - y1 + 1))
                mw = int(0.05 * (x2 - x1 + 1))
                y1 = max(0, y1 - mh)
                y2 = min(img_rgb.shape[0] - 1, y2 + mh)
                x1 = max(0, x1 - mw)
                x2 = min(img_rgb.shape[1] - 1, x2 + mw)
                img_rgb = img_rgb[y1:y2 + 1, x1:x2 + 1]

        # 4) 自動回転（縦長→横長へ）
        h, w = img_rgb.shape[:2]
        if self.auto_rotate and h > w * 1.5:
            img_rgb = np.rot90(img_rgb, k=-1)
            h, w = img_rgb.shape[:2]

        # 5) コントラスト改善（CLAHE）
        gray2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray2 = self._clahe.apply(gray2)

        # 5.5) 軽量Augmentation（学習時のみ）
        if self.augment:
            # small rotation
            if self.aug_rotate_deg and self.aug_rotate_deg > 0:
                ang = (np.random.rand() * 2 - 1) * float(self.aug_rotate_deg)
                h2, w2 = gray2.shape[:2]
                M = cv2.getRotationMatrix2D((w2 * 0.5, h2 * 0.5), ang, 1.0)
                gray2 = cv2.warpAffine(gray2, M, (w2, h2), flags=cv2.INTER_LINEAR, borderValue=255)
            # random blur
            if self.aug_blur_prob and np.random.rand() < float(self.aug_blur_prob):
                k = 3 if np.random.rand() < 0.7 else 5
                gray2 = cv2.GaussianBlur(gray2, (k, k), 0)
            # random gaussian noise
            if self.aug_noise_prob and np.random.rand() < float(self.aug_noise_prob):
                std = float(self.aug_noise_std) if self.aug_noise_std else 3.0
                noise = np.random.normal(0, std, gray2.shape).astype(np.float32)
                gray2 = np.clip(gray2.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6) リサイズ + レターボックス
        ratio = min(self.target_w / w, self.target_h / h)
        new_w = max(1, int(w * ratio))
        new_h = max(1, int(h * ratio))
        resized = cv2.resize(gray2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255,)
        )

        # Random erasing (cutout) on the final canvas
        if self.augment and self.aug_erase_prob and np.random.rand() < float(self.aug_erase_prob):
            H, W = padded.shape[:2]
            # Erase a small block up to 10% area
            eh = max(1, int(H * (0.05 + 0.05 * np.random.rand())))
            ew = max(1, int(W * (0.05 + 0.05 * np.random.rand())))
            y0 = np.random.randint(0, max(1, H - eh + 1))
            x0 = np.random.randint(0, max(1, W - ew + 1))
            padded[y0:y0 + eh, x0:x0 + ew] = 255

        # 7) 正規化
        img_tensor = torch.from_numpy(padded).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)

        # 8) デバッグ保存（任意）
        ok = True
        if fg_ratio < self.min_fg_ratio:
            ok = False
        if self.debug_dir:
            try:
                dbg_path = Path(self.debug_dir) / f"{debug_name.replace('.png','')}_{'ok' if ok else 'skip'}.png"
                cv2.imwrite(str(dbg_path), padded)
            except Exception:
                pass

        return img_tensor, ok

    def _encode(self, text: str) -> torch.Tensor:
        # Build sequence with special tokens: <sos> text... <eos>
        ids: List[int] = [1]  # <sos>
        for ch in text:
            ids.append(self.char_dict.get(ch, 3))  # <unk>=3
            if len(ids) >= self.max_len - 1:
                break
        ids.append(2)  # <eos>
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))  # pad with <pad>=0
        else:
            ids = ids[: self.max_len]
            ids[-1] = 2  # ensure last token is <eos>
        return torch.tensor(ids, dtype=torch.long)


def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def _eval_cer(
    model: torch.nn.Module,
    recog: TextRecognizer,
    images_dir: Path,
    labels_path: Path,
    target_h: int,
    target_w: int,
    sample_size: int = 2000,
    batch_size: int = 128,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Evaluate CER on a subset of preprocessed line images.

    Assumes images are already cropped/letterboxed; performs light normalization only.
    """
    if not images_dir.exists() or not labels_path.exists():
        return {'cer': 0.0, 'num': 0}

    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    # build list of existing files
    items = [(images_dir / k, v) for k, v in labels.items() if (images_dir / k).exists()]
    if not items:
        return {'cer': 0.0, 'num': 0}
    if sample_size and len(items) > sample_size:
        random.seed(42)
        items = random.sample(items, sample_size)

    model_was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        # process in mini-batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            tensors = []
            for p, _ in batch:
                try:
                    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    # Ensure size
                    h, w = img.shape[:2]
                    if h != target_h or w != target_w:
                        ratio = min(target_w / w, target_h / h)
                        new_w = max(1, int(w * ratio))
                        new_h = max(1, int(h * ratio))
                        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        pad_w = target_w - new_w
                        pad_h = target_h - new_h
                        img = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255,))
                    t = torch.from_numpy(img).float() / 255.0
                    t = (t - 0.5) / 0.5
                    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                    tensors.append(t)
                except Exception:
                    continue
            if not tensors:
                continue
            batch_t = torch.cat(tensors, dim=0).to(device)
            out = model(batch_t)
            logits = out['logits']  # (B, T, C)
            pred_ids = torch.argmax(logits, dim=-1)
            # decode and compare
            for j, (_, gt) in enumerate(batch):
                if j >= pred_ids.size(0):
                    break
                text = recog._decode(pred_ids[j])
                total += float(cer_metric(gt, text))
                n += 1
    if model_was_training:
        model.train()
    return {'cer': (total / n) if n > 0 else 0.0, 'num': n}


class ExponentialMovingAverage:
    """Simple EMA for model parameters to improve generalization/accuracy.

    Maintains shadow weights: w_ema = decay * w_ema + (1 - decay) * w
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        # Initialize shadow with current params
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def copy_to(self, model: torch.nn.Module):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.shadow:
                    p.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().cpu() for k, v in self.shadow.items()}


def main():
    ap = argparse.ArgumentParser(description='Train text recognition (ABINet/SATRN)')
    ap.add_argument('--config', type=str, required=False, default='configs/rec_abinet.yaml')
    ap.add_argument('--model', type=str, default='abinet', choices=['abinet', 'satrn'])
    ap.add_argument('--resume', type=str, default='')
    ap.add_argument('--max-steps', type=int, default=0, help='Limit steps per epoch for smoke run (0=disabled)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Enable cuDNN autotune for performance
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    # Training configuration (moved up so losses can read it)
    train_cfg = cfg.get('training', {})

    # Create recognizer to obtain vocabulary and model
    # Use charset from config if provided
    tmp_cfg_data = cfg.get('data', {})
    vocab_path = Path(tmp_cfg_data.get('charset_path', 'data/charset_ja.txt'))
    model_max_len = int(cfg.get('model', {}).get('max_len', 50))
    recog = TextRecognizer(model_type=args.model, device=device, vocab_path=vocab_path, max_len=model_max_len)
    model = recog.model
    model.train()

    # Dataset setup (expects images_dir + labels in config)
    data_cfg = cfg.get('data', {})
    images_dir = Path(data_cfg.get('images_dir', 'data/rec/lines'))
    labels_path = Path(data_cfg.get('labels', 'data/rec/labels.json'))
    batch_size = int(data_cfg.get('batch_size', 32))
    num_workers = int(data_cfg.get('num_workers', 0))
    prefetch_factor_cfg = data_cfg.get('prefetch_factor', 2)
    try:
        prefetch_factor_cfg = int(prefetch_factor_cfg) if prefetch_factor_cfg is not None else None
    except Exception:
        prefetch_factor_cfg = 2
    # DataLoader robustness options
    persistent_workers_cfg = bool(data_cfg.get('persistent_workers', False))
    timeout_cfg = int(data_cfg.get('timeout', 0))
    max_len = int(cfg.get('model', {}).get('max_len', 50))
    target_w = int(data_cfg.get('target_width', 320))
    target_h = int(data_cfg.get('target_height', 48))
    min_fg_ratio = float(data_cfg.get('min_fg_ratio', 0.003))
    auto_rotate = bool(data_cfg.get('auto_rotate', True))
    auto_crop = bool(data_cfg.get('auto_crop', True))
    alt_images_dir = data_cfg.get('alt_images_dir', None)

    ds = LineImageDataset(
        images_dir,
        labels_path,
        recog.char_dict,
        max_len=max_len,
        target_h=target_h,
        target_w=target_w,
        min_fg_ratio=min_fg_ratio,
        auto_rotate=auto_rotate,
        auto_crop=auto_crop,
        alt_images_dir=alt_images_dir,
    )
    # Hook augmentations from config.data.augment_*
    aug_cfg = data_cfg
    ds.augment = bool(aug_cfg.get('augment', False))
    ds.aug_rotate_deg = int(aug_cfg.get('aug_rotate_deg', 5))
    ds.aug_blur_prob = float(aug_cfg.get('aug_blur_prob', 0.1))
    ds.aug_noise_prob = float(aug_cfg.get('aug_noise_prob', 0.2))
    ds.aug_noise_std = float(aug_cfg.get('aug_noise_std', 3.0))
    ds.aug_erase_prob = float(aug_cfg.get('aug_erase_prob', 0.05))
    def _worker_init_fn(_):
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

    # Reduce OpenCV threads in main process as well
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(persistent_workers_cfg and num_workers > 0),
        prefetch_factor=(prefetch_factor_cfg if num_workers > 0 else None),
        timeout=(timeout_cfg if timeout_cfg > 0 else 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    # Loss and optimizer
    if args.model == 'abinet':
        ls = float(train_cfg.get('label_smoothing', 0.0))
        ctc_w = float(train_cfg.get('ctc', {}).get('weight', 0.0))
        # Loss weights can be provided via cfg.loss.{w_vision,w_language,w_fusion}
        loss_cfg = cfg.get('loss', {})
        w_vision = float(loss_cfg.get('w_vision', 0.0))
        w_language = float(loss_cfg.get('w_language', 1.0))
        w_fusion = float(loss_cfg.get('w_fusion', 0.0))
        criterion = ABINetLoss(ignore_index=0, label_smoothing=ls,
                               w_vision=w_vision, w_language=w_language, w_fusion=w_fusion,
                               ctc_weight=ctc_w, blank_index=0)
    else:
        # generic CE loss for logits
        try:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=float(train_cfg.get('label_smoothing', 0.0)))
        except TypeError:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    epochs = int(train_cfg.get('epochs', 5))
    lr = float(train_cfg.get('optimizer', {}).get('lr', 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(train_cfg.get('use_amp', True))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == 'cuda'))
    # Optional scheduler (MultiStepLR / OneCycleLR)
    scheduler = None
    sched_cfg = train_cfg.get('scheduler', {})
    if sched_cfg.get('type', '').lower() in ('multisteplr', 'multistep'):
        import torch.optim as optim
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = sched_cfg.get('milestones', [10, 15])
        gamma = float(sched_cfg.get('gamma', 0.1))
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif sched_cfg.get('type', '').lower() in ('onecyclelr', 'onecycle'):
        from torch.optim.lr_scheduler import OneCycleLR
        max_lr = float(sched_cfg.get('max_lr', lr))
        pct_start = float(sched_cfg.get('pct_start', 0.1))
        steps_per_epoch = max(1, len(dl))
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=int(train_cfg.get('epochs', 5)), steps_per_epoch=steps_per_epoch, pct_start=pct_start, anneal_strategy='cos', div_factor=25.0, final_div_factor=1e3)

    print(
        f"[rec] device={device} dataset_size={len(ds)} batch_size={batch_size} "
        f"num_workers={num_workers} epochs={epochs}",
        flush=True,
    )

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt.get('model_state_dict', ckpt)
        # Tolerant load: drop keys with shape mismatch and allow missing/unexpected
        model_sd = model.state_dict()
        clean = {}
        dropped = []
        for k, v in sd.items():
            if k not in model_sd:
                dropped.append((k, 'missing_in_model'))
                continue
            mv = model_sd[k]
            if isinstance(v, torch.Tensor) and isinstance(mv, torch.Tensor) and tuple(v.shape) == tuple(mv.shape):
                clean[k] = v
            else:
                dropped.append((k, f"ckpt={tuple(v.shape) if isinstance(v, torch.Tensor) else type(v)} model={tuple(mv.shape) if isinstance(mv, torch.Tensor) else type(mv)}"))
        missing, unexpected = model.load_state_dict(clean, strict=False)
        try:
            print(f"[rec][resume] loaded {len(clean)} tensors; dropped {len(dropped)}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        except Exception:
            pass

    model.to(device)

    # TensorBoard writer
    writer = None
    global_step = 0
    try:
        if SummaryWriter is not None:
            tb_dir = os.getenv('TB_LOGDIR', 'runs/rec_abinet')
            Path(tb_dir).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
    except Exception:
        writer = None

    # HPO/Isolation friendly directories
    logs_dir = Path(os.getenv('REC_LOGDIR', 'logs'))
    weights_dir = Path(os.getenv('REC_WEIGHTSDIR', 'weights/rec'))
    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Global steps limit for quick runs (HPO)
    global_steps_limit = int(args.max_steps) if args.max_steps else 0
    global_steps = 0

    # EMA setup
    ema = None
    ema_cfg = train_cfg.get('ema', {})
    if bool(ema_cfg.get('enable', True)):
        ema_decay = float(ema_cfg.get('decay', 0.999))
        ema = ExponentialMovingAverage(model, decay=ema_decay)

    stop_training = False
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for step, (imgs, tgts) in enumerate(dl, start=1):
            # Heartbeat
            if step % 50 == 0:
                try:
                    logs_dir.mkdir(exist_ok=True)
                    with open(logs_dir / 'rec_heartbeat.txt', 'w') as hb:
                        hb.write(f"epoch={epoch} step={step}\n")
                except Exception:
                    pass
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.model == 'abinet':
                with torch.cuda.amp.autocast(enabled=(use_amp and device == 'cuda')):
                    # Teacher forcing schedule (linear from start->end over total progress)
                    tf_cfg = train_cfg.get('teacher_forcing', {})
                    tf_start = float(tf_cfg.get('start', 1.0))
                    tf_end = float(tf_cfg.get('end', 0.5))
                    total_epochs = max(1, int(train_cfg.get('epochs', 5)))
                    progress = (epoch - 1 + step / max(1, len(dl))) / total_epochs
                    tf_ratio = tf_start + (tf_end - tf_start) * progress
                    use_teacher = (np.random.rand() < tf_ratio)
                    decoder_input = tgts if use_teacher else model.greedy_tokens(imgs, max_len=tgts.size(1))
                    outputs = model(imgs, tgts, decoder_input=decoder_input)
                    losses = criterion(outputs, tgts)
                    loss = losses['total']
            else:
                with torch.cuda.amp.autocast(enabled=(use_amp and device == 'cuda')):
                    outputs = model(imgs, tgts)  # logits in outputs['logits']
                    logits = outputs['logits']  # (B, seq, C)
                    B, T, C = logits.shape
                    loss = criterion(logits.reshape(B * T, C), tgts.reshape(B * T))
            # NaN/Inf guard
            if not torch.isfinite(loss):
                print(f"[rec][warn] non-finite loss detected at epoch {epoch} step {step}: {float(loss)}; skipping update", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            # Gradient clipping
            max_grad_norm = float(train_cfg.get('max_grad_norm', 1.0))
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                total_norm = torch.tensor(0.0)
            scaler.step(optimizer)
            scaler.update()
            # Update EMA
            if ema is not None:
                ema.update(model)
            total_loss += float(loss.item())
            n_batches += 1
            global_steps += 1
            # TensorBoard scalars
            global_step += 1
            if writer is not None:
                try:
                    writer.add_scalar('train/loss', float(loss.item()), global_step)
                    writer.add_scalar('train/grad_norm', float(total_norm), global_step)
                    writer.add_scalar('train/lr', float(optimizer.param_groups[0]['lr']), global_step)
                except Exception:
                    pass
            if step % 200 == 0:
                avg = total_loss / max(1, n_batches)
                print(f"[rec][epoch {epoch}] step {step} avg_loss={avg:.4f}", flush=True)
            # periodic step checkpoints
            save_every_steps = int(train_cfg.get('save_steps', 0))
            if save_every_steps and (step % save_every_steps == 0):
                torch.save({'model_state_dict': model.state_dict()}, weights_dir / f'{args.model}_ep{epoch}_step{step}.pth')
            # periodic evaluation by steps
            eval_every = int(train_cfg.get('eval_steps', 0))
            if eval_every and (step % eval_every == 0):
                eval_k = int(cfg.get('evaluation', {}).get('sample_size', 2000))
                # Evaluate with EMA weights if enabled
                _backup = None
                if ema is not None:
                    _backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    # Copy EMA weights
                    ema.copy_to(model)
                metrics = _eval_cer(
                    model,
                    recog,
                    images_dir,
                    labels_path,
                    target_h=target_h,
                    target_w=target_w,
                    sample_size=eval_k,
                    batch_size=min(batch_size, 256),
                    device=device,
                )
                # Restore training weights
                if _backup is not None:
                    model.load_state_dict(_backup)
                log = {
                    'epoch': epoch,
                    'step': step,
                    'cer': metrics['cer'],
                    'num': metrics['num'],
                }
                print(f"[rec][eval] {json.dumps(log, ensure_ascii=False)}", flush=True)
                logs_dir.mkdir(exist_ok=True)
                with open(logs_dir / 'rec_eval.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log, ensure_ascii=False) + '\n')
                if writer is not None:
                    try:
                        writer.add_scalar('eval/cer_step', float(metrics['cer']), global_step)
                    except Exception:
                        pass
            if global_steps_limit and global_steps >= global_steps_limit:
                stop_training = True
                break
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}", flush=True)
        if scheduler is not None:
            scheduler.step()

        # epoch-end evaluation
        eval_k = int(cfg.get('evaluation', {}).get('sample_size', 2000))
        # Epoch-end eval (EMA if available)
        _backup = None
        if ema is not None:
            _backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        metrics = _eval_cer(
            model,
            recog,
            images_dir,
            labels_path,
            target_h=target_h,
            target_w=target_w,
            sample_size=eval_k,
            batch_size=min(batch_size, 256),
            device=device,
        )
        if _backup is not None:
            model.load_state_dict(_backup)
        # be robust when no batches were yielded (e.g., empty dataset or early break)
        last_step = locals().get('step', 0)
        log = {
            'epoch': epoch,
            'step': last_step,
            'cer': metrics['cer'],
            'num': metrics['num'],
        }
        print(f"[rec][eval-epoch] {json.dumps(log, ensure_ascii=False)}", flush=True)
        logs_dir.mkdir(exist_ok=True)
        with open(logs_dir / 'rec_eval.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log, ensure_ascii=False) + '\n')
        if writer is not None:
            try:
                writer.add_scalar('eval/cer_epoch', float(metrics['cer']), epoch)
                writer.add_scalar('eval/samples', float(metrics['num']), epoch)
            except Exception:
                pass

        # Save checkpoint per epoch (正しくtorch.saveのみを使用)
        torch.save({'model_state_dict': model.state_dict()}, weights_dir / f'{args.model}_epoch{epoch}.pth')

        # Track and save best (lower CER is better). Prefer EMA weights for best.
        best_meta_path = logs_dir / 'rec_best.json'
        prev_best = None
        if best_meta_path.exists():
            try:
                prev_best = json.loads(best_meta_path.read_text(encoding='utf-8'))
            except Exception:
                prev_best = None
        is_best = (prev_best is None) or (metrics['cer'] < float(prev_best.get('cer', 1e9)))
        if is_best:
            # Save model weights corresponding to the evaluated state.
            # If EMA was used for eval, copy EMA weights into model before saving.
            restore_after = None
            if ema is not None:
                restore_after = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            torch.save({'model_state_dict': model.state_dict()}, weights_dir / f'{args.model}_best.pth')
            if restore_after is not None:
                model.load_state_dict(restore_after)
            meta = {'epoch': epoch, 'cer': metrics['cer'], 'ema': (ema is not None)}
            best_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

        if stop_training:
            print(f"[rec] Reached global step limit ({global_steps_limit}); stopping after epoch {epoch}.", flush=True)
            break

    # Save final
    final_path = weights_dir / f'{args.model}.pth'
    torch.save({'model_state_dict': model.state_dict()}, final_path)
    print(f"Saved final model to {final_path}")
    if writer is not None:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
