#!/usr/bin/env python3
"""
Training entry for table structure recognition (TATR).

Implements Hungarian matching and CE/L1/GIoU losses for TATR-like outputs.
Dataset expects PubTabNet互換（簡易）: 画像 + 注釈（boxes[cxcywh], labels[int]）。
"""
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .modeling.table_tatr import TATR
from .modeling.layout_detr import HungarianMatcher, generalized_box_iou
import json as _json


@dataclass
class TATRTrainConfig:
    images_dir: Path
    annotations: Path
    batch_size: int = 4
    num_workers: int = 4
    img_size: int = 800
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    clip_grad_norm: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: Path = Path('weights/table')
    # Matcher costs
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    # Loss weights
    loss_ce: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0


class PubTabNetLikeDataset(Dataset):
    def __init__(self, images_dir: Path, annotations: Path, img_size: int = 800):
        self.images_dir = Path(images_dir)
        self.img_size = int(img_size)
        data = json.loads(Path(annotations).read_text(encoding='utf-8'))
        # 想定構造: {"annotations": [{"image": "xxx.jpg", "boxes": [[cx,cy,w,h],...], "labels": [int,...]}]}
        self.entries: List[Dict[str, Any]] = data['annotations'] if 'annotations' in data else data

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.entries[idx]
        img_path = self.images_dir / item.get('image', item.get('image_path', ''))
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        scale = min(self.img_size / h0, self.img_size / w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(rgb, (nw, nh))
        canvas = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
        top = (self.img_size - nh) // 2
        left = (self.img_size - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        # Convert image to tensor (robust to environments where Torch reports
        # "Numpy is not available" in worker processes)
        try:
            img_t = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0
        except RuntimeError as e:
            if 'Numpy is not available' in str(e):
                # Fallback: via Python list (slower but unblocks training)
                img_t = torch.tensor(canvas.tolist(), dtype=torch.float32).permute(2, 0, 1) / 255.0
            else:
                raise
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        boxes = torch.tensor(item.get('boxes', []), dtype=torch.float32)
        labels = torch.tensor(item.get('labels', []), dtype=torch.long)

        return {'image': img_t, 'target': {'boxes': boxes, 'labels': labels}, 'meta': {'path': str(img_path)}}


def tatr_collate(batch: List[Dict[str, Any]]):
    images = torch.stack([b['image'] for b in batch], dim=0)
    targets = [b['target'] for b in batch]
    metas = [b['meta'] for b in batch]
    return images, targets, metas


def build_config(cfg: Dict[str, Any]) -> TATRTrainConfig:
    data = cfg.get('data', {})
    training = cfg.get('training', {})
    loss_cfg = training.get('loss', {})
    return TATRTrainConfig(
        images_dir=Path(data.get('images_dir', 'data/table/images')),
        annotations=Path(data.get('annotations', 'data/table/annotations.json')),
        batch_size=int(data.get('batch_size', 4)),
        num_workers=int(data.get('num_workers', 4)),
        img_size=int(data.get('train_transform', [{}])[0].get('size', [800, 800])[0]) if 'train_transform' in data else 800,
        epochs=int(training.get('epochs', 10)),
        lr=float(training.get('optimizer', {}).get('lr', 1e-4)),
        weight_decay=float(training.get('optimizer', {}).get('weight_decay', 1e-4)),
        use_amp=bool(training.get('use_amp', True)),
        clip_grad_norm=float(training.get('clip_max_norm', 0.1)),
        cost_class=float(loss_cfg.get('cost_class', 1.0)),
        cost_bbox=float(loss_cfg.get('cost_bbox', 5.0)),
        cost_giou=float(loss_cfg.get('cost_giou', 2.0)),
        loss_ce=float(loss_cfg.get('loss_ce', 1.0)),
        loss_bbox=float(loss_cfg.get('loss_bbox', 5.0)),
        loss_giou=float(loss_cfg.get('loss_giou', 2.0)),
    )


def tatr_loss(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]],
              matcher: HungarianMatcher, cfg: TATRTrainConfig) -> Dict[str, torch.Tensor]:
    indices = matcher(outputs, targets)
    bs, num_queries = outputs['logits'].shape[:2]
    src_logits = outputs['logits']
    src_boxes = outputs['boxes']

    # 分類損失
    matched_logits = []
    tgt_labels = []
    matched_src_boxes = []
    matched_tgt_boxes = []

    for b, (src_ids, tgt_ids) in enumerate(indices):
        if len(src_ids) == 0 or len(tgt_ids) == 0:
            continue
        matched_logits.append(src_logits[b, src_ids])
        tgt_labels.append(targets[b]['labels'][tgt_ids])
        matched_src_boxes.append(src_boxes[b, src_ids])
        matched_tgt_boxes.append(targets[b]['boxes'][tgt_ids])

    if not matched_logits:
        zero = src_logits.new_zeros(())
        return {'total': zero, 'ce': zero, 'l1': zero, 'giou': zero}

    matched_logits_t = torch.cat(matched_logits, dim=0)
    tgt_labels_t = torch.cat(tgt_labels, dim=0)
    ce = F.cross_entropy(matched_logits_t, tgt_labels_t, reduction='mean') * cfg.loss_ce

    src_b = torch.cat(matched_src_boxes, dim=0)
    tgt_b = torch.cat(matched_tgt_boxes, dim=0)
    l1 = F.l1_loss(src_b, tgt_b, reduction='mean') * cfg.loss_bbox
    giou = (1.0 - generalized_box_iou(box_cxcywh_to_xyxy(src_b), box_cxcywh_to_xyxy(tgt_b)).mean()) * cfg.loss_giou

    total = ce + l1 + giou
    return {'total': total, 'ce': ce, 'l1': l1, 'giou': giou}


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def train_tatr(cfg: TATRTrainConfig):
    ds = PubTabNetLikeDataset(cfg.images_dir, cfg.annotations, cfg.img_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=tatr_collate)

    model = TATR()
    model.to(cfg.device)
    model.train()
    matcher = HungarianMatcher(cfg.cost_class, cfg.cost_bbox, cfg.cost_giou)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    best = math.inf
    # prepare metrics log
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / 'train_table_metrics.jsonl'
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for images, targets, _ in dl:
            images = images.to(cfg.device)
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = model(images, task='structure')
                losses = tatr_loss(outputs, targets, matcher, cfg)
                loss = losses['total']
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            n_batches += 1
        dt = time.time() - t0
        avg = total_loss / max(1, n_batches)
        print(f"[TATR][Epoch {epoch}/{cfg.epochs}] loss={avg:.4f} ({dt:.1f}s)")
        # Save per-epoch
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        ep_path = cfg.out_dir / f'tatr_epoch{epoch}.pth'
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ep_path)
        print(f"[TATR] Saved epoch checkpoint: {ep_path}")
        if avg < best:
            best = avg
            torch.save({'model_state_dict': model.state_dict()}, cfg.out_dir / 'tatr_best.pth')
            print(f"[TATR] Updated best checkpoint: {cfg.out_dir / 'tatr_best.pth'} (best_loss={best:.4f})")

        # Simple validation metric (mAP@0.5 over structure classes) on a subset
        try:
            map50 = _evaluate_tatr(model, ds, max_samples=50)
            print(f"[TATR][Epoch {epoch}] mAP@0.5(val): {map50:.4f}")
            # append metrics line
            try:
                metrics_line = {
                    'epoch': epoch,
                    'loss_avg': float(avg),
                    'map50_val': float(map50),
                    'seconds': float(dt),
                    'timestamp': time.time(),
                }
                with metrics_path.open('a', encoding='utf-8') as mf:
                    mf.write(_json.dumps(metrics_line, ensure_ascii=False) + "\n")
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] TATR eval failed: {e}")

    # finalize
    try:
        summary = {
            'best_loss': float(best),
            'epochs': cfg.epochs,
            'weights_dir': str(cfg.out_dir),
            'best_ckpt': str((cfg.out_dir / 'tatr_best.pth').resolve()),
            'ended_at': time.time(),
        }
        (results_dir / 'train_table_summary.json').write_text(
            _json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print("[TATR] Training completed. Summary saved to results/train_table_summary.json")
    except Exception:
        pass


def _evaluate_tatr(model: TATR, ds: Dataset, max_samples: int = 50, conf_thresh: float = 0.5) -> float:
    """Approximate evaluation using AP@0.5 on structure classes.
    For full TEDS, use src/eval_table.py with HTML pairs after pipeline export.
    """
    import numpy as np
    from .utils.metrics import bbox_iou
    model.eval()
    aps = []
    num_classes = 6
    for i in range(min(max_samples, len(ds))):
        sample = ds[i]
        img = sample['image'].unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(img, task='structure')
        logits = out['logits'][0]  # (num_queries, C)
        boxes = out['boxes'][0]    # (num_queries, 4) in cxcywh (0..1)
        probs = torch.softmax(logits, dim=-1)
        scores, labels = probs.max(dim=-1)
        pred = []
        for j in range(boxes.shape[0]):
            if scores[j] < conf_thresh:
                continue
            cx, cy, w, h = boxes[j].tolist()
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            pred.append({'bbox': [x1, y1, x2, y2], 'class': int(labels[j].item()), 'score': float(scores[j].item())})
        tgt = sample['target']
        gts = []
        for k in range(len(tgt['boxes'])):
            cx, cy, w, h = tgt['boxes'][k].tolist()
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            gts.append({'bbox': [x1, y1, x2, y2], 'class': int(tgt['labels'][k].item())})
        aps.append(_ap_at_iou_local(pred, gts, 0.5))
    model.train()
    return float(sum(aps)/len(aps)) if aps else 0.0


def _ap_at_iou_local(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]], iou_thresh: float = 0.5) -> float:
    import numpy as np
    from .utils.metrics import bbox_iou
    if not preds:
        return 0.0
    classes = sorted(set([p['class'] for p in preds] + [g['class'] for g in gts]))
    ap_list = []
    for cls in classes:
        cls_preds = [p for p in preds if p['class'] == cls]
        cls_preds = sorted(cls_preds, key=lambda x: x.get('score', 0.0), reverse=True)
        cls_gts = [g for g in gts if g['class'] == cls]
        matched = set()
        tp = []
        fp = []
        for p in cls_preds:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(cls_gts):
                if j in matched:
                    continue
                iou = bbox_iou(p['bbox'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_thresh and best_j >= 0:
                tp.append(1)
                fp.append(0)
                matched.add(best_j)
            else:
                tp.append(0)
                fp.append(1)
        if not cls_preds:
            ap_list.append(0.0)
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(1, len(cls_gts))
        precisions = tp_cum / np.maximum(1, tp_cum + fp_cum)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0.0
            ap += p / 11
        ap_list.append(float(ap))
    return float(np.mean(ap_list)) if ap_list else 0.0


def main():
    ap = argparse.ArgumentParser(description='Train table structure (TATR)')
    ap.add_argument('--config', type=str, required=False, default='configs/table_tatr.yaml')
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) if args.config and Path(args.config).exists() else {}
    print('[INFO] Loaded config keys:', list(cfg_raw.keys()))

    cfg = build_config(cfg_raw)
    train_tatr(cfg)


if __name__ == '__main__':
    main()
