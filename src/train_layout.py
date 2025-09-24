#!/usr/bin/env python3
"""
Training entry for layout detection (YOLO/DETR).

Implements DETR training with Hungarian matcher and CE/L1/GIoU losses on
COCO-format datasets. YOLO training is not included here.
"""
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .modeling.layout_yolo import YOLOLayoutDetector
from .modeling.layout_detr import generalized_box_iou


@dataclass
class LayoutTrainConfig:
    images_dir: Path
    annotations: Path
    class_names: List[str]
    batch_size: int = 8
    num_workers: int = 4
    img_size: int = 1024
    epochs: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-4
    use_amp: bool = True
    clip_grad_norm: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: Path = Path('weights/layout')
    # Matcher costs
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    # Loss weights
    loss_ce: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    # YOLO loss weights
    yolo_box: float = 5.0
    yolo_cls: float = 0.5
    yolo_dfl: float = 1.5


class CocoLayoutDataset(Dataset):
    def __init__(self, images_dir: Path, annotations: Path, class_names: List[str], img_size: int = 1024):
        self.images_dir = Path(images_dir)
        self.class_names = class_names
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        self.img_size = int(img_size)

        data = json.loads(Path(annotations).read_text(encoding='utf-8'))
        self.images = {img['id']: img for img in data['images']}
        self.cats = {cat['id']: cat['name'] for cat in data['categories']}
        self.ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data['annotations']:
            self.ann_by_image.setdefault(ann['image_id'], []).append(ann)
        self.ids = list(self.images.keys())
        # Optional PDF fallback (synthetic generation)
        pdf_candidate = Path(annotations).parent / 'synth_layout.pdf'
        self.pdf_path = pdf_candidate if pdf_candidate.exists() else None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.ids[idx]
        info = self.images[img_id]
        img_path = self.images_dir / info['file_name']
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None and self.pdf_path is not None:
            # Render on demand from PDF
            try:
                import pypdfium2 as pdfium
                page_idx = self._infer_page_index(info.get('file_name', ''), img_id)
                doc = pdfium.PdfDocument(str(self.pdf_path))
                if 0 <= page_idx < len(doc):
                    page = doc[page_idx]
                    bitmap = page.render(scale=2.0)
                    pil = bitmap.to_pil()
                    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                bgr = None
        if bgr is None:
            raise FileNotFoundError(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Resize to square img_size with aspect ratio preserved and padding (letterbox)
        h0, w0 = rgb.shape[:2]
        scale = min(self.img_size / h0, self.img_size / w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(rgb, (nw, nh))
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        top = (self.img_size - nh) // 2
        left = (self.img_size - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        img_t = torch.from_numpy(canvas).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        boxes = []
        labels = []
        for ann in self.ann_by_image.get(img_id, []):
            cat_name = self.cats.get(ann['category_id'], None)
            if cat_name is None or cat_name not in self.class_to_id:
                continue
            # bboxをxywh→レターボックス後の正規化cxcywh
            x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
            # scale + pad
            x = x * scale + left
            y = y * scale + top
            w = w * scale
            h = h * scale
            # normalize
            cx = (x + w / 2) / self.img_size
            cy = (y + h / 2) / self.img_size
            nw_ = w / self.img_size
            nh_ = h / self.img_size
            # clip
            cx = float(np.clip(cx, 0, 1))
            cy = float(np.clip(cy, 0, 1))
            nw_ = float(np.clip(nw_, 1e-6, 1))
            nh_ = float(np.clip(nh_, 1e-6, 1))
            boxes.append([cx, cy, nw_, nh_])
            labels.append(self.class_to_id[cat_name])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        return {'image': img_t, 'target': target, 'meta': {'id': img_id, 'path': str(img_path)}}

    def _infer_page_index(self, file_name: str, img_id: int) -> int:
        try:
            stem = Path(file_name).stem
            if stem.startswith('page_'):
                n = int(stem.split('_', 1)[1])
                return max(0, n - 1)
        except Exception:
            pass
        return max(0, img_id - 1)


def layout_collate(batch: List[Dict[str, Any]]):
    images = torch.stack([b['image'] for b in batch], dim=0)
    targets = [b['target'] for b in batch]
    metas = [b['meta'] for b in batch]
    return images, targets, metas


def build_config(cfg: Dict[str, Any]) -> LayoutTrainConfig:
    data = cfg.get('data', {})
    training = cfg.get('training', {})
    loss_cfg = training.get('loss', {})
    return LayoutTrainConfig(
        images_dir=Path(data.get('images_dir', 'data/layout/images')),
        annotations=Path(data.get('annotations', 'data/layout/annotations.json')),
        class_names=data.get('class_names', []),
        batch_size=int(data.get('batch_size', 8)),
        num_workers=int(data.get('num_workers', 4)),
        img_size=int(data.get('val_transform', [{}])[0].get('size', 1024)) if 'val_transform' in data else 1024,
        epochs=int(training.get('epochs', 10)),
        lr=float(training.get('optimizer', {}).get('lr', 2e-4)),
        weight_decay=float(training.get('optimizer', {}).get('weight_decay', 1e-4)),
        use_amp=bool(training.get('use_amp', True)),
        clip_grad_norm=float(training.get('clip_grad_norm', 0.1)),
        cost_class=float(loss_cfg.get('cost_class', 1.0)),
        cost_bbox=float(loss_cfg.get('cost_bbox', 5.0)),
        cost_giou=float(loss_cfg.get('cost_giou', 2.0)),
        loss_ce=float(loss_cfg.get('ce', 1.0)) if 'ce' in loss_cfg else float(loss_cfg.get('cls', 1.0)),
        loss_bbox=float(loss_cfg.get('bbox', 5.0)),
        loss_giou=float(loss_cfg.get('giou', 2.0)),
        yolo_box=float(loss_cfg.get('box', 5.0)),
        yolo_cls=float(loss_cfg.get('cls', 0.5)),
        yolo_dfl=float(loss_cfg.get('dfl', 1.5)),
    )


class YOLOLayoutLoss(nn.Module):
    """Simplified YOLO loss with DFL/cls/box.
    - Assignment: center-based; one positive per GT on selected scale.
    - DFL: CE between predicted distributions and discretized distances.
    - Box: GIoU loss between decoded boxes and GT.
    """
    def __init__(self, num_classes: int, reg_max: int = 16,
                 w_box: float = 5.0, w_cls: float = 0.5, w_dfl: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.w_box = w_box
        self.w_cls = w_cls
        self.w_dfl = w_dfl
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, outputs_levels: List[Dict[str, torch.Tensor]],
                targets: List[Dict[str, torch.Tensor]], img_size: int) -> Dict[str, torch.Tensor]:
        device = outputs_levels[0]['pred'].device
        total_box = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        total_dfl = torch.tensor(0.0, device=device)
        n_pos = 0

        # Precompute grid sizes per level
        grids = [(o['hw'][0], o['hw'][1]) for o in outputs_levels]

        for b in range(outputs_levels[0]['pred'].shape[0]):
            gts = targets[b]
            if gts['boxes'].numel() == 0:
                continue
            # choose scale by box size
            for k in range(len(gts['boxes'])):
                cx, cy, w, h = gts['boxes'][k].tolist()
                cls_id = int(gts['labels'][k].item())
                scale = max(w, h) * img_size
                lvl = 0 if scale < 64 else (1 if scale < 128 else 2)
                H, W = grids[lvl]
                gx = min(W - 1, max(0, int(cx * W)))
                gy = min(H - 1, max(0, int(cy * H)))
                pred = outputs_levels[lvl]['pred'][b]  # (H, W, C)
                # pick this location
                logits = pred[gy, gx]  # (C)
                dist_logits = logits[: 4 * self.reg_max].reshape(4, self.reg_max)
                cls_logits = logits[4 * self.reg_max: 4 * self.reg_max + self.num_classes]

                # DFL target distances in grid units from anchor center
                ax = (gx + 0.5)
                ay = (gy + 0.5)
                x1 = (cx - w / 2) * W
                x2 = (cx + w / 2) * W
                y1 = (cy - h / 2) * H
                y2 = (cy + h / 2) * H
                t_left = torch.tensor(max(0.0, ax - x1), device=device)
                t_top = torch.tensor(max(0.0, ay - y1), device=device)
                t_right = torch.tensor(max(0.0, x2 - ax), device=device)
                t_bottom = torch.tensor(max(0.0, y2 - ay), device=device)
                t = torch.stack([t_left, t_top, t_right, t_bottom])
                # discretize to bins [0..reg_max-1]
                t_bins = torch.clamp(t.long(), 0, self.reg_max - 1)
                # DFL loss: CE per side
                dfl = sum(self.ce(dist_logits[i].unsqueeze(0), t_bins[i].unsqueeze(0)) for i in range(4)) / 4.0
                total_dfl += dfl

                # Decode expected distances to box for IoU loss
                probs = torch.softmax(dist_logits, dim=-1)
                rng = torch.arange(self.reg_max, device=device).float()
                d_exp = torch.sum(probs * rng, dim=-1)  # (4)
                # convert to normalized xyxy
                x1p = (ax - d_exp[0]) / W
                y1p = (ay - d_exp[1]) / H
                x2p = (ax + d_exp[2]) / W
                y2p = (ay + d_exp[3]) / H
                pred_box = torch.stack([x1p, y1p, x2p, y2p])
                gt_box = torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], device=device)
                giou = generalized_box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).mean()
                box_loss = 1.0 - giou
                total_box += box_loss

                # Classification loss (pos only)
                target = torch.zeros(self.num_classes, device=device)
                target[cls_id] = 1.0
                cls_loss = self.bce(cls_logits, target)
                total_cls += cls_loss
                n_pos += 1

        if n_pos == 0:
            return {'total': total_box * 0.0, 'box': total_box * 0.0, 'cls': total_cls * 0.0, 'dfl': total_dfl * 0.0}

        return {
            'total': self.w_box * (total_box / n_pos) + self.w_cls * (total_cls / n_pos) + self.w_dfl * (total_dfl / n_pos),
            'box': total_box / n_pos,
            'cls': total_cls / n_pos,
            'dfl': total_dfl / n_pos,
        }


def detr_loss(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]],
              matcher, cfg: LayoutTrainConfig) -> Dict[str, torch.Tensor]:
    # Lazy import to avoid torchvision dependency when not using DETR
    from .modeling.layout_detr import generalized_box_iou
    indices = matcher(outputs, targets)
    bs, num_queries = outputs['logits'].shape[:2]

    src_logits = outputs['logits']
    src_boxes = outputs['boxes']

    idx = [(i, j) for i, (i_idx, j_idx) in enumerate(indices) for j in i_idx.tolist()]
    if len(idx) == 0:
        # no targets in batch
        ce = src_logits.new_zeros(())
        l1 = src_boxes.new_zeros(())
        giou = src_boxes.new_zeros(())
        total = ce + l1 + giou
        return {'total': total, 'ce': ce, 'l1': l1, 'giou': giou}

    src_flat_idx = []
    tgt_labels = []
    src_box_list = []
    tgt_box_list = []

    for b, (src_ids, tgt_ids) in enumerate(indices):
        src_ids = src_ids.to(torch.long)
        tgt_ids = tgt_ids.to(torch.long)
        src_flat_idx.extend([b * num_queries + s for s in src_ids.tolist()])
        tgt_labels.append(targets[b]['labels'][tgt_ids])
        src_box_list.append(src_boxes[b, src_ids])
        tgt_box_list.append(targets[b]['boxes'][tgt_ids])

    src_flat_idx = torch.tensor(src_flat_idx, device=src_logits.device, dtype=torch.long)
    tgt_labels = torch.cat(tgt_labels, dim=0)
    src_matched_logits = src_logits.flatten(0, 1)[src_flat_idx]
    ce = F.cross_entropy(src_matched_logits, tgt_labels, reduction='mean') * cfg.loss_ce

    src_b = torch.cat(src_box_list, dim=0)
    tgt_b = torch.cat(tgt_box_list, dim=0)
    l1 = F.l1_loss(src_b, tgt_b, reduction='mean') * cfg.loss_bbox
    giou = (1.0 - generalized_box_iou(
        box_cxcywh_to_xyxy(src_b), box_cxcywh_to_xyxy(tgt_b)
    ).mean()) * cfg.loss_giou

    total = ce + l1 + giou
    return {'total': total, 'ce': ce, 'l1': l1, 'giou': giou}


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def train_detr(cfg: LayoutTrainConfig):
    # Lazy import to avoid torchvision dependency when not using DETR
    from .modeling.layout_detr import DETRLayoutDetector, HungarianMatcher
    ds = CocoLayoutDataset(cfg.images_dir, cfg.annotations, cfg.class_names, cfg.img_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=layout_collate)

    model = DETRLayoutDetector(num_classes=len(cfg.class_names), device=cfg.device)
    model.train()
    matcher = HungarianMatcher(cfg.cost_class, cfg.cost_bbox, cfg.cost_giou)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    global_step = 0
    best_loss = math.inf
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        t0 = time.time()
        n_batches = 0
        for images, targets, _ in dl:
            images = images.to(cfg.device, non_blocking=True)
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = model(images)
                losses = detr_loss(outputs, targets, matcher, cfg)
                loss = losses['total']
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            total_loss += float(loss.item())
            n_batches += 1
        dt = time.time() - t0
        avg = total_loss / max(1, n_batches)
        print(f"[DETR][Epoch {epoch}/{cfg.epochs}] loss={avg:.4f} ({dt:.1f}s)")

        # Save per-epoch
        ep_path = cfg.out_dir / f'detr_epoch{epoch}.pth'
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ep_path)
        if avg < best_loss:
            best_loss = avg
            torch.save({'model_state_dict': model.state_dict()}, cfg.out_dir / 'detr_best.pth')

        # Simple validation mAP@0.5 on a subset
        try:
            map50, map5095 = _evaluate_detr(model, ds, max_samples=50)
            print(f"[DETR][Epoch {epoch}] mAP@0.5(val): {map50:.4f}, mAP@[.50:.95](val): {map5095:.4f}")
        except Exception as e:
            print(f"[WARN] DETR eval failed: {e}")


def _evaluate_detr(model: nn.Module, ds: Dataset, max_samples: int = 50,
                   conf_thresh: float = 0.5) -> Tuple[float, float]:
    import numpy as np
    from .utils.metrics import bbox_iou
    model.eval()
    maps = []
    maps_50 = []
    num_classes = model.num_classes
    # collect predictions and gts per image
    for i in range(min(max_samples, len(ds))):
        sample = ds[i]
        img = sample['image'].unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(img)
        logits = out['logits'][0]  # (num_queries, C+1)
        boxes = out['boxes'][0]    # (num_queries, 4) in cxcywh (0..1)
        probs = torch.softmax(logits, dim=-1)
        scores, labels = probs[..., :num_classes].max(dim=-1)
        # filter background by checking prob of bg class
        keep = scores > conf_thresh
        pred = []
        for j in torch.nonzero(keep, as_tuple=False).flatten().tolist():
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

        # AP@0.5 and mAP@[.50:.95]
        ap50 = _ap_at_iou_local(pred, gts, 0.5)
        ap_all = []
        for t in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
            ap_all.append(_ap_at_iou_local(pred, gts, t))
        maps_50.append(ap50)
        maps.append(sum(ap_all)/len(ap_all) if ap_all else 0.0)
    model.train()
    map50 = float(sum(maps_50)/len(maps_50)) if maps_50 else 0.0
    map5095 = float(sum(maps)/len(maps)) if maps else 0.0
    return map50, map5095


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
    ap = argparse.ArgumentParser(description='Train layout detection (YOLO/DETR)')
    ap.add_argument('--config', type=str, required=False, default='configs/layout_yolo.yaml')
    ap.add_argument('--model', type=str, default='detr', choices=['yolo', 'detr'])
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) if args.config and Path(args.config).exists() else {}
    print('[INFO] Loaded config keys:', list(cfg_raw.keys()))

    if args.model == 'detr':
        cfg = build_config(cfg_raw)
        if not cfg.class_names:
            raise ValueError('configs/layout_yolo.yaml の data.class_names にDocLayNet互換のクラス名を指定してください')
        train_detr(cfg)
    else:
        # YOLO training path
        cfg = build_config(cfg_raw)
        if not cfg.class_names:
            raise ValueError('configs/layout_yolo.yaml の data.class_names にDocLayNet互換のクラス名を指定してください')
        train_yolo(cfg)


def train_yolo(cfg: LayoutTrainConfig):
    ds = CocoLayoutDataset(cfg.images_dir, cfg.annotations, cfg.class_names, cfg.img_size)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=layout_collate)

    model = YOLOLayoutDetector(num_classes=len(cfg.class_names), model_size='m', device=cfg.device)
    model.train()
    criterion = YOLOLayoutLoss(num_classes=len(cfg.class_names), reg_max=16,
                               w_box=cfg.yolo_box, w_cls=cfg.yolo_cls, w_dfl=cfg.yolo_dfl)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    best = math.inf
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.epochs + 1):
        total = box_l = cls_l = dfl_l = 0.0
        n_batches = 0
        t0 = time.time()
        for images, targets, _ in dl:
            images = images.to(cfg.device, non_blocking=True)
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = model(images)  # training: raw per-level
                losses = criterion(outputs, targets, cfg.img_size)
                loss = losses['total']
            scaler.scale(loss).backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item())
            box_l += float(losses['box'].item())
            cls_l += float(losses['cls'].item())
            dfl_l += float(losses['dfl'].item())
            n_batches += 1
        dt = time.time() - t0
        avg_total = total / max(1, n_batches)
        print(f"[YOLO][Epoch {epoch}/{cfg.epochs}] total={avg_total:.4f} box={box_l/max(1,n_batches):.4f} cls={cls_l/max(1,n_batches):.4f} dfl={dfl_l/max(1,n_batches):.4f} ({dt:.1f}s)")

        # Save per-epoch
        ep_path = cfg.out_dir / f'yolo_epoch{epoch}.pth'
        torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, ep_path)
        if avg_total < best:
            best = avg_total
            torch.save({'model_state_dict': model.state_dict()}, cfg.out_dir / 'yolo_best.pth')


if __name__ == '__main__':
    main()
