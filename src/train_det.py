#!/usr/bin/env python3
"""
Training entry for text detection (DBNet++ / YOLO).

Implements DBNet++ target generation (gt_prob/gt_thresh/gt_mask/thresh_mask)
from COCO-style polygons or bboxes, and wires DBNetLoss with AMP, grad clip,
and checkpointing. YOLO training is not implemented in this file yet.
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

from .modeling.det_dbnet import DBNetPP, DBNetLoss


@dataclass
class DetTrainConfig:
    images_dir: Path
    annotations: Path
    val_images_dir: Path = None  # optional
    val_annotations: Path = None  # optional
    batch_size: int = 8
    num_workers: int = 4
    short_side: int = 1024  # resize short side
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    clip_grad_norm: float = 5.0
    save_steps: int = 1000
    save_total_limit: int = 3
    eval_steps: int = 0  # optional
    alpha: float = 1.0
    beta: float = 10.0
    ohem_ratio: float = 3.0
    # Target generation options
    shrink_ratio: float = 0.4
    use_shrink_prob: bool = True
    thresh_per_instance: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: Path = Path('weights/det')


class CocoTextDataset(Dataset):
    """COCO互換アノテーションからDBNet++学習用ターゲットを生成するデータセット。

    - 入力画像は短辺をshort_sideに合わせ、32の倍数に丸めてリサイズ
    - ターゲットマップは画像解像度に合わせて生成
    - segmentation(ポリゴン)が無ければbboxから矩形ポリゴンを生成
    """

    def __init__(self, images_dir: Path, annotations: Path, short_side: int = 1024,
                 shrink_ratio: float = 0.4, use_shrink_prob: bool = True,
                 thresh_per_instance: bool = True):
        self.images_dir = Path(images_dir)
        self.short_side = int(short_side)
        self.shrink_ratio = float(shrink_ratio)
        self.use_shrink_prob = bool(use_shrink_prob)
        self.thresh_per_instance = bool(thresh_per_instance)
        data = json.loads(Path(annotations).read_text(encoding='utf-8'))
        self.images = {img['id']: img for img in data['images']}
        self.ann_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data['annotations']:
            self.ann_by_image.setdefault(ann['image_id'], []).append(ann)
        self.ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = self.images_dir / img_info['file_name']
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        # 短辺基準でスケールを算出
        scale = self.short_side / min(orig_h, orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        # 32の倍数に丸め
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        if new_h <= 0:
            new_h = 32
        if new_w <= 0:
            new_w = 32

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # アノテーションのポリゴンをスケール
        polys: List[np.ndarray] = []
        for ann in self.ann_by_image.get(img_id, []):
            segs = ann.get('segmentation', [])
            if isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], list):
                # COCO polygon list (x1,y1,x2,y2,...)
                for seg in segs:
                    arr = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    arr[:, 0] *= (new_w / orig_w)
                    arr[:, 1] *= (new_h / orig_h)
                    polys.append(arr)
            else:
                # bbox [x,y,w,h] から矩形ポリゴン
                bbox = ann.get('bbox', None)
                if bbox is not None and len(bbox) == 4:
                    x, y, w, h = bbox
                    x *= (new_w / orig_w)
                    y *= (new_h / orig_h)
                    w *= (new_w / orig_w)
                    h *= (new_h / orig_h)
                    rect = np.array(
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
                    )
                    polys.append(rect)

        # 画像テンソル化（ImageNet正規化）
        img_t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        # GTマップ生成（画像サイズベース）
        gt_prob, gt_thresh, gt_mask, thresh_mask = self._generate_targets((new_h, new_w), polys)

        # GTボックス（評価用）
        gt_boxes: List[List[float]] = []
        for poly in polys:
            if poly.shape[0] < 3:
                continue
            x1 = float(np.min(poly[:, 0]))
            y1 = float(np.min(poly[:, 1]))
            x2 = float(np.max(poly[:, 0]))
            y2 = float(np.max(poly[:, 1]))
            gt_boxes.append([x1, y1, x2, y2])

        return {
            'image': img_t,
            'gt_prob': gt_prob,
            'gt_thresh': gt_thresh,
            'gt_mask': gt_mask,
            'thresh_mask': thresh_mask,
            'meta': {
                'id': img_id,
                'size': (new_h, new_w),
                'orig_size': (orig_h, orig_w),
                'path': str(img_path)
            },
            'gt_boxes': gt_boxes
        }

    def _generate_targets(
        self, size_hw: Tuple[int, int], polygons: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DBNet++向けのターゲットマップを生成。

        - gt_prob: テキスト領域=1, 背景=0
        - gt_thresh: 領域中心=1→境界で0へ滑らかに遷移（距離変換ベース）
        - gt_mask: 損失計算領域（ここでは全域1）
        - thresh_mask: しきい値回帰の計算領域（ここではテキスト領域内=1）
        """
        h, w = size_hw
        prob_full = np.zeros((h, w), dtype=np.uint8)
        prob_shrink = np.zeros((h, w), dtype=np.uint8)
        thresh_map = np.zeros((h, w), dtype=np.float32)

        # 1) 塗りつぶし（full）と収縮領域（approx）を生成
        for poly in polygons:
            if poly.shape[0] < 3:
                continue
            poly_i = poly.astype(np.int32)
            cv2.fillPoly(prob_full, [poly_i], 1)
            # 収縮距離の近似（DBNetの式に倣う）
            area = float(cv2.contourArea(poly_i))
            peri = float(cv2.arcLength(poly_i, True))
            if peri <= 0 or area <= 0:
                shrink_pixels = 1
            else:
                r = float(self.shrink_ratio)
                shrink_pixels = int(max(1.0, area * (1 - r * r) / (peri + 1e-6)))
            # インスタンスをモルフォロジで収縮
            inst = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(inst, [poly_i], 1)
            k = max(1, shrink_pixels)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            eroded = cv2.erode(inst, kernel)
            prob_shrink = np.maximum(prob_shrink, eroded)

        prob = prob_shrink if self.use_shrink_prob else prob_full

        # 2) しきい値マップ（インスタンスごとに距離正規化してmaxで合成）
        if self.thresh_per_instance and polygons:
            for poly in polygons:
                if poly.shape[0] < 3:
                    continue
                inst = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(inst, [poly.astype(np.int32)], 1)
                dist = cv2.distanceTransform(inst, distanceType=cv2.DIST_L2, maskSize=3)
                m = float(dist.max())
                if m > 0:
                    thresh_local = (dist / (m + 1e-6)).astype(np.float32)
                    thresh_map = np.maximum(thresh_map, thresh_local)
        else:
            dist_inner = cv2.distanceTransform(prob_full, distanceType=cv2.DIST_L2, maskSize=3)
            m = float(dist_inner.max())
            thresh_map = (dist_inner / (m + 1e-6)).astype(np.float32) if m > 0 else np.zeros_like(dist_inner, dtype=np.float32)

        gt_prob = torch.from_numpy(prob.astype(np.float32))[None, ...]
        gt_thresh = torch.from_numpy(thresh_map.astype(np.float32))[None, ...]
        gt_mask = torch.ones((1, h, w), dtype=torch.float32)
        thresh_mask = torch.from_numpy(prob_full.astype(np.float32))[None, ...]

        return gt_prob, gt_thresh, gt_mask, thresh_mask


def collate_fn(batch: List[Dict[str, Any]]):
    # 画像のサイズは同一（short_side & 32丸め）なのでそのままスタック
    images = torch.stack([b['image'] for b in batch], dim=0)
    gt_prob = torch.stack([b['gt_prob'] for b in batch], dim=0)
    gt_thresh = torch.stack([b['gt_thresh'] for b in batch], dim=0)
    gt_mask = torch.stack([b['gt_mask'] for b in batch], dim=0)
    thresh_mask = torch.stack([b['thresh_mask'] for b in batch], dim=0)
    metas = [b['meta'] for b in batch]
    return images, {
        'gt_prob': gt_prob, 'gt_thresh': gt_thresh, 'gt_mask': gt_mask, 'thresh_mask': thresh_mask
    }, metas


def build_config(cfg_dict: Dict[str, Any]) -> DetTrainConfig:
    data = cfg_dict.get('data', {})
    training = cfg_dict.get('training', {})
    images_dir = data.get('images_dir')
    annotations = data.get('annotations')

    # dataset指定のみの場合、ディレクトリ推測はユーザ側設定とし、明示指定を推奨
    if not images_dir or not annotations:
        raise ValueError(
            "configs/det_dbnet.yaml の data.images_dir / data.annotations を指定してください"
        )

    return DetTrainConfig(
        images_dir=Path(images_dir),
        annotations=Path(annotations),
        val_images_dir=Path(data.get('val_images_dir')) if data.get('val_images_dir') else None,
        val_annotations=Path(data.get('val_annotations')) if data.get('val_annotations') else None,
        batch_size=int(data.get('batch_size', 8)),
        num_workers=int(data.get('num_workers', 4)),
        short_side=int(data.get('short_side', 1024)),
        epochs=int(training.get('epochs', 10)),
        lr=float(training.get('optimizer', {}).get('lr', 1e-3)),
        weight_decay=float(training.get('optimizer', {}).get('weight_decay', 1e-4)),
        use_amp=bool(training.get('use_amp', True)),
        clip_grad_norm=float(training.get('clip_grad_norm', 5.0)),
        save_steps=int(training.get('save_steps', 1000)),
        save_total_limit=int(training.get('save_total_limit', 3)),
        eval_steps=int(training.get('eval_steps', 0)),
        alpha=float(training.get('loss', {}).get('alpha', 1.0)),
        beta=float(training.get('loss', {}).get('beta', 10.0)),
        ohem_ratio=float(training.get('loss', {}).get('ohem_ratio', 3.0)),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        out_dir=Path('weights/det')
    )


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': step
    }, path)


def train_dbnet(cfg: DetTrainConfig):
    dataset = CocoTextDataset(
        cfg.images_dir, cfg.annotations, cfg.short_side,
        shrink_ratio=cfg.shrink_ratio,
        use_shrink_prob=cfg.use_shrink_prob,
        thresh_per_instance=cfg.thresh_per_instance,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True,
                        collate_fn=collate_fn)

    model = DBNetPP(backbone='resnet18' if cfg.short_side <= 768 else 'resnet50', device=cfg.device)
    model.train()

    criterion = DBNetLoss(alpha=cfg.alpha, beta=cfg.beta, ohem_ratio=cfg.ohem_ratio).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)

    global_step = 0
    best_loss = math.inf
    ckpts: List[Path] = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_losses: Dict[str, float] = {'total': 0.0, 'prob': 0.0, 'thresh': 0.0, 'binary': 0.0}
        n_batches = 0
        t0 = time.time()

        for images, targets, _ in loader:
            images = images.to(cfg.device, non_blocking=True)
            t = {k: v.to(cfg.device, non_blocking=True) for k, v in targets.items()}

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = model(images)
                losses = criterion(outputs, t)
                loss = losses['total']

            scaler.scale(loss).backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # logging
            for k in epoch_losses.keys():
                epoch_losses[k] += float(losses[k].item()) if k in losses else 0.0
            n_batches += 1
            global_step += 1

            # checkpoint
            if cfg.save_steps and global_step % cfg.save_steps == 0:
                ckpt_path = cfg.out_dir / f'dbnet_step{global_step}.pth'
                save_checkpoint(ckpt_path, model, optimizer, global_step)
                ckpts.append(ckpt_path)
                # ローテーション
                if len(ckpts) > cfg.save_total_limit:
                    old = ckpts.pop(0)
                    try:
                        old.unlink(missing_ok=True)
                    except Exception:
                        pass

        dt = time.time() - t0
        avg = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
        print(f"[DBNet][Epoch {epoch}/{cfg.epochs}] total={avg['total']:.4f} prob={avg['prob']:.4f}"
              f" thresh={avg['thresh']:.4f} binary={avg['binary']:.4f} ({dt:.1f}s)")

        # best checkpoint (by total loss)
        if avg['total'] < best_loss:
            best_loss = avg['total']
            best_path = cfg.out_dir / 'dbnet_best.pth'
            save_checkpoint(best_path, model, optimizer, global_step)

        # optional evaluation on validation (HMean)
        if cfg.val_images_dir and cfg.val_annotations:
            try:
                hmean = _evaluate_dbnet(model, cfg, max_samples=50)
                print(f"[DBNet][Epoch {epoch}] HMean@0.5(val): {hmean:.4f}")
            except Exception as e:
                print(f"[WARN] DBNet eval failed: {e}")

    # final save
    final_path = cfg.out_dir / 'dbnet_final.pth'
    save_checkpoint(final_path, model, optimizer, global_step)
    print(f"Saved final model to {final_path}")


def _evaluate_dbnet(model: nn.Module, cfg: DetTrainConfig, max_samples: int = 50,
                    box_thresh: float = 0.7, iou_thresh: float = 0.5) -> float:
    """Simple H-mean evaluation on a subset of validation data."""
    from .utils.metrics import detection_metrics
    # Build a validation dataset instance (reuse CocoTextDataset for resizing)
    val_ds = CocoTextDataset(cfg.val_images_dir, cfg.val_annotations, cfg.short_side)
    model.eval()
    tp = fp = fn = 0
    import torch
    import numpy as np
    import cv2
    with torch.no_grad():
        for i in range(min(max_samples, len(val_ds))):
            sample = val_ds[i]
            img = sample['image'].unsqueeze(0).to(cfg.device)
            out = model(img)
            # probability map
            prob = out['binary'].squeeze(0).squeeze(0).detach().cpu().numpy()
            bin_map = (prob > box_thresh).astype(np.uint8)
            contours, _ = cv2.findContours(bin_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            preds: List[List[float]] = []
            for c in contours:
                if cv2.contourArea(c) < 10:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                preds.append([float(x), float(y), float(x + w), float(y + h)])
            gts = sample.get('gt_boxes', [])
            m = detection_metrics(preds, gts, iou_thresh)
            tp += int(m['tp'])
            fp += int(m['fp'])
            fn += int(m['fn'])
    model.train()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def main():
    ap = argparse.ArgumentParser(description='Train text detection (DBNet++/YOLO)')
    ap.add_argument('--config', type=str, required=False, default='configs/det_dbnet.yaml')
    ap.add_argument('--model', type=str, default='dbnet', choices=['dbnet', 'yolo'])
    ap.add_argument('--resume', type=str, default='')
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) if args.config and Path(args.config).exists() else {}
    print('[INFO] Loaded config keys:', list(cfg_raw.keys()))

    if args.model == 'dbnet':
        cfg = build_config(cfg_raw)
        print('[INFO] Starting DBNet++ training')
        train_dbnet(cfg)
    else:
        print('[WARN] YOLO text detection training is not implemented yet in this script.')
        print('       Please use DBNet++ for now.')


if __name__ == '__main__':
    main()
