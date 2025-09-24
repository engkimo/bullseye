#!/usr/bin/env python3
"""
Detection evaluation script (skeleton).
Computes precision/recall/F1/H-mean on a provided dataset in COCO-like annotations.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from .pipeline import DocumentProcessor
from .utils.metrics import detection_metrics
from PIL import Image
import os

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False


def _poly_to_bbox(poly: List[List[float]]) -> List[float]:
    arr = np.array(poly, dtype=np.float32)
    x1, y1 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x2, y2 = float(arr[:, 0].max()), float(arr[:, 1].max())
    return [x1, y1, x2, y2]


def _run_inference(images: List[Path]) -> Dict[str, List[List[float]]]:
    proc = DocumentProcessor(det_model='dbnet', rec_model='abinet', layout_model=None, enable_table=False, enable_reading_order=False, enable_llm=False, device='cuda', weights_dir='weights', lite_mode=True)
    out: Dict[str, List[List[float]]] = {}
    for p in images:
        res = proc.process(str(p), max_pages=1)
        boxes: List[List[float]] = []
        if res.pages:
            for b in res.pages[0].text_blocks:
                boxes.append([float(b.bbox[0]), float(b.bbox[1]), float(b.bbox[2]), float(b.bbox[3])])
        out[p.name] = boxes
    return out


def evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    images_dir = Path(config.get('images_dir', ''))
    annotations = Path(config.get('annotations', ''))
    predictions = Path(config.get('predictions', '')) if config.get('predictions') else None
    run_inference = bool(config.get('run_inference', False))
    iou_thresh = float(config.get('iou_threshold', 0.5))
    use_coco_eval = bool(config.get('use_coco_eval', False)) and _HAS_COCO

    if not images_dir.exists() or not annotations.exists():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'hmean': 0.0, 'num_images': 0}

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    coco = json.loads(annotations.read_text(encoding='utf-8'))
    id_to_file = {img['id']: img['file_name'] for img in coco.get('images', [])}
    gt_by_file: Dict[str, List[List[float]]] = {}
    for ann in coco.get('annotations', []):
        img_file = id_to_file.get(ann['image_id'])
        if not img_file:
            continue
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        box = [x, y, x + w, y + h]
        gt_by_file.setdefault(img_file, []).append(box)

    if predictions and predictions.exists():
        # if COCO eval requested, short-circuit to COCOeval path
        if use_coco_eval:
            coco_gt = COCO(str(annotations))
            coco_dt = coco_gt.loadRes(str(predictions))
            coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats  # [AP, AP50, AP75, ...]
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'hmean': 0.0,
                'num_images': len(images),
                'coco_AP': float(stats[0]),
                'coco_AP50': float(stats[1]),
                'coco_AP75': float(stats[2]),
            }
        pred_data = json.loads(predictions.read_text(encoding='utf-8'))
    elif run_inference:
        pred_data = _run_inference(images)
    else:
        pred_data = {}

    agg_tp = agg_fp = agg_fn = 0
    count = 0
    for p in images:
        gts = gt_by_file.get(p.name, [])
        preds = pred_data.get(p.name, [])
        m = detection_metrics(preds, gts, iou_thresh)
        agg_tp += m['tp']
        agg_fp += m['fp']
        agg_fn += m['fn']
        count += 1

    precision = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
    recall = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hmean': f1,
        'num_images': count,
        'tp': agg_tp,
        'fp': agg_fp,
        'fn': agg_fn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, help='Path to eval config (JSON)')
    ap.add_argument('--output', required=False, default='results/detection_metrics.json')
    args = ap.parse_args()

    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))

    metrics = evaluate(cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
