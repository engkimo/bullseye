#!/usr/bin/env python3
"""
Layout evaluation script (skeleton).
Computes mAP (COCO-style) given predictions and ground truth.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from .pipeline import DocumentProcessor
from .pipeline.layout_detector import LayoutDetector
from .utils.metrics import bbox_iou


def _ap_at_iou(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]], iou_thresh: float = 0.5) -> float:
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


def evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    images_dir = Path(config.get('images_dir', ''))
    annotations = Path(config.get('annotations', ''))
    run_inference = bool(config.get('run_inference', False))
    if not images_dir.exists() or not annotations.exists():
        return {'map_50_95': 0.0, 'map_50': 0.0, 'num_images': 0}

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    coco = json.loads(annotations.read_text(encoding='utf-8'))
    id_to_file = {img['id']: img['file_name'] for img in coco.get('images', [])}
    cats = {c['id']: c['name'] for c in coco.get('categories', [])}

    gt_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for ann in coco.get('annotations', []):
        img_file = id_to_file.get(ann['image_id'])
        if not img_file:
            continue
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        gt_by_file.setdefault(img_file, []).append({'bbox': [x, y, x + w, y + h], 'class': ann.get('category_id', 0)})

    # predictions via pipeline (now layout_elements exposed)
    proc = None
    if run_inference:
        proc = DocumentProcessor(det_model='dbnet', rec_model='abinet', layout_model='yolo', enable_table=False, enable_reading_order=False, enable_llm=False, device='cuda', weights_dir='weights', lite_mode=True)

    labels = LayoutDetector.LABELS if hasattr(LayoutDetector, 'LABELS') else []
    pred_by_file: Dict[str, List[Dict[str, Any]]] = {}
    if run_inference and proc is not None:
        for p in images:
            res = proc.process(str(p), max_pages=1)
            dets: List[Dict[str, Any]] = []
            if res.pages and res.pages[0].layout_elements:
                for el in res.pages[0].layout_elements:
                    bbox = [float(el['bbox'][0]), float(el['bbox'][1]), float(el['bbox'][2]), float(el['bbox'][3])]
                    cls_name = el.get('label') or el.get('type') or 'paragraph'
                    cls_id = labels.index(cls_name) if cls_name in labels else 0
                    dets.append({'bbox': bbox, 'class': cls_id, 'score': float(el.get('confidence', 0.5))})
            pred_by_file[p.name] = dets

    # Compute AP@0.5 and mAP@[.50:.95]
    maps = []
    maps_50 = []
    for p in images:
        gts = gt_by_file.get(p.name, [])
        preds = pred_by_file.get(p.name, [])
        ap_50 = _ap_at_iou(preds, gts, 0.5)
        maps_50.append(ap_50)
        ap_all = []
        for t in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:
            ap_all.append(_ap_at_iou(preds, gts, t))
        maps.append(sum(ap_all)/len(ap_all) if ap_all else 0.0)

    return {'map_50_95': float(sum(maps)/len(maps)) if maps else 0.0,
            'map_50': float(sum(maps_50)/len(maps_50)) if maps_50 else 0.0,
            'num_images': len(images)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, help='Path to eval config (JSON)')
    ap.add_argument('--output', required=False, default='results/layout_metrics.json')
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
