#!/usr/bin/env python3
"""
Table evaluation script (skeleton).
Computes TEDS given predicted HTML and ground truth HTML for tables.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from .utils.metrics import teds_like_score

try:
    from teds import TEDS  # type: ignore
    _HAS_TEDS = True
except Exception:
    _HAS_TEDS = False


def evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    pairs = config.get('pairs', [])  # list of {pred_html, gt_html}
    if not pairs:
        return {'teds': 0.0, 'num_tables': 0}
    scores: List[float] = []
    use_teds = bool(config.get('use_teds', False)) and _HAS_TEDS
    teds = TEDS(n_jobs=1) if use_teds else None
    for pair in pairs:
        pred_p = Path(pair['pred_html'])
        gt_p = Path(pair['gt_html'])
        if not pred_p.exists() or not gt_p.exists():
            continue
        pred = pred_p.read_text(encoding='utf-8')
        gt = gt_p.read_text(encoding='utf-8')
        if use_teds and teds is not None:
            try:
                scores.append(float(teds.evaluate(gt, pred)))
                continue
            except Exception:
                pass
        scores.append(teds_like_score(gt, pred, structure_weight=float(config.get('structure_weight', 0.5))))
    teds = sum(scores) / len(scores) if scores else 0.0
    return {'teds': teds, 'num_tables': len(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, help='Path to eval config (JSON)')
    ap.add_argument('--output', required=False, default='results/table_metrics.json')
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
