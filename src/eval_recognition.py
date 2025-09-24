#!/usr/bin/env python3
"""
Recognition evaluation script.
Computes CER on a list of cropped line images + ground truth JSON mapping.

Improvements:
- Accepts either a config JSON file path via --config, or direct CLI args
  --images-dir/--labels/--model.
- Robust path checks (treat empty path as missing) to avoid '.' handling.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image

from .pipeline.text_recognizer import TextRecognizer
from .utils.metrics import cer as cer_metric


def cer(ref: str, hyp: str) -> float:
    # Levenshtein distance / len(ref)
    try:
        import editdistance
        if not ref:
            return 0.0 if not hyp else 1.0
        return editdistance.eval(ref, hyp) / max(1, len(ref))
    except Exception:
        # Fallback: simple char mismatch ratio
        n = max(len(ref), len(hyp), 1)
        return sum(1 for a, b in zip(ref, hyp) if a != b) / n


def evaluate(config: Dict[str, Any]) -> Dict[str, float]:
    images_dir = Path(config.get('images_dir') or '__MISSING__')
    labels_path = Path(config.get('labels') or '__MISSING__')
    if not images_dir.exists() or not images_dir.is_dir():
        return {'cer': 0.0, 'num_lines': 0}
    if not labels_path.exists() or not labels_path.is_file():
        return {'cer': 0.0, 'num_lines': 0}

    labels = json.loads(labels_path.read_text(encoding='utf-8'))  # {filename: text}
    # Optional vocabulary and weights
    vocab_path = config.get('vocab') or config.get('vocab_path')
    weights_path = config.get('weights') or config.get('weights_path')
    max_len = int(config.get('max_len', config.get('model_max_len', 80)))
    # Fallback to best checkpoint if present
    if not weights_path:
        default_best = Path('weights/rec/abinet_best.pth')
        default_final = Path('weights/rec/abinet.pth')
        if default_best.exists():
            weights_path = str(default_best)
        elif default_final.exists():
            weights_path = str(default_final)
    
    recog = TextRecognizer(
        model_type=config.get('model', 'abinet'),
        device='cuda',
        vocab_path=Path(vocab_path) if vocab_path else None,
        weights_path=Path(weights_path) if weights_path else None,
        max_len=max_len,
    )

    total_cer = 0.0
    n = 0
    for name, gt in labels.items():
        p = images_dir / name
        if not p.exists():
            continue
        img = Image.open(p).convert('RGB')
        img_np = np.array(img)
        pred, _ = recog.recognize(img_np)
        total_cer += cer_metric(gt, pred)
        n += 1
    return {'cer': (total_cer / n) if n > 0 else 0.0, 'num_lines': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=False, help='Path to eval config (JSON)')
    ap.add_argument('--images-dir', required=False, help='Directory containing line images')
    ap.add_argument('--labels', required=False, help='Path to labels.json (filename -> text)')
    ap.add_argument('--model', required=False, default='abinet', help='Recognizer model type')
    ap.add_argument('--vocab', required=False, help='Path to vocabulary file (txt/json)')
    ap.add_argument('--weights', required=False, help='Path to weights checkpoint (.pth)')
    ap.add_argument('--max-len', dest='max_len', type=int, required=False, help='Model max sequence length')
    ap.add_argument('--output', required=False, default='results/recognition_metrics.json')
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    else:
        if args.images_dir and args.labels:
            cfg = {
                'images_dir': args.images_dir,
                'labels': args.labels,
                'model': args.model or 'abinet',
            }
            if args.vocab:
                cfg['vocab'] = args.vocab
            if args.weights:
                cfg['weights'] = args.weights
            if args.max_len:
                cfg['max_len'] = args.max_len

    metrics = evaluate(cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
