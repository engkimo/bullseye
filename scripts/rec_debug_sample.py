#!/usr/bin/env python3
"""
Quickly sample predictions and report recognition stats.

Usage:
  python scripts/rec_debug_sample.py \
    --images-dir data/rec/lines_pre_w480 \
    --labels data/rec/labels.normalized.json \
    --vocab data/charset_from_labels.txt \
    --weights weights/rec/abinet_best.pth \
    --max-len 80 \
    --n 200
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is on sys.path so that `src` is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.text_recognizer import TextRecognizer
from src.utils.metrics import cer as cer_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--vocab', required=False)
    ap.add_argument('--weights', required=False)
    ap.add_argument('--model', default='abinet')
    ap.add_argument('--max-len', type=int, default=80)
    ap.add_argument('-n', '--n', type=int, default=200)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels = json.loads(Path(args.labels).read_text('utf-8'))
    items = [(images_dir / k, v) for k, v in labels.items() if (images_dir / k).exists()]
    random.seed(0)
    items = random.sample(items, min(args.n, len(items)))

    recog = TextRecognizer(
        model_type=args.model,
        device='cuda',
        vocab_path=Path(args.vocab) if args.vocab else None,
        weights_path=Path(args.weights) if args.weights else None,
        max_len=args.max_len,
    )

    empties = 0
    eos_hits = 0
    tot_len = 0
    tot_cer = 0.0
    for p, gt in items:
        img = Image.open(p).convert('RGB')
        pred, _ = recog.recognize(np.array(img))
        tot_len += len(pred)
        if pred == '':
            empties += 1
        if pred.endswith('\u0003') or pred.endswith('<eos>'):
            eos_hits += 1
        tot_cer += cer_metric(gt, pred)

    n = len(items)
    print(json.dumps({
        'n': n,
        'avg_pred_len': (tot_len / max(n, 1)),
        'empty_ratio': (empties / max(n, 1)),
        'cer': (tot_cer / max(n, 1)),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
