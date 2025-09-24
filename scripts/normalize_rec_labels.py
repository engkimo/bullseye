#!/usr/bin/env python3
"""Normalize recognition labels file (data/rec/labels.json).

Applies:
 - Unicode NFKC normalization
 - Strip leading/trailing whitespace
 - Collapse internal whitespace
 - Optional max length truncation

Usage:
  python scripts/normalize_rec_labels.py \
    --in data/rec/labels.json \
    --out data/rec/labels.normalized.json \
    --max-len 100
"""
import argparse
import json
import unicodedata
from pathlib import Path


def normalize_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = ' '.join(s.split())
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input labels.json')
    ap.add_argument('--out', dest='out', required=True, help='Output normalized labels.json')
    ap.add_argument('--max-len', type=int, default=0, help='Trim to max length (0 to disable)')
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding='utf-8'))
    out = {}
    for k, v in data.items():
        t = normalize_text(str(v))
        if args.max_len and len(t) > args.max_len:
            t = t[: args.max_len]
        out[k] = t

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {args.out} with {len(out)} entries")


if __name__ == '__main__':
    main()

