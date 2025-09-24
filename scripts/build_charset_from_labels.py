#!/usr/bin/env python3
"""Build a charset file from labels.json by character frequency.

Usage:
  python3 scripts/build_charset_from_labels.py \
    --labels data/rec/labels.normalized.json \
    --out data/charset_from_labels.txt \
    --max-chars 8000
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-chars', type=int, default=8000)
    args = ap.parse_args()

    data = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    ctr = Counter()
    seen_space = False
    seen_zwsp = False
    seen_full_space = False
    for _, text in data.items():
        s = str(text)
        ctr.update(list(s))
        if ' ' in s:
            seen_space = True
        if '\u200b' in s:  # zero-width space
            seen_zwsp = True
        if '\u3000' in s:  # full-width space
            seen_full_space = True

    # Exclude special placeholders if present
    for sp in ['<pad>', '<sos>', '<eos>', '<unk>']:
        if sp in ctr:
            del ctr[sp]

    most = [ch for ch, _ in ctr.most_common(args.max_chars)]
    # Ensure common whitespace characters are present if they appeared in labels
    def _ensure(ch: str):
        if ch not in most:
            most.append(ch)
    if seen_space:
        _ensure(' ')
    if seen_zwsp:
        _ensure('\u200b')
    if seen_full_space:
        _ensure('\u3000')

    # Write one character per line
    with open(args.out, 'w', encoding='utf-8') as f:
        for ch in most:
            # keep whitespace characters (space, full-width space, etc.)
            f.write(ch + '\n')
    print(f"Wrote {args.out} with {len(most)} characters")


if __name__ == '__main__':
    main()
