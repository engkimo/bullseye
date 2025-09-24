#!/usr/bin/env python3
"""
Generate line images + labels.json from a large text corpus (wiki/aozora/etc.).

Usage:
  python scripts/gen_rec_lines_from_txt.py \
    --input data/corpus/wiki_extracted \
    --out data/rec \
    --total 50000 \
    --max-len 32 \
    --vertical-ratio 0.3

Notes:
- Accepts either a directory (will scan files, including wikiextractor outputs) or a single text file.
- Uses reportlab to render text to a temporary PDF and pypdfium2 to rasterize PNGs.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List
import unicodedata

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import pypdfium2 as pdfium


def iter_lines_from_path(path: Path) -> Iterable[str]:
    if path.is_file():
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = unicodedata.normalize('NFKC', line).strip()
                if line:
                    yield line
        return
    # directory: scan files (wikiextractor files are jsonl-like)
    for p in path.rglob('*'):
        if not p.is_file():
            continue
        with open(p, encoding='utf-8', errors='ignore') as f:
            for raw in f:
                s = raw
                # try json access to "text"
                if raw.lstrip().startswith('{') and '"text"' in raw:
                    try:
                        obj = json.loads(raw)
                        s = obj.get('text', '')
                    except Exception:
                        s = raw
                s = unicodedata.normalize('NFKC', s).strip()
                if s:
                    yield s


def pick_sentences(lines: Iterable[str], total: int, max_len: int) -> List[str]:
    # Reservoir sampling of trimmed sentences up to max_len
    pool: List[str] = []
    for line in lines:
        # take a short fragment to fit target recognizer width
        if len(line) > max_len:
            # random window inside line
            start = random.randint(0, max(0, len(line) - max_len))
            frag = line[start:start + max_len]
        else:
            frag = line
        if not frag or frag.isspace():
            continue
        if len(pool) < total:
            pool.append(frag)
        else:
            j = random.randint(0, len(pool) - 1)
            if random.random() < 0.001:  # occasional replacement
                pool[j] = frag
    # If pool smaller than total, just duplicate randomly
    while len(pool) < total:
        pool.append(random.choice(pool) if pool else 'サンプル')
    random.shuffle(pool)
    return pool[:total]


def render_lines_to_png(lines: List[str], out_dir: Path, vertical_ratio: float = 0.0) -> None:
    out_dir = Path(out_dir)
    img_dir = out_dir / 'lines'
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / 'labels.json'

    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))

    tmp_pdf = out_dir / '_lines_from_corpus.pdf'
    c = canvas.Canvas(str(tmp_pdf), pagesize=A4)
    width, height = A4

    labels = {}
    for i, text in enumerate(lines):
        is_vertical = (random.random() < vertical_ratio)
        c.setFont('HeiseiMin-W3', 16)
        if is_vertical:
            # simple vertical rendering: rotate canvas and draw
            c.saveState()
            c.translate(60, height / 2)
            c.rotate(90)
            c.drawString(0, 0, text)
            c.restoreState()
        else:
            c.drawString(60, height / 2, text)
        c.showPage()
        name = f'line_{i:05d}.png'
        labels[name] = text

    c.save()

    # Rasterize pages
    doc = pdfium.PdfDocument(str(tmp_pdf))
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=2.0)
        pil = bitmap.to_pil().convert('RGB')
        pil.save(str(img_dir / f'line_{i:05d}.png'))

    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        tmp_pdf.unlink()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to corpus file or directory (wikiextractor dir ok)')
    ap.add_argument('--out', default='data/rec', help='Output dir (will write lines/ and labels.json)')
    ap.add_argument('--total', type=int, default=50000)
    ap.add_argument('--max-len', type=int, default=32)
    ap.add_argument('--vertical-ratio', type=float, default=0.0)
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out)
    lines = pick_sentences(iter_lines_from_path(src), args.total, args.max_len)
    render_lines_to_png(lines, out_dir, vertical_ratio=args.vertical_ratio)
    print(f'[gen] Generated {len(lines)} lines at {out_dir}/lines with labels {out_dir}/labels.json')


if __name__ == '__main__':
    main()

