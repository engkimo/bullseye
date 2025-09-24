#!/usr/bin/env python3
"""
Generate synthetic Japanese table images and TATR-like annotations.

Output structure (default: data/table):
  data/table/
    images/
      table_000001.png ...
    annotations.json  # {"annotations": [{"image": "table_000001.png", "boxes": [[cx,cy,w,h],...], "labels": [int,...]}]}

Labels (TATR order):
  0: table
  1: table_column
  2: table_row
  3: table_column_header
  4: table_projected_row_header
  5: table_spanning_cell
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont


JP_HEADERS = [
    "品目", "数量", "単価", "金額", "税率", "税額", "小計", "合計", "備考",
    "年月日", "部門", "担当", "注文番号", "請求番号", "摘要",
]

JP_WORDS = [
    "ネジ", "ワッシャー", "ノートPC", "マウス", "キーボード", "ケーブル", "アダプタ", "部品", "検査" ,
    "承認", "発注", "納期", "見積", "請求" ,
]


@dataclass
class GenConfig:
    out_dir: Path
    count: int = 200
    width: int = 1000
    height: int = 1000
    min_rows: int = 5
    max_rows: int = 14
    min_cols: int = 3
    max_cols: int = 8
    margin: int = 40
    line_color: Tuple[int, int, int] = (0, 0, 0)
    text_color: Tuple[int, int, int] = (0, 0, 0)


def _norm_cxcywh(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


def _try_font() -> ImageFont.FreeTypeFont:
    # Prefer a Japanese-capable font; fall back to default
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, 18)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_tables(cfg: GenConfig):
    img_dir = cfg.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = cfg.out_dir / "annotations.json"

    font = _try_font()
    anns: List[Dict] = []

    for i in range(1, cfg.count + 1):
        W, H = cfg.width, cfg.height
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Random grid
        rows = random.randint(cfg.min_rows, cfg.max_rows)
        cols = random.randint(cfg.min_cols, cfg.max_cols)

        # Table outer box (leave margin)
        x1 = cfg.margin
        y1 = cfg.margin
        x2 = W - cfg.margin
        y2 = H - cfg.margin
        draw.rectangle([x1, y1, x2, y2], outline=cfg.line_color, width=2)

        # Column widths and row heights (variable)
        col_xs = [x1]
        remaining_w = x2 - x1
        for c in range(cols - 1):
            split = remaining_w * random.uniform(0.1, 0.4)
            col_xs.append(int(col_xs[-1] + split))
            remaining_w -= split
        col_xs.append(x2)
        col_xs = sorted(col_xs)

        row_ys = [y1]
        remaining_h = y2 - y1
        for r in range(rows - 1):
            split = remaining_h * random.uniform(0.08, 0.3)
            row_ys.append(int(row_ys[-1] + split))
            remaining_h -= split
        row_ys.append(y2)
        row_ys = sorted(row_ys)

        # Draw grid lines
        for cx in col_xs:
            draw.line([(cx, y1), (cx, y2)], fill=cfg.line_color, width=1)
        for ry in row_ys:
            draw.line([(x1, ry), (x2, ry)], fill=cfg.line_color, width=1)

        boxes: List[Tuple[float, float, float, float]] = []
        labels: List[int] = []

        # 0: table (outer)
        boxes.append(_norm_cxcywh(x1, y1, x2, y2, W, H))
        labels.append(0)

        # 1: table_column (vertical stripes)
        for c in range(len(col_xs) - 1):
            cx1 = col_xs[c]
            cx2 = col_xs[c + 1]
            boxes.append(_norm_cxcywh(cx1, y1, cx2, y2, W, H))
            labels.append(1)

        # 2: table_row (horizontal stripes)
        for r in range(len(row_ys) - 1):
            ry1 = row_ys[r]
            ry2 = row_ys[r + 1]
            boxes.append(_norm_cxcywh(x1, ry1, x2, ry2, W, H))
            labels.append(2)

        # 3: table_column_header (first row cells)
        header_r = 0
        for c in range(len(col_xs) - 1):
            cx1 = col_xs[c]
            cx2 = col_xs[c + 1]
            hy1 = row_ys[header_r]
            hy2 = row_ys[header_r + 1]
            # Header text
            text = random.choice(JP_HEADERS)
            draw.text((cx1 + 8, hy1 + 6), text, fill=cfg.text_color, font=font)
            boxes.append(_norm_cxcywh(cx1, hy1, cx2, hy2, W, H))
            labels.append(3)

        # 4: table_projected_row_header（左端列の下数行を行見出しとしてマーク）
        row_hdr_rows = max(1, rows // 4)
        for r in range(1, 1 + row_hdr_rows):
            rx1 = col_xs[0]
            rx2 = col_xs[1]
            ry1 = row_ys[r]
            ry2 = row_ys[r + 1]
            draw.text((rx1 + 8, ry1 + 6), random.choice(JP_WORDS), fill=cfg.text_color, font=font)
            boxes.append(_norm_cxcywh(rx1, ry1, rx2, ry2, W, H))
            labels.append(4)

        # 5: table_spanning_cell（ランダムに1〜2個）
        for _ in range(random.randint(0, 2)):
            c0 = random.randint(0, max(0, (len(col_xs) - 2)))
            c1 = min(len(col_xs) - 2, c0 + random.randint(1, 2))
            r0 = random.randint(1, max(1, (len(row_ys) - 3)))
            r1 = min(len(row_ys) - 2, r0 + random.randint(1, 2))
            sx1 = col_xs[c0]
            sx2 = col_xs[c1 + 1]
            sy1 = row_ys[r0]
            sy2 = row_ys[r1 + 1]
            draw.rectangle([sx1, sy1, sx2, sy2], outline=(128, 128, 128), width=2)
            draw.text((sx1 + 8, sy1 + 6), "結合セル", fill=cfg.text_color, font=font)
            boxes.append(_norm_cxcywh(sx1, sy1, sx2, sy2, W, H))
            labels.append(5)

        name = f"table_{i:06d}.png"
        img.save(str(img_dir / name))
        anns.append({"image": name, "boxes": boxes, "labels": labels})

    ann_path.write_text(json.dumps({"annotations": anns}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen] Generated {len(anns)} tables at {cfg.out_dir}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/table")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()
    cfg = GenConfig(out_dir=Path(args.out), count=args.count, width=args.width, height=args.height)
    generate_tables(cfg)


if __name__ == "__main__":
    main()

