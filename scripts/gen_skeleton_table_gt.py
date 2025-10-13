#!/usr/bin/env python3
"""
Generate HTML ground truth for table-skeleton images (no text) by detecting
grid lines. Outputs next to each image:
  - <stem>.gt.html (first table)
  - <stem>.tables.json ([{"html": "..."}] list)

Usage:
  python scripts/gen_skeleton_table_gt.py --root data/table/images --limit 0 --overwrite false

Notes:
  - Designed for clean table frames. For noisy scans, adjust thresholds.
  - No cell spanning is inferred; produces a regular grid.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2  # type: ignore
import numpy as np
import json


EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in EXT]


def binarize(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Adaptive threshold for varied backgrounds
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)
    return bw


def detect_lines(bw: np.ndarray, h_scale: float = 0.03, v_scale: float = 0.03) -> Tuple[List[int], List[int]]:
    h, w = bw.shape[:2]
    # Horizontal lines
    hk = max(1, int(w * h_scale))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    # Vertical lines
    vk = max(1, int(h * v_scale))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    # Reduce to 1D projections and pick peaks
    h_proj = np.sum(h_lines > 0, axis=1)
    v_proj = np.sum(v_lines > 0, axis=0)
    y_candidates = np.where(h_proj > (0.3 * np.max(h_proj) if np.max(h_proj) > 0 else 1))[0]
    x_candidates = np.where(v_proj > (0.3 * np.max(v_proj) if np.max(v_proj) > 0 else 1))[0]
    # Cluster close indices to single lines
    def cluster(vals: np.ndarray, gap: int = 3) -> List[int]:
        vals = np.array(sorted(set(int(x) for x in vals.tolist())))
        if vals.size == 0:
            return []
        groups = [[int(vals[0])]]
        for v in vals[1:]:
            if v - groups[-1][-1] <= gap:
                groups[-1].append(int(v))
            else:
                groups.append([int(v)])
        centers = [int(np.mean(g)) for g in groups]
        return centers
    y_lines = cluster(y_candidates, gap=2)
    x_lines = cluster(x_candidates, gap=2)
    # Ensure boundaries included
    if 0 not in y_lines:
        y_lines = [0] + y_lines
    if (h - 1) not in y_lines:
        y_lines = y_lines + [h - 1]
    if 0 not in x_lines:
        x_lines = [0] + x_lines
    if (w - 1) not in x_lines:
        x_lines = x_lines + [w - 1]
    y_lines = sorted(set(y_lines))
    x_lines = sorted(set(x_lines))
    # Filter too-dense duplicates
    def dedup(vals: List[int]) -> List[int]:
        out = []
        for v in vals:
            if not out or abs(v - out[-1]) > 2:
                out.append(v)
        return out
    return dedup(y_lines), dedup(x_lines)


def make_html(y: List[int], x: List[int]) -> str:
    rows = max(0, len(y) - 1)
    cols = max(0, len(x) - 1)
    parts = ["<table>"]
    for _ in range(rows):
        parts.append("  <tr>")
        for _ in range(cols):
            parts.append("    <td></td>")
        parts.append("  </tr>")
    parts.append("</table>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", default="false", choices=["true", "false"])
    args = ap.parse_args()

    root = Path(args.root)
    files = list_images(root)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    wrote = 0
    skipped = 0
    for i, fp in enumerate(files, 1):
        stem = fp.with_suffix("")
        gt_json = stem.with_suffix(".tables.json")
        gt_html = stem.with_suffix(".gt.html")
        if args.overwrite.lower() != "true" and (gt_json.exists() or gt_html.exists()):
            skipped += 1
            continue
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        bw = binarize(img)
        y, x = detect_lines(bw)
        if len(y) < 2 or len(x) < 2:
            continue
        html = make_html(y, x)
        try:
            gt_json.write_text(json.dumps([{ "html": html }], ensure_ascii=False, indent=2), encoding="utf-8")
            gt_html.write_text(html, encoding="utf-8")
            wrote += 1
        except Exception:
            continue
        if i % 50 == 0:
            print(f"[{i}/{len(files)}] wrote={wrote} skipped={skipped}")
    print({"processed": len(files), "wrote": wrote, "skipped": skipped})


if __name__ == "__main__":
    main()

