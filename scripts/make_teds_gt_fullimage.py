#!/usr/bin/env python3
"""
Generate TEDS GT from full-image table assumption.

For each image in --root, call bullseye table recognizer with a single box
covering the full image and write:
  - <stem>.tables.json (array of {html})
  - <stem>.gt.html (first table only)

Use when layout detection misses table boxes or to speed up GT creation.

Usage:
  python scripts/make_teds_gt_fullimage.py --root data/table/images --limit 0 --overwrite false
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

# Ensure repo root is importable (so `from src...` works when run as a script)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in EXT]


def ensure_env():
    os.environ.setdefault("DOCJA_BULLSEYE_LOCAL_DIR", str(Path.cwd() / "bullseye" / "src"))
    os.environ.setdefault("DOCJA_NO_HF", "1")
    os.environ.setdefault("DOCJA_NO_INTERNAL_FALLBACK", "1")


def to_np_rgb(img_path: Path) -> np.ndarray:
    im = Image.open(str(img_path)).convert("RGB")
    return np.array(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", default="false", choices=["true", "false"])
    args = ap.parse_args()

    ensure_env()
    from src.integrations.bullseye_table import BullseyeTableRecognizer
    # Choose device based on availability
    device = 'cuda'
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            device = 'cpu'
    except Exception:
        device = 'cpu'
    tbl = BullseyeTableRecognizer(device=device)

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
        arr = to_np_rgb(fp)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        bbox = [0.0, 0.0, float(w), float(h)]
        try:
            tables = tbl.recognize(arr, boxes=[bbox])
            htmls = [t.get("html", "") for t in tables if isinstance(t, dict) and t.get("html")]
            if not htmls:
                continue
            gt_json.write_text(json.dumps([{ "html": h } for h in htmls], ensure_ascii=False, indent=2), encoding="utf-8")
            gt_html.write_text(htmls[0], encoding="utf-8")
            wrote += 1
        except Exception:
            continue
        if i % 20 == 0:
            print(f"[{i}/{len(files)}] wrote={wrote} skipped={skipped}")

    print({"processed": len(files), "wrote": wrote, "skipped": skipped})


if __name__ == "__main__":
    main()
