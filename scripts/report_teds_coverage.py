#!/usr/bin/env python3
"""
Report TEDS ground-truth coverage for a directory of inputs.

Counts how many inputs have a matching `<stem>.gt.html`, `<stem>.html`, or `<stem>.tables.json`.

Usage:
  python scripts/report_teds_coverage.py --root data/table/images
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory to scan for inputs")
    args = ap.parse_args()

    root = Path(args.root)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]

    with_gt = 0
    gt_kinds = {"gt_html": 0, "html": 0, "tables_json": 0}
    for p in files:
        stem = p.with_suffix("")
        if (stem.with_suffix(".gt.html")).exists():
            with_gt += 1
            gt_kinds["gt_html"] += 1
            continue
        if (stem.with_suffix(".html")).exists():
            with_gt += 1
            gt_kinds["html"] += 1
            continue
        if (stem.with_suffix(".tables.json")).exists():
            with_gt += 1
            gt_kinds["tables_json"] += 1
            continue

    total = len(files)
    print({
        "root": str(root),
        "total": total,
        "with_gt": with_gt,
        "coverage": round(with_gt / total, 4) if total else 0.0,
        **gt_kinds,
    })


if __name__ == "__main__":
    main()

