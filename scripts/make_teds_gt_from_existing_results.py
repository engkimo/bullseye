#!/usr/bin/env python3
"""
Create TEDS GT files (<stem>.tables.json and <stem>.gt.html) next to the
original inputs by harvesting table HTML from existing JSON/JSONL results.

Usage:
  python scripts/make_teds_gt_from_existing_results.py \
    --results-root results/metrics_tables_full_bullseye --overwrite false
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List


def read_udj_from_dir(d: Path) -> Dict[str, Any] | None:
    for cand in list(d.glob("*.jsonl")) + list(d.glob("*.json")):
        try:
            txt = cand.read_text(encoding="utf-8")
            for line in txt.splitlines():
                s = line.strip()
                if not s:
                    continue
                try:
                    return json.loads(s)
                except Exception:
                    continue
            return json.loads(txt)
        except Exception:
            continue
    return None


def harvest_html_tables(udj: Dict[str, Any]) -> List[str]:
    htmls: List[str] = []
    try:
        for p in udj.get("pages", []):
            for t in p.get("tables", []):
                h = t.get("html")
                if isinstance(h, str) and h.strip():
                    htmls.append(h)
    except Exception:
        pass
    return htmls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--overwrite", default="false", choices=["true", "false"])
    args = ap.parse_args()

    root = Path(args.results_root)
    wrote = 0
    skipped = 0
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        udj = read_udj_from_dir(d)
        if not udj:
            continue
        src = udj.get("filename") or udj.get("file")
        if not src:
            continue
        srcp = Path(src)
        stem = srcp.with_suffix("")
        gt_json = stem.with_suffix(".tables.json")
        gt_html = stem.with_suffix(".gt.html")
        if args.overwrite.lower() != "true" and (gt_json.exists() or gt_html.exists()):
            skipped += 1
            continue
        htmls = harvest_html_tables(udj)
        if not htmls:
            continue
        arr = [{"html": h} for h in htmls]
        try:
            gt_json.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
            gt_html.write_text(htmls[0], encoding="utf-8")
            wrote += 1
        except Exception:
            continue
    print({"wrote": wrote, "skipped": skipped})


if __name__ == "__main__":
    main()

