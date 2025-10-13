#!/usr/bin/env python3
"""
Bootstrap TEDS ground truth by harvesting table HTML from current predictions.

For each image under --images-root, runs the bullseye pipeline (layout+table)
and writes GT files next to the image when missing:
  - <stem>.tables.json  (preferred; [{"html": "..."}, ...])
  - <stem>.gt.html      (first table only; convenience)

Notes
- This produces "silver" GT derived from current model predictions. Use only
  to unblock evaluation or smoke tests; replace with human-labeled GT for
  true benchmarks.

Usage:
  python scripts/make_teds_gt_from_table_preds.py \
    --images-root data/table/images --limit 2000 --overwrite false
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import List, Dict, Any


IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}


def discover_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            files.append(p)
    return files


def ensure_bullseye_env(env: Dict[str, str]) -> Dict[str, str]:
    e = env.copy()
    e["DOCJA_PROVIDER_ALIAS_LABEL"] = "bullseye"
    e["DOCJA_DET_PROVIDER"] = "bullseye"
    e["DOCJA_REC_PROVIDER"] = "bullseye"
    e["DOCJA_LAYOUT_PROVIDER"] = "bullseye"
    e["DOCJA_TABLE_PROVIDER"] = "bullseye"
    e["DOCJA_FORCE_YOMITOKU"] = "1"
    e["DOCJA_NO_INTERNAL_FALLBACK"] = "1"
    e["DOCJA_NO_HF"] = "1"
    e.setdefault("DOCJA_BULLSEYE_LOCAL_DIR", str(Path.cwd() / "bullseye" / "src"))
    e.setdefault("DOCJA_REC_MODEL_ID", "Ryousukee/bullseye-recparseq")
    e.setdefault("DOCJA_LAYOUT_MODEL_ID", "rtdetrv2")
    return e


def run_docja_tables(img: Path, outdir: Path, env: Dict[str, str]) -> Dict[str, Any] | None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.cli", str(img), "--layout", "--table", "-f", "json", "-o", str(outdir)]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[WARN] docja failed on {img.name}: {e}\n")
        try:
            (outdir / "cli_stderr.log").write_text((e.stderr or b"").decode(errors="ignore"), encoding="utf-8")
        except Exception:
            pass
        return None
    # parse JSON/JSONL
    udj = None
    for cand in list(outdir.glob("*.jsonl")) + list(outdir.glob("*.json")):
        try:
            txt = cand.read_text(encoding="utf-8")
            for line in txt.splitlines():
                s = line.strip()
                if not s:
                    continue
                try:
                    udj = json.loads(s)
                    break
                except Exception:
                    continue
            if udj is None:
                udj = json.loads(txt)
            break
        except Exception:
            continue
    return udj


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
    ap.add_argument("--images-root", required=True, type=str)
    ap.add_argument("--limit", default=0, type=int)
    ap.add_argument("--overwrite", default="false", choices=["true", "false"])
    args = ap.parse_args()

    root = Path(args.images_root)
    imgs = discover_images(root)
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]
    env = ensure_bullseye_env(os.environ.copy())

    wrote = 0
    skipped = 0
    for i, img in enumerate(imgs, 1):
        stem = img.with_suffix("")
        gt_json = stem.with_suffix(".tables.json")
        gt_html = stem.with_suffix(".gt.html")
        if args.overwrite.lower() != "true" and (gt_json.exists() or gt_html.exists()):
            skipped += 1
            if i % 25 == 0:
                print(f"[{i}/{len(imgs)}] skipped existing: {img.name}")
            continue
        tmp_out = Path("results/tmp_gt") / img.stem
        udj = run_docja_tables(img, tmp_out, env)
        if not udj:
            continue
        htmls = harvest_html_tables(udj)
        if not htmls:
            continue
        try:
            # prefer JSON list
            arr = [{"html": h} for h in htmls]
            gt_json.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")
            # also first table as .gt.html for convenience
            gt_html.write_text(htmls[0], encoding="utf-8")
            wrote += 1
        except Exception as e:
            sys.stderr.write(f"[WARN] failed to write GT for {img.name}: {e}\n")
        if i % 10 == 0:
            print(f"[{i}/{len(imgs)}] wrote={wrote} skipped={skipped}")

    print({"processed": len(imgs), "wrote": wrote, "skipped": skipped})


if __name__ == "__main__":
    main()

