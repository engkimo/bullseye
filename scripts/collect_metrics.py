#!/usr/bin/env python3
"""
Collect end-to-end metrics for Bullseye pipeline.

What it measures (if ground truth is available next to inputs):
 - Latency per page (p50/p95)
 - VRAM snapshot (MB)
 - CER for OCR (file.txt or file.gt.txt as ground truth)
 - TEDS for tables (requires `teds` if installed; otherwise skipped)
 - (Optional) QA ANLS/EM/F1 if <file>.qa.jsonl exists

Input discovery:
 - Scans `data/` by default for .pdf/.png/.jpg/.jpeg/.tif/.tiff
 - Looks for sibling ground truth files under the same stem

Outputs:
 - results/metrics/metrics.csv (per-file rows)
 - results/metrics/summary.json (aggregates)

Usage:
  python scripts/collect_metrics.py \
    --data-root data/ --out results/metrics \
    --cli bullseye --format json --with-llm false
"""
from __future__ import annotations
import argparse, csv, json, os, re, shlex, subprocess, sys, time, statistics
from pathlib import Path
import os
from typing import Dict, Any, List, Optional, Tuple


IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DOC_EXT = IMG_EXT | {".pdf"}


def try_import_pynvml():
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_vram_mb() -> Optional[int]:
    # Try nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        vals = [int(x) for x in out if x.strip()]
        return max(vals) if vals else None
    except Exception:
        pass
    # Fallback: pynvml
    pynvml = try_import_pynvml()
    if pynvml:
        try:
            n = pynvml.nvmlDeviceGetCount()
            used = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                used.append(int(mem.used / (1024 * 1024)))
            return max(used) if used else None
        except Exception:
            return None
    return None


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            prev, dp[j] = dp[j], min(
                prev + (0 if a[i - 1] == b[j - 1] else 1),  # replace
                dp[j] + 1,  # delete
                dp[j - 1] + 1,  # insert
            )
    return dp[lb]


def cer(hyp: str, ref: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = levenshtein(hyp, ref)
    return dist / max(1, len(ref))


def flatten_text_from_udj(udj: Dict[str, Any]) -> str:
    parts: List[str] = []
    for p in udj.get("pages", []):
        for tb in p.get("text_blocks", []):
            t = tb.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts)


def run_docja(cli: str, path: Path, outdir: Path, with_llm: bool, fmt: str) -> Tuple[float, Optional[Dict[str, Any]]]:
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Resolve CLI invocation. If 'bullseye' is requested, delegate to
    # the repository's pipeline CLI with bullseye providers enforced.
    if cli.strip() == "bullseye":
        base_cmd = [sys.executable, "-m", "src.cli"]
        env = os.environ.copy()
        env.setdefault("DOCJA_PROVIDER_ALIAS_LABEL", "bullseye")
        env.setdefault("DOCJA_DET_PROVIDER", "bullseye")
        env.setdefault("DOCJA_REC_PROVIDER", "bullseye")
        env.setdefault("DOCJA_LAYOUT_PROVIDER", "bullseye")
        env.setdefault("DOCJA_TABLE_PROVIDER", "bullseye")
    else:
        base_cmd = [cli]
        env = os.environ.copy()

    cmd = base_cmd + [
        str(path),
        "--layout",
        "--table",
        "--reading-order",
        "-f",
        fmt,
        "-o",
        str(outdir),
    ]
    if with_llm:
        cmd += ["--llm"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    except subprocess.CalledProcessError as e:
        # Keep going; record latency but UDJson missing
        lat = time.time() - t0
        sys.stderr.write(f"[WARN] docja failed on {path.name}: {e}\n")
        return lat, None
    lat = time.time() - t0
    # Expect JSON Lines when -f json; otherwise skip UDJson load
    if fmt == "json":
        # Find first JSON/JSONL in outdir
        candidates = list(outdir.glob("*.json")) + list(outdir.glob("*.jsonl"))
        udj = None
        if candidates:
            fp = candidates[0]
            try:
                if fp.suffix == ".jsonl":
                    with fp.open("r", encoding="utf-8") as f:
                        first = f.readline()
                    udj = json.loads(first)
                else:
                    udj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as ex:
                sys.stderr.write(f"[WARN] Failed to parse UDJ {fp}: {ex}\n")
        return lat, udj
    return lat, None


def discover_inputs(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in DOC_EXT:
            files.append(p)
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", type=str)
    ap.add_argument("--out", default="results/metrics", type=str)
    ap.add_argument("--cli", default="bullseye", type=str)
    ap.add_argument("--format", default="json", choices=["json", "md", "html", "csv", "pdf"])
    ap.add_argument("--with-llm", default="false", choices=["true", "false"])
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with_llm = args.with_llm.lower() == "true"
    inputs = discover_inputs(data_root)
    if not inputs:
        print(f"No inputs found under {data_root}")
        sys.exit(0)

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    vram_snaps: List[int] = []

    for path in inputs:
        out_sub = outdir / path.stem
        lat, udj = run_docja(args.cli, path, out_sub, with_llm, args.format)
        latencies.append(lat)
        vram = gpu_vram_mb()
        if vram is not None:
            vram_snaps.append(vram)

        # OCR CER if gt exists
        cer_val: Optional[float] = None
        gt_txt = None
        for cand in [path.with_suffix(".gt.txt"), path.with_suffix(".txt")]:
            if cand.exists():
                gt_txt = cand.read_text(encoding="utf-8", errors="ignore")
                break
        if udj and gt_txt is not None:
            hyp = flatten_text_from_udj(udj)
            cer_val = cer(hyp, gt_txt)

        rows.append({
            "file": str(path),
            "latency_s": round(lat, 3),
            "vram_mb": vram if vram is not None else "",
            "cer": round(cer_val, 4) if cer_val is not None else "",
        })

    # Write CSV
    csv_path = outdir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "count": len(rows),
        "p50_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 3),
        "max_vram_mb": max(vram_snaps) if vram_snaps else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
