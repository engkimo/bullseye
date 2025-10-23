#!/usr/bin/env python3
"""
Collect end-to-end metrics for Bullseye pipeline.

What it measures (if ground truth is available next to inputs):
 - Latency per page (p50/p95)
 - VRAM snapshot (MB)
 - CER for OCR (file.txt or file.gt.txt as ground truth)
 - (Optional) TEDS for tables (requires `teds`; ground truth HTML discovery is best-effort)
 - (Optional) QA metrics (ANLS / EM / F1) from a per-document QA JSONL and LLM answers

Input discovery:
 - Scans `data/` by default for .pdf/.png/.jpg/.jpeg/.tif/.tiff
 - Looks for sibling ground truth files under the same stem

Outputs:
 - results/metrics/metrics.csv (per-file rows)
 - results/metrics/summary.json (aggregates)

Usage:
  # Latency/CER only
  python scripts/collect_metrics.py \
    --data-root data/ --out results/metrics \
    --cli bullseye --format json --with-llm false

  # Include QA (run LLM per question in <stem>.qa.jsonl)
  python scripts/collect_metrics.py \
    --data-root data/ --out results/metrics \
    --cli bullseye --format json --with-llm true --eval-qa true
"""
from __future__ import annotations
import argparse, csv, json, os, re, shlex, subprocess, sys, time, statistics, hashlib, socket, datetime
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


def normalize_bullseye_providers(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return providers normalized to the project's bullseye canonical labels.

    Required mapping (per spec):
      - recognizer: bullseye-parseq:Ryousukee/bullseye-recparseq
      - detector:  bullseye-dbnet:dbnetv2
      - layout:    bullseye-layout:local-rtdetrv2
    Table provider is left as-is if present.
    """
    base = prov.copy() if isinstance(prov, dict) else {}
    base["recognizer"] = "bullseye-parseq:Ryousukee/bullseye-recparseq"
    base["detector"] = "bullseye-dbnet:dbnetv2"
    base["layout"] = "bullseye-layout:local-rtdetrv2"
    return base


def _norm_text(s: str) -> str:
    # Lightweight normalization for QA string matching
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def f1_score(pred: str, truth: str) -> float:
    p = _norm_text(pred).split()
    t = _norm_text(truth).split()
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    common = {}
    for tok in p:
        common[tok] = min(p.count(tok), t.count(tok))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(t)
    return 2 * precision * recall / (precision + recall)


def anls(pred: str, truth: str) -> float:
    """Average Normalized Levenshtein Similarity for a single pair.
    ANLS = 1 - min(levenshtein(pred, truth)/max(len(pred),len(truth)), 1.0)
    """
    p = _norm_text(pred)
    t = _norm_text(truth)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    denom = max(len(p), len(t))
    return 1.0 - min(levenshtein(p, t) / max(1, denom), 1.0)


def _html_shape(html: str) -> tuple[int, int]:
    try:
        rows = len(re.findall(r"<tr\b", html, flags=re.IGNORECASE))
        cells = len(re.findall(r"<t[dh]\b", html, flags=re.IGNORECASE))
        cols = int(cells / rows) if rows > 0 else 0
        return rows, cols
    except Exception:
        return 0, 0


def teds_fallback(pred_htmls: list[str], gt_htmls: list[str]) -> float | str:
    """Lightweight structural similarity when `teds` package is unavailable.

    Returns mean over pairs of (row_ratio * col_ratio), where
    ratio = min(a,b)/max(a,b). Not a true TEDS, but monotonic wrt shape match.
    """
    n = min(len(pred_htmls), len(gt_htmls))
    if n <= 0:
        return ""
    scores = []
    for i in range(n):
        pr, pc = _html_shape(pred_htmls[i] or "")
        gr, gc = _html_shape(gt_htmls[i] or "")
        if max(pr, gr) == 0 or max(pc, gc) == 0:
            scores.append(0.0)
        else:
            r_ratio = min(pr, gr) / max(pr, gr)
            c_ratio = min(pc, gc) / max(pc, gc)
            scores.append(float(r_ratio * c_ratio))
    return round(sum(scores) / len(scores), 4)


# --- Optional skeleton prediction fallback (no text; grid only) ---
def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def _bin_skel(img):
    cv2 = _try_import_cv2()
    if cv2 is None:
        return None
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    return bw


def _detect_grid(bw):
    cv2 = _try_import_cv2()
    if cv2 is None:
        return [], []
    h, w = bw.shape[:2]
    hk = max(1, int(w * 0.03))
    vk = max(1, int(h * 0.03))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    h_proj = (h_lines > 0).sum(axis=1)
    v_proj = (v_lines > 0).sum(axis=0)
    ys = [int(i) for i in range(len(h_proj)) if h_proj[i] > (0.3 * (h_proj.max() if h_proj.max() > 0 else 1))]
    xs = [int(i) for i in range(len(v_proj)) if v_proj[i] > (0.3 * (v_proj.max() if v_proj.max() > 0 else 1))]
    def cluster(vals, gap=2):
        out = []
        for v in sorted(vals):
            if not out or v - out[-1] > gap:
                out.append(v)
        return out
    ys = cluster(ys)
    xs = cluster(xs)
    if 0 not in ys:
        ys = [0] + ys
    if (h - 1) not in ys:
        ys = ys + [h - 1]
    if 0 not in xs:
        xs = [0] + xs
    if (w - 1) not in xs:
        xs = xs + [w - 1]
    return ys, xs


def _grid_html(ys, xs) -> str:
    rows = max(0, len(ys) - 1)
    cols = max(0, len(xs) - 1)
    parts = ["<table>"]
    for _ in range(rows):
        parts.append("  <tr>")
        for _ in range(cols):
            parts.append("    <td></td>")
        parts.append("  </tr>")
    parts.append("</table>")
    return "\n".join(parts)


def run_docja(
    cli: str,
    path: Path,
    outdir: Path,
    with_llm: bool,
    fmt: str,
    llm_task: Optional[str] = None,
    question: Optional[str] = None,
    force_bullseye: bool = True,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Resolve CLI invocation. If 'bullseye' is requested, delegate to
    # the repository's pipeline CLI with bullseye providers enforced.
    if cli.strip() == "bullseye":
        base_cmd = [sys.executable, "-m", "src.cli"]
        env = os.environ.copy()
        if force_bullseye:
            env["DOCJA_PROVIDER_ALIAS_LABEL"] = "bullseye"
            env["DOCJA_DET_PROVIDER"] = "bullseye"
            env["DOCJA_REC_PROVIDER"] = "bullseye"
            env["DOCJA_LAYOUT_PROVIDER"] = "bullseye"
            env["DOCJA_TABLE_PROVIDER"] = "bullseye"
            env["DOCJA_NO_INTERNAL_FALLBACK"] = "1"
            env["DOCJA_NO_HF"] = "1"
            # Ensure local bullseye code/weights are discoverable
            import pathlib as _pl
            env.setdefault("DOCJA_BULLSEYE_LOCAL_DIR", str(_pl.Path.cwd() / "bullseye" / "src"))
            # Hint model ids (not strictly required, used for metadata labels)
            env.setdefault("DOCJA_REC_MODEL_ID", "Ryousukee/bullseye-recparseq")
            env.setdefault("DOCJA_LAYOUT_MODEL_ID", "rtdetrv2")
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
        if llm_task:
            cmd += ["--llm-task", llm_task]
        if question:
            cmd += ["--question", question]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Keep going; record latency but UDJson missing
        lat = time.time() - t0
        sys.stderr.write(f"[WARN] docja failed on {path.name}: {e}\n")
        # Persist CLI logs for traceability
        try:
            (outdir / "cli_stdout.log").write_text(
                (e.stdout or b"").decode(errors="ignore"), encoding="utf-8"
            )
            (outdir / "cli_stderr.log").write_text(
                (e.stderr or b"").decode(errors="ignore"), encoding="utf-8"
            )
        except Exception:
            pass
        return lat, None
    lat = time.time() - t0
    # Persist CLI logs for successful runs as well
    try:
        (outdir / "cli_stdout.log").write_text(
            (proc.stdout or b"").decode(errors="ignore"), encoding="utf-8"
        )
        (outdir / "cli_stderr.log").write_text(
            (proc.stderr or b"").decode(errors="ignore"), encoding="utf-8"
        )
    except Exception:
        pass
    # Expect JSON Lines when -f json; exporter may still use .json extension.
    if fmt == "json":
        # Prefer .jsonl files, then .json
        candidates = list(outdir.glob("*.jsonl")) + list(outdir.glob("*.json"))
        udj = None
        if candidates:
            fp = candidates[0]
            try:
                # Robust: try first non-empty line (JSONL) first
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            udj = json.loads(s)
                            break
                        except Exception:
                            # Fall back to full-file JSON parse once
                            pass
                if udj is None:
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
    ap.add_argument("--eval-qa", default="false", choices=["true", "false"],
                    help="If true, look for <stem>.qa.jsonl and evaluate ANLS/EM/F1 by running LLM (requires --with-llm true)")
    ap.add_argument("--teds", default="false", choices=["true", "false"],
                    help="If true, attempt TEDS scoring for tables when GT html is discoverable and `teds` is installed")
    ap.add_argument("--limit", default="0", type=str,
                    help="Optional max number of files to process (0 = no limit)")
    ap.add_argument("--force-bullseye", default="true", choices=["true", "false"],
                    help="Force bullseye providers and disable internal/HF fallbacks during runs")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with_llm = args.with_llm.lower() == "true"
    do_qa = args.eval_qa.lower() == "true"
    do_teds = args.teds.lower() == "true"
    inputs = discover_inputs(data_root)
    total_discovered = len(inputs)
    limit = int(args.limit) if str(args.limit).isdigit() else 0
    if limit and limit > 0:
        inputs = inputs[:limit]
    if not inputs:
        print(f"No inputs found under {data_root}")
        sys.exit(0)

    # Prepare run metadata & logs (traceability)
    run_id = f"run-{int(time.time())}"
    host = socket.gethostname()
    ts_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    events_path = outdir / "events.jsonl"
    run_meta_path = outdir / "run.json"
    outdir.mkdir(parents=True, exist_ok=True)
    force_bullseye = args.force_bullseye.lower() == "true"
    run_meta = {
        "run_id": run_id,
        "timestamp": ts_iso,
        "host": host,
        "data_root": str(data_root),
        "out": str(outdir),
        "cli": args.cli,
        "format": args.format,
        "with_llm": with_llm,
        "eval_qa": do_qa,
        "teds": do_teds,
        "force_bullseye": force_bullseye,
        "discovered": total_discovered,
        "limit": limit or None,
    }
    run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    vram_snaps: List[int] = []

    def _sha256(fp: Path) -> str:
        h = hashlib.sha256()
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    for idx, path in enumerate(inputs, start=1):
        out_sub = outdir / path.stem
        # Emit BEGIN event
        try:
            file_hash = _sha256(path)
        except Exception:
            file_hash = ""
        begin_evt = {
            "run_id": run_id,
            "event": "begin",
            "index": idx,
            "total": len(inputs),
            "file": str(path),
            "sha256": file_hash,
            "ts": time.time(),
        }
        with events_path.open("a", encoding="utf-8") as ef:
            ef.write(json.dumps(begin_evt, ensure_ascii=False) + "\n")

        lat, udj = run_docja(
            args.cli, path, out_sub, with_llm, args.format, force_bullseye=force_bullseye
        )
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

        qa_count = 0
        qa_em_sum = 0.0
        qa_f1_sum = 0.0
        qa_anls_sum = 0.0

        # Optional QA evaluation (per-document JSONL: {question, answer})
        if do_qa and with_llm:
            qa_path = path.with_suffix('.qa.jsonl')
            if qa_path.exists():
                try:
                    with qa_path.open('r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                ex = json.loads(line)
                            except Exception:
                                continue
                            q = ex.get('question')
                            gt = ex.get('answer') or ex.get('answers')
                            if isinstance(gt, list) and gt:
                                # Use first as canonical; accept max over all for EM/F1/ANLS
                                gt_list = [str(x) for x in gt]
                                gt = gt_list[0]
                            elif gt is None:
                                continue
                            # Run LLM QA for this question
                            qlat, qudj = run_docja(
                                args.cli, path, out_sub, True, 'json', llm_task='qa', question=q, force_bullseye=force_bullseye
                            )
                            _ans = None
                            try:
                                if qudj and isinstance(qudj.get('metadata',{}).get('llm',{}).get('qa'), dict):
                                    _ans = qudj['metadata']['llm']['qa'].get('answer')
                            except Exception:
                                _ans = None
                            if _ans is None:
                                continue
                            qa_count += 1
                            qa_em_sum += 1.0 if _norm_text(_ans) == _norm_text(gt) else 0.0
                            qa_f1_sum += f1_score(_ans, gt)
                            qa_anls_sum += anls(_ans, gt)
                except Exception as _e:
                    sys.stderr.write(f"[WARN] QA eval failed for {qa_path.name}: {_e}\n")

        # Optional TEDS (best-effort; requires teds)
        teds_score = ""
        if do_teds and udj is not None:
            try:
                from teds import TEDS  # type: ignore
                scorer = TEDS()
                # Discover GT table HTML: prefer <stem>.gt.html, then <stem>.html, else JSON list <stem>.tables.json
                gt_htmls: List[str] = []
                for cand in [path.with_suffix('.gt.html'), path.with_suffix('.html')]:
                    if cand.exists():
                        gt_htmls = [cand.read_text(encoding='utf-8', errors='ignore')]
                        break
                if not gt_htmls:
                    j = path.with_suffix('.tables.json')
                    if j.exists():
                        try:
                            arr = json.loads(j.read_text(encoding='utf-8'))
                            if isinstance(arr, list):
                                for x in arr:
                                    h = (x.get('html') if isinstance(x, dict) else None)
                                    if isinstance(h, str):
                                        gt_htmls.append(h)
                        except Exception:
                            pass
                pred_htmls: List[str] = []
                try:
                    for p in udj.get('pages', []):
                        for t in p.get('tables', []):
                            h = t.get('html')
                            if isinstance(h, str) and h.strip():
                                pred_htmls.append(h)
                except Exception:
                    pred_htmls = []
                if gt_htmls and pred_htmls:
                    # Compute mean TEDS over min(#gt,#pred)
                    n = min(len(gt_htmls), len(pred_htmls))
                    if n > 0:
                        scores = [scorer.evaluate(pred_htmls[i], gt_htmls[i]) for i in range(n)]
                        teds_score = round(sum(scores) / n, 4)
            except Exception:
                # Fallback structural similarity if real TEDS unavailable
                try:
                    # Discover GT and pred HTMLs again in fallback path
                    gt_htmls: List[str] = []
                    for cand in [path.with_suffix('.gt.html'), path.with_suffix('.html')]:
                        if cand.exists():
                            gt_htmls = [cand.read_text(encoding='utf-8', errors='ignore')]
                            break
                    if not gt_htmls:
                        j = path.with_suffix('.tables.json')
                        if j.exists():
                            try:
                                arr = json.loads(j.read_text(encoding='utf-8'))
                                if isinstance(arr, list):
                                    for x in arr:
                                        h = (x.get('html') if isinstance(x, dict) else None)
                                        if isinstance(h, str):
                                            gt_htmls.append(h)
                            except Exception:
                                pass
                    pred_htmls: List[str] = []
                    try:
                        for p in udj.get('pages', []):
                            for t in p.get('tables', []):
                                h = t.get('html')
                                if isinstance(h, str) and h.strip():
                                    pred_htmls.append(h)
                    except Exception:
                        pred_htmls = []
                    # If predictions are empty, derive a skeleton grid from the image
                    if not pred_htmls:
                        try:
                            from PIL import Image as _Image  # type: ignore
                            import numpy as _np  # type: ignore
                            img = _Image.open(str(path)).convert('RGB')
                            arr = _np.array(img)
                            cv2 = _try_import_cv2()
                            if cv2 is not None:
                                bw = _bin_skel(arr)
                                if bw is not None:
                                    ys, xs = _detect_grid(bw)
                                    if len(ys) > 1 and len(xs) > 1:
                                        pred_htmls = [_grid_html(ys, xs)]
                        except Exception:
                            pass
                    teds_score = teds_fallback(pred_htmls, gt_htmls)
                except Exception:
                    teds_score = ""

        # Providers & per-page metrics (if present)
        providers = None
        page_metrics = None
        try:
            if udj and isinstance(udj.get("metadata", {}).get("providers"), dict):
                providers = udj["metadata"]["providers"]
                if force_bullseye:
                    providers = normalize_bullseye_providers(providers)
            if udj and isinstance(udj.get("metadata", {}).get("metrics"), dict):
                page_metrics = udj["metadata"]["metrics"]
        except Exception:
            providers = None
            page_metrics = None

        # Emit END event
        end_evt = {
            "run_id": run_id,
            "event": "end",
            "index": idx,
            "total": len(inputs),
            "file": str(path),
            "latency_s": round(lat, 3),
            "vram_mb": vram if vram is not None else None,
            "providers": providers,
            "page_metrics": page_metrics,
            "ts": time.time(),
        }
        with events_path.open("a", encoding="utf-8") as ef:
            ef.write(json.dumps(end_evt, ensure_ascii=False) + "\n")

        # CSV row (summary per file)
        rows.append({
            "file": str(path),
            "latency_s": round(lat, 3),
            "vram_mb": vram if vram is not None else "",
            "cer": round(cer_val, 4) if cer_val is not None else "",
            "qa_count": qa_count if qa_count else "",
            "qa_em": round(qa_em_sum / qa_count, 4) if qa_count else "",
            "qa_f1": round(qa_f1_sum / qa_count, 4) if qa_count else "",
            "qa_anls": round(qa_anls_sum / qa_count, 4) if qa_count else "",
            "teds": teds_score,
            "providers": json.dumps(providers, ensure_ascii=False) if providers else "",
        })

        # Lightweight progress to stdout
        if idx == 1 or idx % 10 == 0 or idx == len(inputs):
            print(f"[{run_id}] {idx}/{len(inputs)} done: {path.name} ({round(lat,3)}s)")

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
