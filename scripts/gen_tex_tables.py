#!/usr/bin/env python3
"""
Generate TeX tables from CSV baselines and local metrics.

Inputs (default locations):
 - docs/paper/baselines/docqa.csv
 - docs/paper/baselines/layout.csv
 - docs/paper/baselines/tables.csv
 - results/metrics/metrics.csv (optional; aggregates QA/TEDS)
 - results/layout_metrics.json (optional; mAP values from src/eval_layout.py)

Outputs:
 - docs/paper/tables/docqa.tex
 - docs/paper/tables/layout.tex
 - docs/paper/tables/tables.tex
 - TBF logs (what remains "tbf" and why):
   - docs/paper/tables/tbf.log (human-readable)
   - results/metrics/tbf_log.jsonl (machine-readable)
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import json
from datetime import datetime, timezone


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_tex_table(path: Path, caption: str, header: list[str], rows: list[list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n  \\centering\n  \\small\n")
        f.write(f"  \\caption{{{caption}}}\n")
        f.write("  \\begin{tabular}{" + ("l" + "c" * (len(header) - 1)) + "}\n")
        f.write("    \\toprule\n")
        f.write("    " + " & ".join(header) + " \\\\ \n")
        f.write("    \\midrule\n")
        for r in rows:
            f.write("    " + " & ".join(r) + " \\\\ \n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-root", default="docs/paper/baselines", type=str)
    ap.add_argument("--metrics", default="results/metrics/metrics.csv", type=str)
    ap.add_argument("--layout-metrics", default="results/layout_metrics.json", type=str)
    ap.add_argument("--tbf-log", default="docs/paper/tables/tbf.log", type=str)
    ap.add_argument("--tbf-log-jsonl", default="results/metrics/tbf_log.jsonl", type=str)
    ap.add_argument("--out-root", default="docs/paper/tables", type=str)
    args = ap.parse_args()

    broot = Path(args.baselines_root)
    outroot = Path(args.out_root)
    metrics = load_csv(Path(args.metrics))
    layout_metrics = None
    if Path(args.layout_metrics).exists():
        try:
            layout_metrics = json.loads(Path(args.layout_metrics).read_text(encoding="utf-8"))
        except Exception:
            layout_metrics = None

    # TBF accumulator
    tbf_entries: list[dict] = []

    # A. Doc QA
    b_docqa = load_csv(broot / "docqa.csv")
    header = ["Model", "DocVQA (ANLS)", "ChartQA (ANLS)", "TextVQA (ANLS)", "Source"]
    rows = [[r.get("model",""), r.get("docvqa",""), r.get("chartqa",""), r.get("textvqa",""), r.get("source","") ] for r in b_docqa]
    # Append Bullseye aggregated QA (doc-level) if available
    if metrics:
        # Compute mean ANLS across rows that have qa_anls
        anls_vals = []
        for m in metrics:
            v = m.get("qa_anls")
            try:
                if v != "":
                    anls_vals.append(float(v))
            except Exception:
                pass
        anls_mean = (sum(anls_vals)/len(anls_vals)) if anls_vals else None
        if anls_mean is None:
            tbf_entries.append({
                "section": "docqa",
                "field": "DocVQA(ANLS)",
                "reason": "No QA metrics were aggregated. Provide <stem>.qa.jsonl next to inputs and run collector with --with-llm true --eval-qa true.",
            })
        rows.append([
            "Bullseye (ours)",
            f"{anls_mean:.3f}" if anls_mean is not None else "tbf",
            "tbf",
            "tbf",
            "This work"
        ])
        # Always tbf for ChartQA/TextVQA unless a dataset-specific QA pipeline is provided
        tbf_entries.append({
            "section": "docqa",
            "field": "ChartQA(TextVQA)",
            "reason": "Not aggregated by default. Provide dataset-specific QA files or extend collector to tag dataset splits.",
        })
    write_tex_table(outroot / "docqa.tex", "Document QA results.", header, rows)

    # B. Layout
    b_layout = load_csv(broot / "layout.csv")
    header = ["Method", "Split", "mAP@0.5:0.95", "Source"]
    rows = [[r.get("method",""), r.get("split",""), r.get("map",""), r.get("source","") ] for r in b_layout]
    bull_map = None
    if layout_metrics and isinstance(layout_metrics, dict):
        bull_map = layout_metrics.get("map_50_95")
    if not isinstance(bull_map, (int, float)):
        tbf_entries.append({
            "section": "layout",
            "field": "mAP@0.5:0.95",
            "reason": "results/layout_metrics.json missing or no map_50_95. Run: python -m src.eval_layout --config <doclaynet_cfg> --output results/layout_metrics.json",
        })
    rows.append(["Bullseye (ours)", "DocLayNet", f"{bull_map:.3f}" if isinstance(bull_map,(int,float)) else "tbf", "This work"])
    write_tex_table(outroot / "layout.tex", "DocLayNet(-P) layout detection.", header, rows)

    # C. Tables
    b_tables = load_csv(broot / "tables.csv")
    header = ["Method", "Dataset", "Metric", "Score", "Source"]
    rows = [[r.get("method",""), r.get("dataset",""), r.get("metric",""), r.get("score",""), r.get("source","") ] for r in b_tables]
    # Aggregate TEDS from metrics.csv if present
    teds_vals = []
    for m in metrics:
        v = m.get("teds")
        try:
            if v != "":
                teds_vals.append(float(v))
        except Exception:
            pass
    teds_mean = (sum(teds_vals)/len(teds_vals)) if teds_vals else None
    if teds_mean is None:
        tbf_entries.append({
            "section": "tables",
            "field": "TEDS(PubTabNet)",
            "reason": "No TEDS computed. Install `teds` and provide GT HTML (e.g., <stem>.gt.html) then run collector with --teds true.",
        })
    rows.append(["Bullseye (ours)", "PubTabNet", "TEDS (All)", f"{teds_mean:.3f}" if teds_mean is not None else "tbf", "This work"])
    write_tex_table(outroot / "tables.tex", "Table structure recognition.", header, rows)

    # Emit TBF logs (human + machine)
    try:
        # human-readable
        tbf_log_path = Path(args.tbf_log)
        tbf_log_path.parent.mkdir(parents=True, exist_ok=True)
        with tbf_log_path.open("w", encoding="utf-8") as f:
            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            f.write(f"# TBF log generated at {ts}\n")
            for e in tbf_entries:
                f.write(f"- [{e['section']}] {e['field']}: {e['reason']}\n")
        # machine-readable JSONL
        tbf_jsonl = Path(args.tbf_log_jsonl)
        tbf_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with tbf_jsonl.open("a", encoding="utf-8") as jf:
            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            for e in tbf_entries:
                rec = {"ts": ts, **e}
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

    print("Generated TeX tables under", outroot)
    if tbf_entries:
        print("TBF entries logged to:", args.tbf_log, "and", args.tbf_log_jsonl)


if __name__ == "__main__":
    main()
