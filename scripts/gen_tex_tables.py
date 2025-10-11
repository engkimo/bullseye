#!/usr/bin/env python3
"""
Generate TeX tables from CSV baselines and local metrics.

Inputs (default locations):
 - docs/paper/baselines/docqa.csv
 - docs/paper/baselines/layout.csv
 - docs/paper/baselines/tables.csv
 - results/metrics/metrics.csv (optional; adds Bullseye row)

Outputs:
 - docs/paper/tables/docqa.tex
 - docs/paper/tables/layout.tex
 - docs/paper/tables/tables.tex
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path


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
    ap.add_argument("--out-root", default="docs/paper/tables", type=str)
    args = ap.parse_args()

    broot = Path(args.baselines_root)
    outroot = Path(args.out_root)
    metrics = load_csv(Path(args.metrics))

    # A. Doc QA
    b_docqa = load_csv(broot / "docqa.csv")
    header = ["Model", "DocVQA (ANLS)", "ChartQA (ANLS)", "TextVQA (ANLS)", "Source"]
    rows = [[r.get("model",""), r.get("docvqa",""), r.get("chartqa",""), r.get("textvqa",""), r.get("source","") ] for r in b_docqa]
    # Append Bullseye placeholder or computed
    if metrics:
        # Aggregate ANLS not available ⇒ leave placeholder
        rows.append(["Bullseye (ours)", "tbf", "tbf", "tbf", "This work"])
    write_tex_table(outroot / "docqa.tex", "Document QA results.", header, rows)

    # B. Layout
    b_layout = load_csv(broot / "layout.csv")
    header = ["Method", "Split", "mAP@0.5:0.95", "Source"]
    rows = [[r.get("method",""), r.get("split",""), r.get("map",""), r.get("source","") ] for r in b_layout]
    rows.append(["Bullseye (ours)", "DocLayNet", "tbf", "This work"])
    write_tex_table(outroot / "layout.tex", "DocLayNet(-P) layout detection.", header, rows)

    # C. Tables
    b_tables = load_csv(broot / "tables.csv")
    header = ["Method", "Dataset", "Metric", "Score", "Source"]
    rows = [[r.get("method",""), r.get("dataset",""), r.get("metric",""), r.get("score",""), r.get("source","") ] for r in b_tables]
    rows.append(["Bullseye (ours)", "PubTabNet", "TEDS (All)", "tbf", "This work"])
    write_tex_table(outroot / "tables.tex", "Table structure recognition.", header, rows)

    print("Generated TeX tables under", outroot)


if __name__ == "__main__":
    main()

