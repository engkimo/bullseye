#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import List

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import pypdfium2 as pdfium


JP_SAMPLES: List[str] = [
    "請求金額は一二三四五円です。",
    "発注日 二〇二五年八月三一日。",
    "合計 三万二千円（税込）。",
    "株式会社サンプル 電機 部品",
    "住所 東京都千代田区一丁目一番地",
    "品目 ネジ サイズM3 数量10",
    "備考 返品不可 納期厳守",
]

EN_SAMPLES: List[str] = [
    "Invoice total is 1,234 JPY.",
    "Issue date: 2025-08-31.",
    "Total amount: 32,000 (tax incl.)",
    "Sample Electric Co., Ltd.",
    "Tokyo Chiyoda 1-1-1",
    "Item: screw size M3 qty 10",
    "Note: No return, on-time delivery",
]


def generate_lines(out_dir: Path, n_total: int = 200, jp_ratio: float = 0.8):
    out_dir = Path(out_dir)
    img_dir = out_dir / "lines"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.json"

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    labels = {}
    # Generate a temporary PDF and render each page to a PNG line image
    tmp_pdf = out_dir / "_lines.pdf"
    c = canvas.Canvas(str(tmp_pdf), pagesize=A4)
    width, height = A4

    for i in range(n_total):
        is_jp = (i < int(n_total * jp_ratio))
        text = random.choice(JP_SAMPLES if is_jp else EN_SAMPLES)

        c.setFont("HeiseiMin-W3", 16)
        # clear page background and draw text roughly centered
        c.drawString(60, height / 2, text)
        c.showPage()
        # Name
        name = f"line_{i:05d}.png"
        labels[name] = text

    c.save()

    # Render PDF pages to images
    doc = pdfium.PdfDocument(str(tmp_pdf))
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=2.0)
        pil = bitmap.to_pil()
        pil = pil.convert("RGB")
        pil.save(str(img_dir / f"line_{i:05d}.png"))

    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    # Cleanup temp pdf if desired
    try:
        tmp_pdf.unlink()
    except Exception:
        pass

    print(f"[gen] Generated {n_total} line images at {img_dir} with labels {labels_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rec")
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--jp-ratio", type=float, default=0.8)
    args = ap.parse_args()
    generate_lines(Path(args.out), args.total, args.jp_ratio)

