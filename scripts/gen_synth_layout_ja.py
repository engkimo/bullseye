#!/usr/bin/env python3
import random
import json
from pathlib import Path
from typing import List, Dict

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


CATEGORIES = [
    {"id": 1, "name": "title"},
    {"id": 2, "name": "section_heading"},
    {"id": 3, "name": "paragraph"},
    {"id": 4, "name": "caption"},
    {"id": 5, "name": "figure"},
    {"id": 6, "name": "table"},
    {"id": 7, "name": "list"},
    {"id": 8, "name": "footnote"},
    {"id": 9, "name": "header"},
    {"id": 10, "name": "footer"},
]


def _rand_text() -> str:
    samples = [
        "これはサンプルの段落です。日本語のレイアウト学習用に生成しています。",
        "図1: サンプル図のキャプションです。",
        "表1: サンプル表のキャプションです。",
        "- 箇条書きの項目その1",
        "- 箇条書きの項目その2",
    ]
    return random.choice(samples)


def gen_coco(out_dir: Path, n_pages: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    ann = {"info": {"description": "VisLayJA-Synth", "version": "1.0"},
           "images": [], "annotations": [], "categories": CATEGORIES}

    pdf_path = out_dir / "synth_layout.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    ann_id = 1
    for i in range(n_pages):
        img_id = i + 1
        ann["images"].append({"id": img_id, "file_name": f"page_{img_id}.png",
                               "width": int(width), "height": int(height)})

        # Header
        c.setFont('HeiseiMin-W3', 10)
        c.drawString(40, height - 30, "ヘッダ: Visionalizer-AI 合成ページ")
        ann["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 9,
                                    "bbox": [40, height - 40, 200, 20], "area": 4000, "iscrowd": 0})
        ann_id += 1

        # Title
        c.setFont('HeiseiMin-W3', 18)
        c.drawString(60, height - 80, f"サンプル文書のタイトル {img_id}")
        ann["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                    "bbox": [60, height - 95, 400, 30], "area": 12000, "iscrowd": 0})
        ann_id += 1

        # Paragraphs (2 columns)
        c.setFont('HeiseiMin-W3', 12)
        y = height - 130
        for col_x in (60, width/2 + 20):
            for _ in range(6):
                text = _rand_text()
                c.drawString(col_x, y, text)
                ann["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 3,
                                            "bbox": [col_x, y-12, 300, 18], "area": 5400, "iscrowd": 0})
                ann_id += 1
                y -= 20
            y = height - 130

        # Figure + caption
        c.rect(60, 200, 200, 120)
        c.drawString(60, 180, "図: サンプル図")
        ann["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 5,
                                    "bbox": [60, 200, 200, 120], "area": 24000, "iscrowd": 0})
        ann_id += 1
        ann["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 4,
                                    "bbox": [60, 175, 200, 20], "area": 4000, "iscrowd": 0})
        ann_id += 1

        # Footer
        c.setFont('HeiseiMin-W3', 10)
        c.drawString(40, 20, f"フッタ: ページ {img_id}")
        ann["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 10,
                                    "bbox": [40, 10, 200, 20], "area": 4000, "iscrowd": 0})
        ann_id += 1

        c.showPage()

    c.save()
    # Render PDF pages to PNG images using pypdfium2
    try:
        import pypdfium2 as pdfium
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        doc = pdfium.PdfDocument(str(pdf_path))
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=2.0)  # ~144DPI x2
            pil = bitmap.to_pil()
            pil.save(str(img_dir / f"page_{i+1}.png"))
        print(f"[gen] Rendered {len(doc)} PNG pages to {img_dir}")
    except Exception as e:
        print(f"[warn] Failed to render images from PDF: {e}")
    (out_dir / "annotations.json").write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[gen] Generated COCO annotations at {out_dir/ 'annotations.json'} and PDF {pdf_path}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/synth_layout_ja')
    ap.add_argument('--pages', type=int, default=20)
    args = ap.parse_args()
    gen_coco(Path(args.out), n_pages=args.pages)
