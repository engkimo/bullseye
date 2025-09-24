#!/usr/bin/env python3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from pathlib import Path


def make_pdf(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(p), pagesize=A4)
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
    c.setFont('HeiseiMin-W3', 14)
    c.drawString(72, 800, 'これはDocJAのサンプルPDFです。')
    c.drawString(72, 780, 'レイアウト解析と表の検出をテストします。')
    c.drawString(72, 760, '合計金額: 123,456円')
    c.showPage()
    c.save()


if __name__ == '__main__':
    make_pdf('data/samples/sample.pdf')

