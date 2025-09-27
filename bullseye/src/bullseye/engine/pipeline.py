from __future__ import annotations

from typing import Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..core.interfaces import (
    TextDetectorIF,
    TextRecognizerIF,
    LayoutAnalyzerIF,
)
from ..ocr import ocr_aggregate
from ..schemas import (
    OCRSchema,
    ParagraphSchema,
    DocumentAnalyzerSchema,
)
from ..utils.page_grouping import (
    combine_flags,
    judge_page_direction,
    extract_paragraph_within_figure,
    extract_words_within_element,
)
from ..reading_order import prediction_reading_order
from ..utils.misc import quad_to_xyxy


class DocumentEngine:
    """A minimal, non-invasive facade that composes detector/recognizer/layout.

    This does not replace DocumentAnalyzer yet; it offers a stable entrypoint
    for future provider-agnostic wiring.
    """

    def __init__(
        self,
        detector: TextDetectorIF,
        recognizer: TextRecognizerIF,
        layout: LayoutAnalyzerIF,
        visualize: bool = False,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.layout = layout
        self.visualize = visualize

    def __call__(self, img: np.ndarray) -> Tuple[Any, Any, Any]:
        # Run detection and layout in parallel, then recognition.
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_det = ex.submit(self.detector, img)
            f_lay = ex.submit(self.layout, img)
            det_res, det_vis = f_det.result()
            lay_res, lay_vis = f_lay.result()

        rec_vis_in = det_vis if self.visualize else None
        rec_res, rec_vis = self.recognizer(img, getattr(det_res, 'points', None), rec_vis_in)
        return det_res, rec_res, lay_res

    # Experimental: high-level aggregation similar to DocumentAnalyzer
    def analyze(
        self,
        img: np.ndarray,
        *,
        reading_order: str = "auto",
        ignore_meta: bool = False,
        split_text_across_cells: bool = False,
    ) -> Tuple[DocumentAnalyzerSchema, Any, Any]:
        det_res, rec_res, lay_res = self(img)

        if split_text_across_cells:
            try:
                from ..utils.cell_assignment import split_text_across_cells as _split
                det_res = _split(det_res, lay_res)
            except Exception:
                pass

        ocr = OCRSchema(words=ocr_aggregate(det_res, rec_res))

        # aggregate paragraphs/tables/figures
        paragraphs = []
        check_list = [False] * len(ocr.words)

        # tables: attach cell contents
        for table in getattr(lay_res, 'tables', []):
            for cell in table.cells:
                words, direction, flags = extract_words_within_element(ocr.words, cell)
                if words is None:
                    words = ""
                cell.contents = words
                check_list = combine_flags(check_list, flags)

        # paragraphs from layout elements
        for paragraph in getattr(lay_res, 'paragraphs', []):
            words, direction, flags = extract_words_within_element(ocr.words, paragraph)
            if words is None:
                continue
            p = ParagraphSchema(
                contents=words,
                box=paragraph.box,
                direction=direction,
                order=0,
                role=paragraph.role,
            )
            check_list = combine_flags(check_list, flags)
            paragraphs.append(p)

        # remaining words as single paragraphs
        for i, word in enumerate(ocr.words):
            if not check_list[i]:
                direction = word.direction
                p = ParagraphSchema(
                    contents=word.content,
                    box=quad_to_xyxy(word.points),
                    direction=direction,
                    order=0,
                    role=None,
                )
                paragraphs.append(p)

        figures, flags = extract_paragraph_within_figure(paragraphs, getattr(lay_res, 'figures', []))
        paragraphs = [p for p, f in zip(paragraphs, flags) if not f]

        page_direction = judge_page_direction(paragraphs)

        headers = [p for p in paragraphs if p.role == "page_header" and not ignore_meta]
        footers = [p for p in paragraphs if p.role == "page_footer" and not ignore_meta]
        page_contents = [p for p in paragraphs if p.role is None or p.role == "section_headings"]

        elements = page_contents + list(getattr(lay_res, 'tables', [])) + figures

        prediction_reading_order(headers, "left2right")
        prediction_reading_order(footers, "left2right")

        ro = "right2left" if (reading_order == "auto" and page_direction == "vertical") else (
            reading_order if reading_order != "auto" else "top2bottom"
        )
        prediction_reading_order(elements, ro, img)

        for e in elements:
            e.order += len(headers)
        for f in footers:
            f.order += len(elements) + len(headers)

        paragraphs = headers + page_contents + footers
        paragraphs = sorted(paragraphs, key=lambda x: x.order)
        figures = sorted(figures, key=lambda x: x.order)
        tables = sorted(getattr(lay_res, 'tables', []), key=lambda x: x.order)

        result = DocumentAnalyzerSchema(
            paragraphs=paragraphs,
            tables=tables,
            figures=figures,
            words=ocr.words,
        )
        return result, ocr, lay_res
