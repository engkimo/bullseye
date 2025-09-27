from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from bullseye.text_detector import TextDetector
from bullseye.text_recognizer import TextRecognizer

from .layout_analyzer import LayoutAnalyzer
from .ocr import OCRSchema, ocr_aggregate
from .reading_order import prediction_reading_order
from .utils.misc import calc_overlap_ratio, is_contained, quad_to_xyxy
from .utils.visualizer import det_visualizer, reading_order_visualizer
from .schemas import ParagraphSchema, FigureSchema, DocumentAnalyzerSchema
from .utils.geometry import is_vertical, is_noise
from .utils.page_grouping import (
    combine_flags,
    judge_page_direction,
    extract_paragraph_within_figure,
    extract_words_within_element,
)
from .utils.cell_assignment import split_text_across_cells
from .utils.logger import set_logger

logger = set_logger(__name__, "INFO")


 



def recursive_update(original, new_data):
    for key, value in new_data.items():
        # `value`が辞書の場合、再帰的に更新
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            recursive_update(original[key], value)
        # `value`が辞書でない場合、またはキーが存在しない場合に上書き
        else:
            original[key] = value
    return original


 


class DocumentAnalyzer:
    def __init__(
        self,
        configs={},
        device="cuda",
        visualize=False,
        ignore_meta=False,
        reading_order="auto",
        split_text_across_cells=False,
        provider=None,
    ):
        default_configs = {
            "ocr": {
                "text_detector": {
                    "device": device,
                    "visualize": visualize,
                },
                "text_recognizer": {
                    "device": device,
                    "visualize": visualize,
                },
            },
            "layout_analyzer": {
                "layout_parser": {
                    "device": device,
                    "visualize": visualize,
                },
                "table_structure_recognizer": {
                    "device": device,
                    "visualize": visualize,
                },
            },
        }

        self.reading_order = reading_order

        if isinstance(configs, dict):
            recursive_update(default_configs, configs)
        else:
            raise ValueError(
                "configs must be a dict."
            )

        if provider is not None:
            # provider adapters implement the same call signature as existing modules
            self.text_detector = provider.detector
            self.text_recognizer = provider.recognizer
            self.layout = provider.layout
        else:
            self.text_detector = TextDetector(
                **default_configs["ocr"]["text_detector"],
            )
            self.text_recognizer = TextRecognizer(
                **default_configs["ocr"]["text_recognizer"]
            )

            self.layout = LayoutAnalyzer(
                configs=default_configs["layout_analyzer"],
            )
        self.visualize = visualize

        self.ignore_meta = ignore_meta
        self.split_text_across_cells = split_text_across_cells

    def aggregate(self, ocr_res, layout_res):
        """Aggregate OCR words and layout elements into page paragraphs/tables/figures."""
        paragraphs = []
        check_list = [False] * len(ocr_res.words)
        for table in layout_res.tables:
            for cell in table.cells:
                words, direction, flags = extract_words_within_element(
                    ocr_res.words, cell
                )

                if words is None:
                    words = ""

                cell.contents = words
                check_list = combine_flags(check_list, flags)

        for paragraph in layout_res.paragraphs:
            words, direction, flags = extract_words_within_element(
                ocr_res.words, paragraph
            )

            if words is None:
                continue

            paragraph = {
                "contents": words,
                "box": paragraph.box,
                "direction": direction,
                "order": 0,
                "role": paragraph.role,
            }

            check_list = combine_flags(check_list, flags)
            paragraph = ParagraphSchema(**paragraph)
            paragraphs.append(paragraph)

        for i, word in enumerate(ocr_res.words):
            direction = word.direction
            if not check_list[i]:
                paragraph = {
                    "contents": word.content,
                    "box": quad_to_xyxy(word.points),
                    "direction": direction,
                    "order": 0,
                    "role": None,
                }

                paragraph = ParagraphSchema(**paragraph)
                paragraphs.append(paragraph)

        figures, check_list = extract_paragraph_within_figure(
            paragraphs, layout_res.figures
        )

        paragraphs = [
            paragraph for paragraph, flag in zip(paragraphs, check_list) if not flag
        ]

        page_direction = judge_page_direction(paragraphs)

        headers = [
            paragraph
            for paragraph in paragraphs
            if paragraph.role == "page_header" and not self.ignore_meta
        ]

        footers = [
            paragraph
            for paragraph in paragraphs
            if paragraph.role == "page_footer" and not self.ignore_meta
        ]

        page_contents = [
            paragraph
            for paragraph in paragraphs
            if paragraph.role is None or paragraph.role == "section_headings"
        ]

        elements = page_contents + layout_res.tables + figures

        prediction_reading_order(headers, "left2right")
        prediction_reading_order(footers, "left2right")

        if self.reading_order == "auto":
            reading_order = (
                "right2left" if page_direction == "vertical" else "top2bottom"
            )
        else:
            reading_order = self.reading_order

        prediction_reading_order(elements, reading_order, self.img)

        for i, element in enumerate(elements):
            element.order += len(headers)
        for i, footer in enumerate(footers):
            footer.order += len(elements) + len(headers)

        paragraphs = headers + page_contents + footers
        paragraphs = sorted(paragraphs, key=lambda x: x.order)
        figures = sorted(figures, key=lambda x: x.order)
        tables = sorted(layout_res.tables, key=lambda x: x.order)

        outputs = {
            "paragraphs": paragraphs,
            "tables": tables,
            "figures": figures,
            "words": ocr_res.words,
        }

        return outputs

    async def run(self, img):
        """Run detection and layout in parallel, then recognize + aggregate."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                # loop.run_in_executor(executor, self.ocr, img),
                loop.run_in_executor(executor, self.text_detector, img),
                loop.run_in_executor(executor, self.layout, img),
            ]

            results = await asyncio.gather(*tasks)

            results_det, _ = results[0]
            results_layout, layout = results[1]

            if self.split_text_across_cells:
                results_det = split_text_across_cells(results_det, results_layout)

            vis_det = None
            if self.visualize:
                vis_det = det_visualizer(
                    img,
                    results_det.points,
                )

            results_rec, ocr = self.text_recognizer(img, results_det.points, vis_det)

            outputs = {"words": ocr_aggregate(results_det, results_rec)}
            results_ocr = OCRSchema(**outputs)
            outputs = self.aggregate(results_ocr, results_layout)

        results = DocumentAnalyzerSchema(**outputs)
        logger.debug(
            "DocumentAnalyzer: paragraphs=%d tables=%d figures=%d words=%d",
            len(results.paragraphs), len(results.tables), len(results.figures), len(results.words)
        )
        return results, ocr, layout

    def __call__(self, img):
        """Synchronous wrapper that executes the full document analysis pipeline."""
        self.img = img
        results, ocr, layout = asyncio.run(self.run(img))

        if self.visualize:
            layout = reading_order_visualizer(layout, results)

        return results, ocr, layout
