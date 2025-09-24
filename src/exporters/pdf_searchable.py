from typing import Any, Optional
from pathlib import Path
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PDFSearchableExporter:
    """Export document processing results as searchable PDF without PyMuPDF.

    - Renders background pages via pypdfium2 when original PDF is available.
    - Overlays (optionally invisible) text using reportlab.
    """

    def __init__(self, preserve_layout: bool = True, invisible_text: bool = True):
        self.preserve_layout = preserve_layout
        self.invisible_text = invisible_text

    def export(self, result: Any, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
        except ImportError:
            logger.error("reportlab is required for PDF export. Install with: pip install reportlab")
            raise

        c = canvas.Canvas(str(output_path))

        # Iterate pages
        for page in result.pages:
            width = float(page.width)
            height = float(page.height)
            c.setPageSize((width, height))

            # Draw original page as background if requested and available
            if self.preserve_layout and hasattr(result, 'original_pdf_path') and Path(getattr(result, 'original_pdf_path')).exists():
                try:
                    bg_img = self._render_pdf_page_to_image(getattr(result, 'original_pdf_path'), page.page_num)
                    if bg_img is not None:
                        c.drawImage(ImageReader(bg_img), 0, 0, width=width, height=height, preserveAspectRatio=False, mask='auto')
                except Exception as e:
                    logger.warning(f"Background render failed for page {page.page_num}: {e}")

            # Overlay text blocks (invisible or faint)
            self._draw_text_blocks(c, page, width, height)

            # Optionally overlay table cell text
            self._draw_table_cells(c, page, width, height)

            c.showPage()

        c.save()

    def _render_pdf_page_to_image(self, pdf_path: str, page_num_1based: int) -> Optional[Image.Image]:
        try:
            import pypdfium2 as pdfium
        except ImportError:
            logger.warning("pypdfium2 not installed; background rendering skipped")
            return None

        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            idx = page_num_1based - 1
            if idx < 0 or idx >= len(doc):
                return None
            page = doc[idx]
            scale = 150.0 / 72.0  # lighter background DPI
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            return pil.convert('RGB')
        except Exception as e:
            logger.warning(f"pypdfium2 render error: {e}")
            return None

    def _draw_text_blocks(self, c, page, width: float, height: float):
        # Determine drawing style
        c.saveState()
        # Try to make text invisible
        if self.invisible_text:
            try:
                c.setFillAlpha(0.0)
            except Exception:
                # Fallback: use white color (may still be faintly visible on non-white backgrounds)
                c.setFillColorRGB(1, 1, 1)
        else:
            c.setFillColorRGB(0, 0, 0)

        # Order blocks
        blocks = page.text_blocks
        if page.reading_order:
            try:
                blocks = [page.text_blocks[i] for i in page.reading_order]
            except Exception:
                pass

        for block in blocks:
            text = getattr(block, 'text', '') or ''
            x1, y1, x2, y2 = [float(v) for v in block.bbox]
            # Convert image-space (top-left origin) to PDF-space (bottom-left origin)
            px = x1
            py = height - y2
            font_size = self._estimate_font_size(block, x2 - x1, y2 - y1)
            try:
                c.setFont("Helvetica", font_size)
            except Exception:
                pass

            # Draw multi-line text within box: naive line split
            lines = text.split('\n') if text else []
            leading = font_size * 1.1
            ty = py + (y2 - y1) - leading
            for line in lines:
                if ty < 0:
                    break
                c.drawString(px, ty, line)
                ty -= leading

        c.restoreState()

    def _draw_table_cells(self, c, page, width: float, height: float):
        if not getattr(page, 'tables', None):
            return
        c.saveState()
        if self.invisible_text:
            try:
                c.setFillAlpha(0.0)
            except Exception:
                c.setFillColorRGB(1, 1, 1)
        else:
            c.setFillColorRGB(0, 0, 0)

        for table in page.tables:
            for cell in getattr(table, 'cells', []):
                if 'bbox' not in cell:
                    continue
                text = cell.get('text', '')
                x1, y1, x2, y2 = [float(v) for v in cell['bbox']]
                px = x1
                py = height - y2
                font_size = max(8, min(12, (y2 - y1) * 0.4))
                try:
                    c.setFont("Helvetica", font_size)
                except Exception:
                    pass
                c.drawString(px, py + (y2 - y1 - font_size * 1.1) / 2.0, text)
        c.restoreState()

    def _estimate_font_size(self, block, w: float, h: float) -> float:
        base_sizes = {
            'title': 16,
            'section_heading': 14,
            'caption': 9,
            'footnote': 8,
            'text': 11,
            'paragraph': 11,
        }
        base = base_sizes.get(getattr(block, 'block_type', 'text'), 11)
        if w <= 0 or h <= 0:
            return base
        # Simple fit heuristic
        text_len = max(1, len(getattr(block, 'text', '') or ''))
        est_lines = max(1.0, h / (base * 1.2))
        chars_per_line = max(1.0, w / (base * 0.6))
        needed_lines = text_len / chars_per_line
        if needed_lines > est_lines:
            scale = (est_lines / needed_lines)
            return max(6, base * scale)
        return base
