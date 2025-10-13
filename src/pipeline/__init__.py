import logging
import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import time

from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .layout_detector import LayoutDetector
from .reading_order import ReadingOrderEstimator
try:
    from ..llm.router import LLMRouterClient
except Exception:  # pragma: no cover
    LLMRouterClient = None  # type: ignore


logger = logging.getLogger(__name__)


class TextBlock:
    def __init__(self, bbox: List[float], text: str, confidence: float = 1.0,
                 block_type: str = 'text', metadata: Dict[str, Any] = None):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.text = text
        self.confidence = confidence
        self.block_type = block_type
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            'bbox': self.bbox,
            'text': self.text,
            'confidence': self.confidence,
            'type': self.block_type,
            'metadata': self.metadata
        }


class Table:
    def __init__(self, bbox: List[float], cells: List[Dict], 
                 html: str = '', markdown: str = ''):
        self.bbox = bbox
        self.cells = cells
        self.html = html
        self.markdown = markdown
    
    def to_dict(self):
        return {
            'bbox': self.bbox,
            'cells': self.cells,
            'html': self.html,
            'markdown': self.markdown
        }


class Page:
    def __init__(self, page_num: int, width: int, height: int):
        self.page_num = page_num
        self.width = width
        self.height = height
        self.text_blocks: List[TextBlock] = []
        self.tables: List[Table] = []
        self.figures: List[Dict] = []
        self.reading_order: List[int] = []
        self.layout_elements: List[Dict[str, Any]] = []
        # Extended: structured diagrams
        self.graphs: List[Dict[str, Any]] = []  # Flow/graph JSONs
        self.charts: List[Dict[str, Any]] = []  # Gantt/chart JSONs
    
    def add_text_block(self, block: TextBlock):
        self.text_blocks.append(block)
    
    def add_table(self, table: Table):
        self.tables.append(table)
    
    def to_dict(self):
        return {
            'page_num': self.page_num,
            'width': self.width,
            'height': self.height,
            'text_blocks': [b.to_dict() for b in self.text_blocks],
            'tables': [t.to_dict() for t in self.tables],
            'figures': self.figures,
            'reading_order': self.reading_order,
            'layout_elements': self.layout_elements,
            'graphs': self.graphs,
            'charts': self.charts
        }


class DocumentResult:
    def __init__(self, filename: str):
        self.filename = filename
        self.pages: List[Page] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_page(self, page: Page):
        self.pages.append(page)
    
    def to_dict(self):
        return {
            'filename': self.filename,
            'pages': [p.to_dict() for p in self.pages],
            'metadata': self.metadata
        }


class DocumentProcessor:
    def __init__(self, 
                 det_model: str = 'dbnet',
                 rec_model: str = 'abinet',
                 rec_provider: str = 'internal',
                 rec_hf_model_id: Optional[str] = None,
                 det_provider: str = 'internal',
                 det_hf_model_id: Optional[str] = None,
                 layout_model: Optional[str] = None,
                 layout_provider: str = 'internal',
                 layout_hf_model_id: Optional[str] = None,
                 enable_table: bool = False,
                 table_provider: str = 'internal',
                 table_hf_model_id: Optional[str] = None,
                 enable_reading_order: bool = False,
                 enable_llm: bool = False,
                 device: str = 'cuda',
                 weights_dir: str = 'weights',
                 lite_mode: bool = False,
                 table_ocr_conf_threshold: Optional[float] = None,
                 table_blank_low_confidence: Optional[bool] = None):
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.weights_dir = Path(weights_dir)
        
        # Initialize models
        logger.info(f"Initializing DocumentProcessor on {self.device}")
        
        # Text detection (provider override supported)
        self.text_detector = None
        chosen_det_provider = (det_provider or 'internal').lower()
        if chosen_det_provider == 'bullseye' and os.getenv('DOCJA_FORCE_BULLSEYE_SERVICE','0') == '1':
            chosen_det_provider = 'bullseye-service'
        no_internal_fallback = (os.getenv('DOCJA_NO_INTERNAL_FALLBACK','0')=='1')
        no_hf = (os.getenv('DOCJA_NO_HF','0')=='1')
        if chosen_det_provider in ('bullseye',):
            try:
                from ..integrations.bullseye_dbnet import BullseyeDbnetDetector
                # Map HF id hint to model_name
                model_name = 'dbnetv2'
                if det_hf_model_id and isinstance(det_hf_model_id, str):
                    model_name = 'dbnetv2' if 'v2' in det_hf_model_id else 'dbnet'
                self.text_detector = BullseyeDbnetDetector(
                    model_name=model_name,
                    device=self.device,
                    fail_hard=no_internal_fallback,
                )
            except Exception as e:
                logger.warning(f"Bullseye detector init failed: {e}")
                if no_internal_fallback:
                    raise
        elif chosen_det_provider in ('bullseye-service','bullseye_svc','bullseye-http','bullseye_http'):
            try:
                from ..integrations.bullseye_service import BullseyeServiceDetector
                self.text_detector = BullseyeServiceDetector()
            except Exception as e:
                logger.warning(f"Bullseye service detector init failed: {e}")
                if no_internal_fallback:
                    raise
        if self.text_detector is None:
            try:
                self.text_detector = TextDetector(
                    model_type=det_model,
                    weights_path=self.weights_dir / f"det/{det_model}.pth",
                    device=self.device,
                    lite_mode=lite_mode
                )
            except Exception as e:
                logger.warning(f"Internal TextDetector init failed: {e}. Proceeding without detector; OCR may be empty")
        
        # Text recognition (allow disabling)
        self.text_recognizer = None
        if rec_model and str(rec_model).lower() != 'none':
            rec_prov = (rec_provider or 'internal').lower()
            if rec_prov == 'bullseye' and os.getenv('DOCJA_FORCE_BULLSEYE_SERVICE','0') == '1':
                rec_prov = 'bullseye-service'
            if rec_prov in ('bullseye',):
                try:
                    from ..integrations.bullseye import BullseyeParseqRecognizer
                    # Prefer explicit env repos for bullseye if provided
                    model_id = (
                        rec_hf_model_id
                        or os.getenv('DOCJA_REC_MODEL_ID')
                        or os.getenv('DOCJA_BULLSEYE_REC_REPO')
                        or 'Ryousukee/bullseye-recparseq'
                    )
                    self.text_recognizer = BullseyeParseqRecognizer(
                        model_id=model_id, device=self.device, no_hf=no_hf, fail_hard=no_internal_fallback
                    )
                    # record provider meta later in result
                except Exception as e:
                    logger.warning(f"Bullseye recognizer init failed: {e}")
                    if no_internal_fallback:
                        raise
                    try:
                        self.text_recognizer = TextRecognizer(
                            model_type=rec_model,
                            weights_path=self.weights_dir / f"rec/{rec_model}.pth",
                            device=self.device
                        )
                    except Exception as ie:
                        logger.warning(f"TextRecognizer init failed ({rec_model}): {ie}. Proceeding without recognizer.")
            elif rec_prov in ('bullseye-service','bullseye_svc','bullseye-http','bullseye_http'):
                try:
                    from ..integrations.bullseye_service import BullseyeServiceRecognizer
                    self.text_recognizer = BullseyeServiceRecognizer()
                except Exception as e:
                    logger.warning(f"Bullseye service recognizer init failed: {e}")
                    if no_internal_fallback:
                        raise
            else:
                try:
                    self.text_recognizer = TextRecognizer(
                        model_type=rec_model,
                        weights_path=self.weights_dir / f"rec/{rec_model}.pth",
                        device=self.device
                    )
                except Exception as e:
                    logger.warning(f"TextRecognizer init failed ({rec_model}): {e}. Proceeding without recognizer.")
        
        # Layout detection (optional)
        self.layout_detector = None
        lay_prov = (layout_provider or 'internal').lower()
        if lay_prov == 'bullseye' and os.getenv('DOCJA_FORCE_BULLSEYE_SERVICE','0') == '1':
            lay_prov = 'bullseye-service'
        if lay_prov in ('bullseye',):
            try:
                from ..integrations.bullseye_rtdetr import BullseyeRtDetrLayout
                # Use provided HF id for metadata if any; otherwise a canonical label
                model_id = layout_hf_model_id or 'rtdetrv2'
                self.layout_detector = BullseyeRtDetrLayout(model_id=model_id, device=self.device, no_hf=no_hf)
            except Exception as e:
                logger.warning(f"Bullseye layout init failed: {e}")
                if no_internal_fallback:
                    raise
        elif lay_prov in ('bullseye-service','bullseye_svc','bullseye-http','bullseye_http'):
            try:
                from ..integrations.bullseye_service import BullseyeServiceLayout
                self.layout_detector = BullseyeServiceLayout()
            except Exception as e:
                logger.warning(f"Bullseye service layout init failed: {e}")
                if no_internal_fallback:
                    raise
        if self.layout_detector is None and layout_model:
            self.layout_detector = LayoutDetector(
                model_type=layout_model,
                weights_path=self.weights_dir / f"layout/{layout_model}.pth",
                device=self.device
            )
        
        # Table recognition (optional, lazy import to avoid heavy deps when unused)
        self.table_recognizer = None
        if enable_table:
            chosen_table_provider = (table_provider or 'internal').lower()
            if chosen_table_provider == 'bullseye' and os.getenv('DOCJA_FORCE_BULLSEYE_SERVICE','0') == '1':
                chosen_table_provider = 'bullseye-service'
            if chosen_table_provider in ('bullseye',):
                try:
                    from ..integrations.bullseye_table import BullseyeTableRecognizer as _BullseyeTable
                    self.table_recognizer = _BullseyeTable(
                        device=self.device,
                        text_recognizer=self.text_recognizer,
                        ocr_conf_threshold=(table_ocr_conf_threshold if table_ocr_conf_threshold is not None else (0.5 if lite_mode else 0.3)),
                        blank_low_confidence=(table_blank_low_confidence if table_blank_low_confidence is not None else True),
                        fail_hard=no_internal_fallback
                    )
                except Exception as e:
                    logger.warning(f"Bullseye table init failed: {e}")
                    if no_internal_fallback:
                        raise
            elif chosen_table_provider in ('bullseye-service','bullseye_svc','bullseye-http','bullseye_http'):
                try:
                    from ..integrations.bullseye_service import BullseyeServiceTable as _BullseyeTableSvc
                    self.table_recognizer = _BullseyeTableSvc()
                except Exception as e:
                    logger.warning(f"Bullseye service table init failed: {e}")
                    if no_internal_fallback:
                        raise
            if self.table_recognizer is None:
                try:
                    from .table_struct import TableRecognizer  # lazy import
                    self.table_recognizer = TableRecognizer(
                        weights_path=self.weights_dir / "table/tatr.pth",
                        device=self.device,
                        text_recognizer=None,  # set after text_recognizer initialized
                        ocr_conf_threshold=(table_ocr_conf_threshold if table_ocr_conf_threshold is not None else (0.5 if lite_mode else 0.3)),
                        blank_low_confidence=(table_blank_low_confidence if table_blank_low_confidence is not None else True)
                    )
                except Exception as e:
                    logger.error("Failed to initialize TableRecognizer: %s", e)
                    raise
        
        # Reading order (optional)
        self.reading_order_estimator = None
        if enable_reading_order:
            self.reading_order_estimator = ReadingOrderEstimator()
        
        self.enable_llm = enable_llm
        try:
            self.llm_client = (LLMRouterClient() if (enable_llm and LLMRouterClient is not None) else None)
        except Exception:
            self.llm_client = None

        logger.info("DocumentProcessor initialized successfully")

        # Wire table cell OCR
        if self.table_recognizer is not None and self.text_recognizer is not None:
            try:
                self.table_recognizer.text_recognizer = self.text_recognizer
            except Exception:
                pass
        # Provider metadata
        try:
            self._providers: Dict[str, Any] = {}
            alias_label = (os.getenv('DOCJA_PROVIDER_ALIAS_LABEL', '') or '').lower().strip()
            def _alias(v: str) -> str:
                return v
            # recognizer
            if getattr(self, 'text_recognizer', None) is not None:
                prov = getattr(self.text_recognizer, 'provider_label', 'internal') if hasattr(self.text_recognizer, 'provider_label') else 'internal'
                name = getattr(self.text_recognizer, 'name', None)
                lab = str(name() if callable(name) else (name or prov))
                self._providers['recognizer'] = _alias(lab)
            # detector
            if getattr(self, 'text_detector', None) is not None:
                name = getattr(self.text_detector, 'name', None)
                lab = str(name() if callable(name) else (name or 'internal-detector'))
                self._providers['detector'] = _alias(lab)
            # layout
            if getattr(self, 'layout_detector', None) is not None:
                name = getattr(self.layout_detector, 'name', None)
                lab = str(name() if callable(name) else (name or 'internal-layout'))
                self._providers['layout'] = _alias(lab)
            # table
            if getattr(self, 'table_recognizer', None) is not None:
                name = getattr(self.table_recognizer, 'name', None)
                lab = str(name() if callable(name) else (name or 'internal-table'))
                self._providers['table'] = _alias(lab)
        except Exception:
            self._providers = {}
    
    def process(self, 
                input_path: Union[str, Path],
                max_pages: Optional[int] = None,
                extract_figures: bool = False,
                vis_save_dir: Optional[Union[str, Path]] = None,
                llm_task: Optional[str] = None,
                llm_question: Optional[str] = None,
                llm_schema: Optional[Dict[str, Any]] = None,
                extract_figure_text: bool = False,
                pdf_direct_text: bool = False) -> DocumentResult:
        """Process a document through the full pipeline."""
        
        input_path = Path(input_path)
        logger.info(f"Processing document: {input_path}")
        
        # Load images or extract direct text from PDF
        if input_path.suffix.lower() == '.pdf':
            if pdf_direct_text:
                return self._process_pdf_direct_text(input_path, max_pages)
            else:
                images = self._load_pdf_images(input_path, max_pages)
        else:
            images = [Image.open(input_path).convert('RGB')]
        
        # Create result
        result = DocumentResult(str(input_path))
        # Keep original PDF path for exporters (e.g., searchable PDF layer)
        if input_path.suffix.lower() == '.pdf':
            try:
                setattr(result, 'original_pdf_path', str(input_path))
            except Exception:
                # Non-fatal: continue without original path
                pass
        # Attach provider metadata if any
        if getattr(self, '_providers', None):
            try:
                result.metadata.setdefault('providers', {}).update(self._providers)
            except Exception:
                pass
        
        # Prepare visualization output
        vis_root: Optional[Path] = None
        if vis_save_dir:
            vis_root = Path(vis_save_dir)
            vis_root.mkdir(parents=True, exist_ok=True)

        # Process each page
        per_page_metrics: List[Dict[str, Any]] = []
        for page_idx, image in enumerate(images):
            logger.info(f"Processing page {page_idx + 1}/{len(images)}")
            _t_page0 = time.time()
            
            # Create page object
            page = Page(
                page_num=page_idx + 1,
                width=image.width,
                height=image.height
            )
            
            # Convert to numpy
            img_np = np.array(image)
            
            # Text detection
            _t = time.time()
            text_regions = self.text_detector.detect(img_np)
            det_ms = int((time.time() - _t) * 1000)
            logger.debug(f"Detected {len(text_regions)} text regions")
            
            # Layout detection (if enabled)
            layout_blocks = []
            if self.layout_detector:
                _t = time.time()
                layout_blocks = self.layout_detector.detect(img_np)
                layout_ms = int((time.time() - _t) * 1000)
                logger.debug(f"Detected {len(layout_blocks)} layout blocks")
                page.layout_elements = layout_blocks
            else:
                layout_ms = 0

            # Figure extraction (optional, requires layout detection results)
            if extract_figures and layout_blocks:
                try:
                    for el in layout_blocks:
                        label = el.get('label') or el.get('type')
                        if label == 'figure' and 'bbox' in el:
                            x1, y1, x2, y2 = [int(v) for v in el['bbox']]
                            x1 = max(0, min(x1, image.width - 1))
                            y1 = max(0, min(y1, image.height - 1))
                            x2 = max(x1 + 1, min(x2, image.width))
                            y2 = max(y1 + 1, min(y2, image.height))
                            fig_crop = img_np[y1:y2, x1:x2]
                            fig_entry: Dict[str, Any] = {'bbox': [x1, y1, x2, y2], 'image': fig_crop}
                            # Optional: extract text inside figure
                            if extract_figure_text and self.text_detector is not None and self.text_recognizer is not None:
                                try:
                                    dets = self.text_detector.detect(fig_crop)
                                    texts = []
                                    for d in dets:
                                        bx1, by1, bx2, by2 = self._polygon_to_bbox(d.get('polygon', []))
                                        bx1 = max(0, min(int(bx1), fig_crop.shape[1] - 1))
                                        by1 = max(0, min(int(by1), fig_crop.shape[0] - 1))
                                        bx2 = max(bx1 + 1, min(int(bx2), fig_crop.shape[1]))
                                        by2 = max(by1 + 1, min(int(by2), fig_crop.shape[0]))
                                        sub = fig_crop[by1:by2, bx1:bx2]
                                        t, _c = self.text_recognizer.recognize(sub)
                                        if t:
                                            texts.append(t)
                                    if texts:
                                        fig_entry['text'] = "\n".join(texts)
                                except Exception:
                                    pass
                            page.figures.append(fig_entry)
                except Exception as e:
                    logger.warning(f"Figure extraction failed on page {page_idx+1}: {e}")
            
            # Process text regions
            recog_ms = 0
            for region in text_regions:
                # Crop region (clamped and with min size)
                x1, y1, x2, y2 = self._polygon_to_bbox(region['polygon'])
                # Clamp to image bounds
                x1 = max(0, min(int(x1), image.width - 1))
                y1 = max(0, min(int(y1), image.height - 1))
                x2 = max(0, min(int(x2), image.width))
                y2 = max(0, min(int(y2), image.height))
                # Ensure at least 1px size
                if x2 <= x1:
                    x2 = min(x1 + 1, image.width)
                if y2 <= y1:
                    y2 = min(y1 + 1, image.height)
                cropped = img_np[y1:y2, x1:x2]
                
                # Text recognition
                if self.text_recognizer is not None:
                    _t = time.time()
                    text, confidence = self.text_recognizer.recognize(cropped)
                    recog_ms += int((time.time() - _t) * 1000)
                else:
                    text, confidence = "", 0.0
                
                # Determine block type from layout
                block_type = self._get_block_type(region, layout_blocks)
                
                # Create text block
                block = TextBlock(
                    bbox=[x1, y1, x2, y2],
                    text=text,
                    confidence=confidence,
                    block_type=block_type
                )
                page.add_text_block(block)
            
            # Table detection/recognition (if enabled)
            table_ms = 0
            if self.table_recognizer:
                tables: List[Dict[str, Any]] = []
            # If recognizer exposes recognize(img, boxes), prefer that path
                if hasattr(self.table_recognizer, 'recognize') and not hasattr(self.table_recognizer, 'detect_and_recognize'):
                    # Extract table boxes from layout results
                    table_boxes: List[List[float]] = []
                    for el in (layout_blocks or []):
                        label = el.get('label') or el.get('type')
                        if str(label).lower() == 'table' and 'bbox' in el:
                            table_boxes.append([float(v) for v in el['bbox']])
                    if not table_boxes:
                        logger.warning("Table recognizer requires layout table boxes; none found. Skipping tables on this page.")
                        tables = []
                    else:
                        try:
                            _t = time.time()
                            # type: ignore[attr-defined]
                            tables = self.table_recognizer.recognize(img_np, table_boxes)  # noqa
                            table_ms += int((time.time() - _t) * 1000)
                        except Exception as e:
                            logger.warning(f"Table recognize (boxes) failed: {e}")
                            tables = []
                else:
                    try:
                        _t = time.time()
                        tables = self.table_recognizer.detect_and_recognize(img_np)  # type: ignore[attr-defined]
                        table_ms += int((time.time() - _t) * 1000)
                    except Exception as e:
                        logger.warning(f"Table recognize (detect+struct) failed: {e}")
                        tables = []
                for table_data in tables:
                    table = Table(
                        bbox=table_data.get('bbox', [0, 0, 0, 0]),
                        cells=table_data.get('cells', []),
                        html=table_data.get('html', ''),
                        markdown=table_data.get('markdown', '')
                    )
                    page.add_table(table)
            
            # Reading order estimation (if enabled)
            if self.reading_order_estimator:
                _t = time.time()
                reading_order = self.reading_order_estimator.estimate(
                    page.text_blocks, page.width, page.height
                )
                page.reading_order = reading_order
                read_ms = int((time.time() - _t) * 1000)
            else:
                read_ms = 0

            # Flow/Gantt parsing (v0 heuristic)
            parse_ms = 0
            try:
                from ..parsers.flow import parse_flow_from_page
                from ..parsers.gantt import parse_gantt_from_page
                p_dict = page.to_dict()
                _t = time.time()
                # convert BGR->RGB for parsers using image
                img_rgb = img_np[:, :, ::-1]
                page.graphs = parse_flow_from_page(p_dict, img_rgb)
                page.charts = parse_gantt_from_page(p_dict, img_rgb)
                parse_ms = int((time.time() - _t) * 1000)
            except Exception as _e:
                logger.debug(f"Flow/Gantt parse skipped: {_e}")
            
            # LLM integration (page-level summary)
            page_llm_ms = 0
            if self.enable_llm and self.llm_client is not None and (llm_task in (None, 'summary')):
                try:
                    page_texts = []
                    order = page.reading_order if page.reading_order else list(range(len(page.text_blocks)))
                    for idx in order:
                        if 0 <= idx < len(page.text_blocks):
                            page_texts.append(page.text_blocks[idx].text)
                    doc_text = "\n".join(page_texts)[:4000]
                    # Optionally attach the current page image for multimodal models (gemma3)
                    img_for_llm = None
                    if os.getenv('DOCJA_LLM_USE_IMAGE', '0') == '1':
                        try:
                            img_for_llm = [Image.fromarray(img_np[:, :, ::-1]).convert('RGB')]
                        except Exception:
                            img_for_llm = None
                    _t = time.time()
                    summary = self.llm_client.summarize(doc_text, images=img_for_llm)
                    if summary:
                        if 'llm' not in result.metadata:
                            result.metadata['llm'] = {}
                        result.metadata['llm'][f'page_{page.page_num}_summary'] = summary
                    page_llm_ms = int((time.time() - _t) * 1000)
                except Exception as e:
                    logger.warning(f"LLM integration failed on page {page_idx+1}: {e}")

            # Visualization overlays
            if vis_root is not None:
                try:
                    # Optional reject rule for specific vis output paths
                    vis_reject_substr = os.getenv('DOCJA_VIS_REJECT_SUBSTR', 'di_latest/vis')
                    save_allowed = True
                    try:
                        if vis_reject_substr and vis_reject_substr in str(vis_root):
                            save_allowed = False
                    except Exception:
                        save_allowed = True
                    if save_allowed:
                        vis_img = self._draw_overlays(img_np.copy(), text_regions, layout_blocks, page.tables, page.graphs, page.charts)
                        from PIL import Image as _PILImage
                        _PILImage.fromarray(vis_img[:, :, ::-1]).save(str(vis_root / f"{Path(input_path).stem}_page{page_idx+1}.png"))
                    else:
                        logger.info(f"Visualization rejected for path substring '{vis_reject_substr}'; skipping save.")
                except Exception as e:
                    logger.warning(f"Failed to save visualization for page {page_idx+1}: {e}")

            # Add page to result
            result.add_page(page)
            # Page timing
            try:
                per_page_metrics.append({
                    'page': page.page_num,
                    'total_ms': int((time.time() - _t_page0) * 1000),
                    'detect_ms': det_ms,
                    'layout_ms': layout_ms,
                    'recog_ms': recog_ms,
                    'table_ms': table_ms,
                    'read_ms': read_ms,
                    'parse_ms': parse_ms,
                    'page_llm_ms': page_llm_ms,
                    'blocks': len(page.text_blocks),
                    'tables': len(page.tables),
                })
            except Exception:
                pass
        
        # Document-level LLM (QA / JSON extraction)
        doc_llm = {}
        if self.enable_llm and self.llm_client is not None and llm_task in ('qa', 'json', 'summary'):
            try:
                # Aggregate text from all pages in estimated reading order
                all_texts: List[str] = []
                for pg in result.pages:
                    order = pg.reading_order if pg.reading_order else list(range(len(pg.text_blocks)))
                    for idx in order:
                        if 0 <= idx < len(pg.text_blocks):
                            all_texts.append(pg.text_blocks[idx].text)
                doc_text_all = "\n".join(all_texts)[:12000]
                images_for_llm = None
                if os.getenv('DOCJA_LLM_USE_IMAGE', '0') == '1':
                    try:
                        # Use the first page image to aid summary/qa for gemma3
                        if len(images) > 0:
                            images_for_llm = [images[0]]
                    except Exception:
                        images_for_llm = None
                if 'llm' not in result.metadata:
                    result.metadata['llm'] = {}
                if llm_task == 'qa' and llm_question:
                    _t = time.time()
                    ans = self.llm_client.qa(doc_text_all, llm_question, images=images_for_llm)
                    # 回答がNone/空でも試行記録を残す（デバッグ/可観測性向上）
                    result.metadata['llm']['qa'] = {
                        'question': llm_question,
                        'answer': (ans if ans is not None else None)
                    }
                    doc_llm['qa_ms'] = int((time.time() - _t) * 1000)
                elif llm_task == 'summary':
                    _t = time.time()
                    summ = self.llm_client.summarize(doc_text_all, images=images_for_llm)
                    if summ:
                        result.metadata['llm']['summary'] = summ
                    doc_llm['summary_ms'] = int((time.time() - _t) * 1000)
                elif llm_task == 'json' and llm_schema:
                    _t = time.time()
                    js = self.llm_client.extract_json(doc_text_all, llm_schema, images=images_for_llm)
                    if js:
                        result.metadata['llm']['extraction'] = {'schema': llm_schema, 'result': js}
                    doc_llm['extract_ms'] = int((time.time() - _t) * 1000)
            except Exception as e:
                logger.warning(f"LLM document-level task failed: {e}")

        # Attach metrics into metadata
        try:
            result.metadata.setdefault('metrics', {})['pages'] = per_page_metrics
            if doc_llm:
                result.metadata.setdefault('metrics', {})['doc_llm'] = doc_llm
        except Exception:
            pass
        logger.info(f"Processing complete: {len(result.pages)} pages")
        # Emit metrics jsonl (best-effort)
        try:
            import json as _json
            mpath = Path('logs') / 'metrics.jsonl'
            mpath.parent.mkdir(exist_ok=True)
            with mpath.open('a', encoding='utf-8') as f:
                f.write(_json.dumps({
                    'file': str(input_path),
                    'pages': len(result.pages),
                    'per_page_ms': per_page_metrics,
                    'ts': time.time(),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return result

    def _process_pdf_direct_text(self, pdf_path: Path, max_pages: Optional[int]) -> DocumentResult:
        """Extract embedded text from PDF without OCR and build a DocumentResult.

        Uses pdfminer.six when available for per-line boxes; otherwise falls back
        to pypdfium2 and outputs a single block per page.
        """
        result = DocumentResult(str(pdf_path))
        result.metadata.setdefault('providers', {})['recognizer'] = 'pdf_direct_text'
        try:
            # Prefer pdfminer for detailed layout
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer, LTTextLine
            page_iter = extract_pages(str(pdf_path))
            for page_index, layout in enumerate(page_iter):
                if max_pages and page_index >= max_pages:
                    break
                width = int(getattr(layout, 'width', 2480))
                height = int(getattr(layout, 'height', 3508))
                page = Page(page_num=page_index + 1, width=width, height=height)
                for element in layout:
                    if isinstance(element, LTTextContainer):
                        for line in element:
                            if isinstance(line, LTTextLine):
                                text = (line.get_text() or '').rstrip('\n')
                                if not text.strip():
                                    continue
                                x0, y0, x1, y1 = line.bbox
                                # Convert PDF-space (bottom-left origin) -> image-space (top-left origin)
                                bbox = [float(x0), float(height - y1), float(x1), float(height - y0)]
                                block = TextBlock(bbox=bbox, text=text, confidence=1.0, block_type='paragraph')
                                page.add_text_block(block)
                result.add_page(page)
            return result
        except Exception as e:
            logger.warning(f"pdfminer not available or failed ({e}); falling back to pypdfium2 simple text")
        # Fallback: pypdfium2, one block per page
        try:
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(str(pdf_path))
            try:
                page_count = len(doc)
                for page_index in range(page_count):
                    if max_pages and page_index >= max_pages:
                        break
                    page_obj = doc.get_page(page_index)
                    try:
                        tp = page_obj.get_textpage()
                        try:
                            # get_text_range redirects to bounded when called without args in recent versions
                            text = tp.get_text_range()
                        except Exception:
                            try:
                                text = tp.get_text_bounded()
                            except Exception:
                                text = ''
                    finally:
                        # Close text page first (if created)
                        try:
                            tp.close()
                        except Exception:
                            pass
                    # Page metrics
                    try:
                        width = int(page_obj.get_width() or 595)
                        height = int(page_obj.get_height() or 842)
                    except Exception:
                        width, height = 595, 842
                    # Build page result
                    page = Page(page_num=page_index + 1, width=width, height=height)
                    block = TextBlock(
                        bbox=[0.0, 0.0, float(width), float(height)],
                        text=text or '',
                        confidence=1.0,
                        block_type='paragraph')
                    page.add_text_block(block)
                    result.add_page(page)
                    # Close page object
                    try:
                        page_obj.close()
                    except Exception:
                        pass
            finally:
                # Ensure document is closed last
                try:
                    doc.close()
                except Exception:
                    pass
            return result
        except Exception as ee:
            logger.error(f"pypdfium2 fallback failed: {ee}")
            # As a last resort, return an empty document with metadata
            return result
    
    def _load_pdf_images(self, pdf_path: Path, max_pages: Optional[int] = None) -> List[Image.Image]:
        """Load PDF pages as images using pypdfium2 (no poppler dependency)."""
        try:
            import pypdfium2 as pdfium
        except ImportError as e:
            logger.error("pypdfium2 is required for PDF rendering. Install with: pip install pypdfium2")
            raise

        images: List[Image.Image] = []
        try:
            # Ensure document resources are properly released
            doc = pdfium.PdfDocument(str(pdf_path))
            try:
                total = len(doc)
                count = min(total, max_pages) if max_pages else total
                # Render at approximately 300 DPI
                scale = 300.0 / 72.0
                for i in range(count):
                    page = doc[i]
                    bitmap = page.render(scale=scale)
                    pil = bitmap.to_pil()
                    images.append(pil.convert('RGB'))
            finally:
                try:
                    doc.close()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error loading PDF via pypdfium2: {e}")
            raise
        return images
    
    def _polygon_to_bbox(self, polygon: List[List[float]]) -> List[int]:
        """Convert polygon to bounding box."""
        polygon = np.array(polygon)
        x1 = int(np.min(polygon[:, 0]))
        y1 = int(np.min(polygon[:, 1]))
        x2 = int(np.max(polygon[:, 0]))
        y2 = int(np.max(polygon[:, 1]))
        return [x1, y1, x2, y2]
    
    def _get_block_type(self, region: Dict, layout_blocks: List[Dict]) -> str:
        """Determine block type from layout detection."""
        if not layout_blocks:
            return 'text'
        
        region_center = self._get_center(region['polygon'])
        
        for layout in layout_blocks:
            if self._point_in_bbox(region_center, layout['bbox']):
                return layout['label']
        
        return 'text'
    
    def _get_center(self, polygon: List[List[float]]) -> List[float]:
        """Get center point of polygon."""
        polygon = np.array(polygon)
        center_x = np.mean(polygon[:, 0])
        center_y = np.mean(polygon[:, 1])
        return [center_x, center_y]
    
    def _point_in_bbox(self, point: List[float], bbox: List[float]) -> bool:
        """Check if point is inside bbox."""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def _draw_overlays(self, image_bgr: np.ndarray,
                       text_regions: List[Dict[str, Any]],
                       layout_blocks: List[Dict[str, Any]],
                       tables: List[Table],
                       graphs: Optional[List[Dict[str, Any]]] = None,
                       charts: Optional[List[Dict[str, Any]]] = None) -> np.ndarray:
        import cv2
        import os
        vis_profile = (os.getenv('DOCJA_VIS_PROFILE', '') or '').lower().strip()
        # Raw mode: no overlays at all, return original image
        if vis_profile in ('none', 'off', 'raw', 'disable'):
            return image_bgr
        vis = image_bgr.copy()
        # Text regions (green polygon)
        for r in text_regions:
            poly = np.array(r.get('polygon', []), dtype=np.int32)
            if poly.size:
                cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        # Layout blocks (blue bbox)
        if layout_blocks:
            for lb in layout_blocks:
                x1, y1, x2, y2 = [int(v) for v in lb.get('bbox', [0, 0, 0, 0])]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Tables (magenta bbox)
        if tables:
            for t in tables:
                x1, y1, x2, y2 = [int(v) for v in t.bbox]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        # Graphs (flow) draw edges (yellow) and node boxes (cyan)
        if graphs:
            for g in graphs:
                if g.get('type') != 'flow':
                    continue
                centers = {}
                for node in g.get('nodes', []):
                    try:
                        x1, y1, x2, y2 = [int(v) for v in (node.get('bbox') or [0,0,0,0])]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        centers[node.get('id')] = (int((x1+x2)/2), int((y1+y2)/2))
                    except Exception:
                        pass
                for e in g.get('edges', []):
                    a = centers.get(e.get('source'))
                    b = centers.get(e.get('target'))
                    if a and b:
                        cv2.arrowedLine(vis, a, b, (0, 255, 255), 2, tipLength=0.02)
        # Charts (gantt) draw task bars (orange)
        if charts:
            for c in charts:
                if c.get('type') != 'gantt':
                    continue
                # Resolve visual profile and flags
                vis_profile = os.getenv('DOCJA_VIS_PROFILE', '').lower().strip()
                draw_columns = os.getenv('DOCJA_GANTT_DRAW_COLUMNS', '1') != '0'
                draw_tasks = os.getenv('DOCJA_GANTT_DRAW_TASKS', '1') != '0'
                # Clean profile disables debug-like overlays
                if vis_profile in ('clean', 'prod', 'production'):
                    draw_columns = False
                    draw_tasks = False
                # draw columns (light gray) limited to grid bbox if available
                cols = c.get('columns') or []
                gb = c.get('grid_bbox') or {}
                y_top = max(0, int(gb.get('y_top', 0)))
                y_bottom = min(vis.shape[0]-1, int(gb.get('y_bottom', vis.shape[0]-1)))
                if y_bottom <= y_top:
                    y_top, y_bottom = 0, vis.shape[0]-1
                if draw_columns:
                    for col in cols:
                        try:
                            x = int(col.get('x_px'))
                            cv2.line(vis, (x, y_top), (x, y_bottom), (180, 180, 180), 1)
                        except Exception:
                            pass
                # draw detected active cells as yellow dots and polylines
                cells = c.get('cells_active') or []
                # Only draw active cells guide if explicitly enabled
                if cells and os.getenv('DOCJA_GANTT_DRAW_ACTIVE', '0') == '1':
                    try:
                        # connect only contiguous cells (col_index diff == 1) within the same row and near-horizontal alignment
                        from collections import defaultdict
                        by_row = defaultdict(list)
                        for cell in cells:
                            by_row[int(cell.get('row_index', 0))].append(cell)
                        for ri, arr in by_row.items():
                            # sort by column index primarily, fall back to bbox x
                            arr = sorted(arr, key=lambda z: (int(z.get('col_index', 0)), int((z.get('bbox') or [0,0,0,0])[0])))
                            # draw dots
                            centers = []
                            for cell in arr:
                                x1, y1, x2, y2 = [int(v) for v in (cell.get('bbox') or [0,0,0,0])]
                                cx = int((x1 + x2) / 2)
                                cy = int((y1 + y2) / 2)
                                centers.append((cx, cy, int(cell.get('col_index', 0))))
                                cv2.circle(vis, (cx, cy), 2, (0, 255, 255), -1)
                            # connect only adjacent columns and if rows align (dy small)
                            if os.getenv('DOCJA_GANTT_DRAW_CELL_LINKS', '1') != '0':
                                for i in range(len(centers) - 1):
                                    xA, yA, cA = centers[i]
                                    xB, yB, cB = centers[i + 1]
                                    if (cB - cA) == 1 and abs(yA - yB) <= max(3, int(0.003 * vis.shape[0])):
                                        cv2.line(vis, (xA, yA), (xB, yB), (0, 255, 255), 1)
                    except Exception:
                        pass
                if draw_tasks:
                    for t in c.get('tasks', []):
                        try:
                            x1 = int(t.get('start_px_snapped', t.get('start_px', 0)))
                            x2 = int(t.get('end_px_snapped', t.get('end_px', 0)))
                            # Prefer explicit bbox from cell-based detection when available
                            tb = t.get('bbox')
                            if isinstance(tb, (list, tuple)) and len(tb) == 4:
                                y1 = int(tb[1])
                                y2 = int(tb[3])
                            else:
                                # Fallback: rough band from row index
                                y = 20 + int(t.get('row_index', 0)) * 24
                                y1, y2 = y, y + 12
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), -1)
                        except Exception:
                            pass
        return vis
