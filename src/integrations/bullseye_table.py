import os
import sys
import logging
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


class BullseyeTableRecognizer:
    """Adapter for table structure recognizer branded as bullseye.

    Provides `.recognize(image_rgb, boxes?)` API compatible with internal table pipeline.
    """

    def __init__(self,
                 device: str = 'cuda',
                 text_recognizer: Optional[Any] = None,
                 ocr_conf_threshold: float = 0.3,
                 blank_low_confidence: bool = True,
                 provider_label: str = 'bullseye-table',
                 fail_hard: bool = False):
        self.device = device
        self.text_recognizer = text_recognizer
        self.ocr_conf_threshold = float(ocr_conf_threshold)
        self.blank_low_confidence = bool(blank_low_confidence)
        self.provider_label = provider_label

        # Locate local upstream repo (bullseye)
        local_dir = (os.getenv('DOCJA_BULLSEYE_LOCAL_DIR', '')).strip()
        if not local_dir:
            # Prefer bullseye/src if present
            guess_b = os.path.join(os.getcwd(), 'bullseye', 'src')
            if os.path.exists(os.path.join(guess_b, 'bullseye', 'table_structure_recognizer.py')):
                local_dir = guess_b
        if local_dir and local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        _BZTable = None  # type: ignore
        try:
            import importlib.util
            from pathlib import Path
            import types
            pkg_path = str(Path(local_dir) / 'bullseye')
            if 'bullseye' not in sys.modules:
                b = types.ModuleType('bullseye')
                b.__path__ = [pkg_path]
                sys.modules['bullseye'] = b
            file_path = Path(local_dir) / 'bullseye' / 'table_structure_recognizer.py'
            spec = importlib.util.spec_from_file_location('bullseye.table_structure_recognizer', file_path)
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                sys.modules['bullseye.table_structure_recognizer'] = m
                spec.loader.exec_module(m)  # type: ignore
                try:
                    from ..logging_filters import sanitize_bullseye_loggers, install_bullseye_log_filter
                    install_bullseye_log_filter()
                    sanitize_bullseye_loggers()
                except Exception:
                    pass
                _BZTable = getattr(m, 'TableStructureRecognizer')
            else:
                raise ImportError('spec loader not available')
        except Exception:
            raise

        hf_repo = (os.getenv('DOCJA_BULLSEYE_TABLE_REPO', '')).strip()
        path_cfg = None
        if hf_repo:
            try:
                import tempfile
                from pathlib import Path
                d = tempfile.mkdtemp(prefix='docja_bullseye_tbl_')
                p = Path(d) / 'cfg.yaml'
                p.write_text(f"hf_hub_repo: '{hf_repo}'\n", encoding='utf-8')
                path_cfg = str(p)
            except Exception:
                path_cfg = None

        self._tbl = _BZTable(
            model_name='rtdetrv2',
            device='cuda' if (self.device == 'cuda') else 'cpu',
            from_pretrained=True,
            path_cfg=path_cfg,
        )

    @property
    def name(self) -> str:
        return self.provider_label

    def recognize(self, image_rgb: np.ndarray, boxes: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """Recognize table structures for given table boxes.

        Returns a list of table dicts: {bbox, html, markdown?, cells}
        """
        # Convert RGB->BGR for upstream recognizer
        img_bgr = image_rgb[..., ::-1].copy()

        if not boxes or not isinstance(boxes, list):
            # Upstream requires table_boxes; without it, skip gracefully
            logger.warning("bullseye table recognizer requires table_boxes; received none")
            return []

        # Call upstream recognizer, accepting either positional or keyword form
        try:
            outputs, _vis = self._tbl(img_bgr, boxes)
        except TypeError:
            outputs, _vis = self._tbl(img_bgr, table_boxes=boxes)

        tables: List[Dict[str, Any]] = []

        def _coerce_bbox(b: Any) -> Optional[List[float]]:
            if b is None:
                return None
            try:
                if hasattr(b, 'tolist'):
                    b = b.tolist()
                return [float(x) for x in b]
            except Exception:
                return None

        def _coerce_cell(obj: Any) -> Dict[str, Any]:
            try:
                # Pydantic v2
                if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
                    d = obj.model_dump()
                # Pydantic v1
                elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
                    d = obj.dict()
                elif isinstance(obj, dict):
                    d = obj
                else:
                    # Best-effort via attributes
                    d = {
                        'row': getattr(obj, 'row', None),
                        'col': getattr(obj, 'col', None),
                        'row_span': getattr(obj, 'row_span', getattr(obj, 'rowSpan', 1)) or 1,
                        'col_span': getattr(obj, 'col_span', getattr(obj, 'colSpan', 1)) or 1,
                        'is_header': bool(getattr(obj, 'is_header', getattr(obj, 'header', False))),
                        'content': getattr(obj, 'text', getattr(obj, 'content', '')) or ''
                    }
                    bx = getattr(obj, 'bbox', getattr(obj, 'box', None))
                    if bx is not None:
                        if hasattr(bx, 'tolist'):
                            bx = bx.tolist()
                        try:
                            d['bbox'] = [float(x) for x in bx]
                        except Exception:
                            pass
                # Normalize some keys
                if 'text' in d and 'content' not in d:
                    d['content'] = d.pop('text')
                # Ensure JSON-serializable types
                for k in list(d.keys()):
                    v = d[k]
                    if hasattr(v, 'tolist'):
                        d[k] = v.tolist()
                return d
            except Exception:
                return {'content': ''}

        def _emit_one(obj: Any, fallback_bbox: Optional[List[float]] = None):
            html = None
            markdown = None
            cells_any: Any = None
            bbox = fallback_bbox
            if isinstance(obj, dict):
                html = obj.get('html') or obj.get('contents')
                markdown = obj.get('markdown')
                cells_any = obj.get('cells')
                bbox = _coerce_bbox(obj.get('bbox') or obj.get('box') or bbox)
            else:
                html = getattr(obj, 'html', None) or getattr(obj, 'contents', None)
                markdown = getattr(obj, 'markdown', None)
                cells_any = getattr(obj, 'cells', None)
                bbox = _coerce_bbox(getattr(obj, 'bbox', None) or getattr(obj, 'box', None) or bbox)
            cells: List[Dict[str, Any]] = []
            if isinstance(cells_any, dict) and 'cells' in cells_any:
                cells_any = cells_any['cells']
            if isinstance(cells_any, list):
                cells = [_coerce_cell(c) for c in cells_any]
            tables.append({
                'bbox': bbox or [0.0, 0.0, 0.0, 0.0],
                'html': str(html) if html is not None else '',
                'markdown': str(markdown) if markdown is not None else '',
                'cells': cells,
            })

        # Normalize various output shapes to a list of tables
        if hasattr(outputs, 'tables') and isinstance(getattr(outputs, 'tables'), list):
            for idx, t in enumerate(outputs.tables):  # type: ignore[attr-defined]
                fb = boxes[idx] if idx < len(boxes) else None
                _emit_one(t, _coerce_bbox(fb))
        elif isinstance(outputs, (list, tuple)):
            for idx, t in enumerate(outputs):
                fb = boxes[idx] if idx < len(boxes) else None
                _emit_one(t, _coerce_bbox(fb))
        else:
            # Single object
            _emit_one(outputs, _coerce_bbox(boxes[0] if boxes else None))

        return tables
