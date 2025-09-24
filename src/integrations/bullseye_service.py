import io
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


logger = logging.getLogger(__name__)


def _b64_image_bytes(image_rgb: np.ndarray, fmt: str = 'PNG') -> bytes:
    img = Image.fromarray(image_rgb.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class BullseyeServiceClient:
    """HTTP client for bullseye-core private service.

    Expects endpoints (suggestion; actual service may provide aliases):
    - POST {base}/v1/det/detect
    - POST {base}/v1/rec/recognize
    - POST {base}/v1/layout/detect
    - POST {base}/v1/table/recognize (with boxes) or {base}/v1/table/detect_and_recognize
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 timeout: Optional[float] = None,
                 provider_label: Optional[str] = None):
        self.base_url = (base_url or os.getenv('DOCJA_BULLSEYE_ENDPOINT') or 'http://localhost:8088').rstrip('/')
        self.timeout = float(os.getenv('DOCJA_BULLSEYE_TIMEOUT', str(timeout or 45)))
        self.provider_label = provider_label or (os.getenv('DOCJA_PROVIDER_ALIAS_LABEL') or 'bullseye-svc')

    # Low-level helpers
    def _post_image(self, path: str, image_rgb: np.ndarray, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        files = {
            'file': ('image.png', _b64_image_bytes(image_rgb), 'image/png')
        }
        data = {}
        if extra:
            data['options'] = json.dumps(extra, ensure_ascii=False)
        logger.debug(f"POST {url} (timeout={self.timeout})")
        r = requests.post(url, files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        logger.debug(f"POST {url} json (timeout={self.timeout})")
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # High-level API
    def detect_text(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        res = self._post_image('/v1/det/detect', image_rgb)
        regions = res.get('text_regions') or []
        # Normalize keys
        out: List[Dict[str, Any]] = []
        for rg in regions:
            poly = rg.get('polygon') or []
            bbox = rg.get('bbox') or []
            conf = float(rg.get('confidence', 0.0))
            out.append({'polygon': poly, 'bbox': bbox, 'confidence': conf, 'type': 'text_line'})
        return out

    def recognize_text(self, image_rgb: np.ndarray) -> Tuple[str, float]:
        res = self._post_image('/v1/rec/recognize', image_rgb)
        return str(res.get('text') or ''), float(res.get('confidence') or 0.0)

    def detect_layout(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        res = self._post_image('/v1/layout/detect', image_rgb)
        els = res.get('layout_elements') or []
        out: List[Dict[str, Any]] = []
        for el in els:
            bbox = [float(v) for v in (el.get('bbox') or [0, 0, 0, 0])]
            typ = el.get('type') or el.get('label') or 'paragraph'
            conf = float(el.get('confidence') or 0.0)
            out.append({'bbox': bbox, 'type': str(typ), 'confidence': conf})
        return out

    def recognize_tables(self, image_rgb: np.ndarray, table_boxes: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        if table_boxes:
            payload = {
                'boxes': table_boxes
            }
            # Send image + boxes
            url = f"{self.base_url}/v1/table/recognize"
            files = {'file': ('image.png', _b64_image_bytes(image_rgb), 'image/png')}
            data = {'options': json.dumps(payload, ensure_ascii=False)}
            r = requests.post(url, files=files, data=data, timeout=self.timeout)
            r.raise_for_status()
            res = r.json()
        else:
            res = self._post_image('/v1/table/detect_and_recognize', image_rgb)
        tables = res.get('tables') or []
        out: List[Dict[str, Any]] = []
        for t in tables:
            out.append({
                'bbox': [float(v) for v in (t.get('bbox') or [0, 0, 0, 0])],
                'cells': t.get('cells') or [],
                'html': t.get('html') or '',
                'markdown': t.get('markdown') or ''
            })
        return out

    # Provider labels
    def label(self, component: str) -> str:
        return f"{self.provider_label}:{component}"


class BullseyeServiceDetector:
    def __init__(self, client: Optional[BullseyeServiceClient] = None):
        self.client = client or BullseyeServiceClient()
        self.provider_label = self.client.label('det')

    def name(self) -> str:
        return self.provider_label

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        return self.client.detect_text(image_rgb)


class BullseyeServiceRecognizer:
    def __init__(self, client: Optional[BullseyeServiceClient] = None):
        self.client = client or BullseyeServiceClient()
        self.provider_label = self.client.label('rec')

    def name(self) -> str:
        return self.provider_label

    def recognize(self, image_rgb: np.ndarray) -> Tuple[str, float]:
        return self.client.recognize_text(image_rgb)


class BullseyeServiceLayout:
    def __init__(self, client: Optional[BullseyeServiceClient] = None):
        self.client = client or BullseyeServiceClient()
        self.provider_label = self.client.label('layout')

    def name(self) -> str:
        return self.provider_label

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        return self.client.detect_layout(image_rgb)


class BullseyeServiceTable:
    def __init__(self, client: Optional[BullseyeServiceClient] = None, text_recognizer: Any = None):
        self.client = client or BullseyeServiceClient()
        self.provider_label = self.client.label('table')
        self.text_recognizer = text_recognizer

    def name(self) -> str:
        return self.provider_label

    def recognize(self, image_rgb: np.ndarray, table_boxes: List[List[float]]) -> List[Dict[str, Any]]:
        return self.client.recognize_tables(image_rgb, table_boxes)

    def detect_and_recognize(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        return self.client.recognize_tables(image_rgb, None)

