import io
import os
import json
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


app = FastAPI(title="bullseye-core service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image_rgb(file: UploadFile) -> np.ndarray:
    b = file.file.read()
    img = Image.open(io.BytesIO(b)).convert('RGB')
    return np.array(img)


def _bbox_from_polygon(poly: List[List[float]]) -> List[float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _tmp_cfg_from_env(var_name: str) -> Optional[str]:
    repo = (os.getenv(var_name, '') or '').strip()
    if not repo:
        return None
    d = tempfile.mkdtemp(prefix=f"bullseye_{var_name.lower()}_")
    p = os.path.join(d, 'cfg.yaml')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(f"hf_hub_repo: '{repo}'\n")
    return p


# Lazy singletons
_detector = None
_recognizer = None
_layout = None
_table = None


def _device() -> str:
    d = (os.getenv('BULLSEYE_DEVICE') or '').strip()
    if d:
        return d
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _init_detector():
    global _detector
    if _detector is not None:
        return
    from bullseye.text_detector import TextDetector
    path_cfg = _tmp_cfg_from_env('DOCJA_BULLSEYE_DET_REPO')
    infer_onnx = (os.getenv('DOCJA_BULLSEYE_DET_ONNX','0') == '1')
    name = (os.getenv('DOCJA_BULLSEYE_DET_NAME') or 'dbnetv2')
    _detector = TextDetector(model_name=name, path_cfg=path_cfg, device=_device(), from_pretrained=True, infer_onnx=infer_onnx)


def _init_recognizer():
    global _recognizer
    if _recognizer is not None:
        return
    from bullseye.text_recognizer import TextRecognizer
    path_cfg = _tmp_cfg_from_env('DOCJA_BULLSEYE_REC_REPO')
    name = (os.getenv('DOCJA_BULLSEYE_REC_NAME') or 'parseqv2')
    _recognizer = TextRecognizer(model_name=name, path_cfg=path_cfg, device=_device(), from_pretrained=True)


def _init_layout():
    global _layout
    if _layout is not None:
        return
    from bullseye.layout_parser import LayoutParser
    path_cfg = _tmp_cfg_from_env('DOCJA_BULLSEYE_LAYOUT_REPO')
    name = (os.getenv('DOCJA_BULLSEYE_LAYOUT_NAME') or 'rtdetrv2v2')
    _layout = LayoutParser(model_name=name, path_cfg=path_cfg, device=_device(), from_pretrained=True)


def _init_table():
    global _table
    if _table is not None:
        return
    from bullseye.table_structure_recognizer import TableStructureRecognizer
    path_cfg = _tmp_cfg_from_env('DOCJA_BULLSEYE_TABLE_REPO')
    name = (os.getenv('DOCJA_BULLSEYE_TABLE_NAME') or 'rtdetrv2')
    _table = TableStructureRecognizer(model_name=name, path_cfg=path_cfg, device=_device(), from_pretrained=True)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": _device()}


@app.post("/v1/det/detect")
def det_detect(file: UploadFile = File(...)):
    _init_detector()
    img = _read_image_rgb(file)
    # TextDetector expects BGR
    img_bgr = img[:, :, ::-1]
    results, _vis = _detector(img_bgr)  # TextDetectorSchema
    polys = results.points
    scores = results.scores
    out = []
    for i, poly in enumerate(polys):
        p = [[float(x), float(y)] for x, y in poly]
        out.append({
            'polygon': p,
            'bbox': _bbox_from_polygon(p),
            'confidence': float(scores[i]) if i < len(scores) else 0.0,
            'type': 'text_line'
        })
    return JSONResponse({"text_regions": out})


@app.post("/v1/rec/recognize")
def rec_recognize(file: UploadFile = File(...)):
    _init_recognizer()
    img = _read_image_rgb(file)
    img_bgr = img[:, :, ::-1]
    res, _vis = _recognizer(img_bgr)  # TextRecognizerSchema
    contents = res.contents or []
    scores = res.scores or []
    text = contents[0] if contents else ""
    conf = float(scores[0]) if scores else 0.0
    return JSONResponse({"text": text, "confidence": conf})


@app.post("/v1/layout/detect")
def layout_detect(file: UploadFile = File(...)):
    _init_layout()
    img = _read_image_rgb(file)
    img_bgr = img[:, :, ::-1]
    res, _vis = _layout(img_bgr)  # LayoutParserSchema
    elements: List[Dict[str, Any]] = []
    def _append(arr: List[Any], typ: str):
        for el in (arr or []):
            bbox = [float(v) for v in (el.box or [0,0,0,0])]
            elements.append({'type': typ, 'bbox': bbox, 'confidence': float(getattr(el,'score',0.0))})
    _append(getattr(res, 'paragraphs', []), 'paragraph')
    _append(getattr(res, 'figures', []), 'figure')
    # Tables as layout boxes (structure is handled in table endpoint)
    _append(getattr(res, 'tables', []), 'table')
    return JSONResponse({"layout_elements": elements})


@app.post("/v1/table/recognize")
def table_recognize(file: UploadFile = File(...), options: Optional[str] = Form(None)):
    _init_table()
    img = _read_image_rgb(file)
    img_bgr = img[:, :, ::-1]
    boxes = None
    if options:
        try:
            opt = json.loads(options)
            boxes = opt.get('boxes')
        except Exception:
            boxes = None
    boxes = boxes or []
    # Convert to int boxes expected by TSR (x1,y1,x2,y2)
    boxes_xyxy = [[int(v) for v in b] for b in boxes]
    results, _vis = _table(img_bgr, boxes_xyxy)
    out = []
    for t in results:
        out.append({
            'bbox': [float(v) for v in t.box],
            'cells': [
                {
                    'col': c.col,
                    'row': c.row,
                    'col_span': c.col_span,
                    'row_span': c.row_span,
                    'bbox': [float(v) for v in c.box],
                    'content': c.contents or ''
                } for c in t.cells
            ],
            'html': '',
            'markdown': ''
        })
    return JSONResponse({"tables": out})


@app.post("/v1/table/detect_and_recognize")
def table_detect_and_recognize(file: UploadFile = File(...)):
    # Basic strategy: layout->table boxes->recognize
    _init_layout(); _init_table()
    img = _read_image_rgb(file)
    img_bgr = img[:, :, ::-1]
    lay, _ = _layout(img_bgr)
    table_boxes = [list(map(float, el.box)) for el in getattr(lay, 'tables', [])]
    table_boxes = [[int(v) for v in b] for b in table_boxes]
    results, _vis = _table(img_bgr, table_boxes)
    out = []
    for t in results:
        out.append({
            'bbox': [float(v) for v in t.box],
            'cells': [
                {
                    'col': c.col,
                    'row': c.row,
                    'col_span': c.col_span,
                    'row_span': c.row_span,
                    'bbox': [float(v) for v in c.box],
                    'content': c.contents or ''
                } for c in t.cells
            ],
            'html': '',
            'markdown': ''
        })
    return JSONResponse({"tables": out})

