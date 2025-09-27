from __future__ import annotations

from typing import Optional

from ..core.interfaces import (
    TextDetectorIF,
    TextRecognizerIF,
    LayoutAnalyzerIF,
    TableStructureRecognizerIF,
)
from ..providers.base import BaseProvider, ProviderConfig


class _Detector(TextDetectorIF):
    def __init__(self, device: str = "cuda", visualize: bool = False, infer_onnx: bool = False):
        from ..text_detector import TextDetector  # lazy import

        self.impl = TextDetector(device=device, visualize=visualize, infer_onnx=infer_onnx)

    def __call__(self, img):
        return self.impl(img)


class _Recognizer(TextRecognizerIF):
    def __init__(self, device: str = "cuda", visualize: bool = False, infer_onnx: bool = False):
        from ..text_recognizer import TextRecognizer  # lazy import

        self.impl = TextRecognizer(device=device, visualize=visualize, infer_onnx=infer_onnx)

    def __call__(self, img, polygons=None, vis=None):
        return self.impl(img, polygons, vis)


class _Layout(LayoutAnalyzerIF):
    def __init__(self, device: str = "cuda", visualize: bool = False):
        from ..layout_analyzer import LayoutAnalyzer  # lazy import

        self.impl = LayoutAnalyzer(configs={"layout_parser": {"device": device, "visualize": visualize},
                                            "table_structure_recognizer": {"device": device, "visualize": visualize}})

    def __call__(self, img):
        return self.impl(img)


class _Table(TableStructureRecognizerIF):
    def __init__(self, device: str = "cuda", visualize: bool = False):
        from ..layout_analyzer import LayoutAnalyzer  # lazy import, table is inside LayoutAnalyzer
        # direct table call is optional; prefer using LayoutAnalyzer
        self.impl = LayoutAnalyzer(configs={"layout_parser": {"device": device, "visualize": visualize},
                                            "table_structure_recognizer": {"device": device, "visualize": visualize}})

    def __call__(self, img):
        return self.impl(img)


class YomiProvider(BaseProvider):
    def __init__(self, cfg: Optional[ProviderConfig] = None) -> None:
        super().__init__(cfg)
        self.detector = _Detector(device=self.cfg.device, visualize=self.cfg.visualize, infer_onnx=self.cfg.infer_onnx)
        self.recognizer = _Recognizer(device=self.cfg.device, visualize=self.cfg.visualize, infer_onnx=self.cfg.infer_onnx)
        self.layout = _Layout(device=self.cfg.device, visualize=self.cfg.visualize)
        self.table = _Table(device=self.cfg.device, visualize=self.cfg.visualize)

