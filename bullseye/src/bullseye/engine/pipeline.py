from __future__ import annotations

from typing import Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..core.interfaces import (
    TextDetectorIF,
    TextRecognizerIF,
    LayoutAnalyzerIF,
)


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
