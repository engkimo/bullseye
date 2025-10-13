import os
import sys
import logging
from typing import List, Dict, Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


class BullseyeDbnetDetector:
    """Adapter for DBNet/DBNetv2 branded as bullseye.

    - Accepts DOCJA_BULLSEYE_* envs.
    - Prefers local repo/weights when available (models/bullseye/*).
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 device: str = 'cuda',
                 provider_label: str = 'bullseye-dbnet',
                 infer_onnx: Optional[bool] = None,
                 fail_hard: bool = False):
        self.device = device
        self.provider_label = provider_label
        self.model_name = model_name or 'dbnetv2'
        self.infer_onnx = bool(int(os.getenv('DOCJA_BULLSEYE_DET_ONNX', '0'))) if infer_onnx is None else infer_onnx

        # Locate local upstream package (bullseye)
        local_dir = (os.getenv('DOCJA_BULLSEYE_LOCAL_DIR', '')).strip()
        if not local_dir:
            guess = os.path.join(os.getcwd(), 'bullseye', 'src')
            if os.path.exists(os.path.join(guess, 'bullseye', 'text_detector.py')):
                local_dir = guess
        if local_dir and local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        _BZTextDetector = None  # type: ignore
        try:
            # Direct file import first to avoid package __init__
            import importlib.util
            from pathlib import Path
            import types
            # Create stub package pointing to bullseye src
            pkg_path = str(Path(local_dir) / 'bullseye')
            if 'bullseye' not in sys.modules:
                b = types.ModuleType('bullseye')
                b.__path__ = [pkg_path]
                sys.modules['bullseye'] = b
            # Load under bullseye.* to make logger names bullseye.*
            file_path = Path(local_dir) / 'bullseye' / 'text_detector.py'
            spec = importlib.util.spec_from_file_location('bullseye.text_detector', file_path)
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                sys.modules['bullseye.text_detector'] = m
                spec.loader.exec_module(m)  # type: ignore
                # Remove upstream handlers immediately to avoid duplicate logs
                try:
                    from ..logging_filters import sanitize_bullseye_loggers, install_bullseye_log_filter
                    install_bullseye_log_filter()
                    sanitize_bullseye_loggers()
                except Exception:
                    pass
                _BZTextDetector = getattr(m, 'TextDetector')
            else:
                raise ImportError('spec loader not available')
        except Exception:
            raise

        # Optional local weights override
        # Prefer explicit HF repo id over local paths
        hf_repo = (os.getenv('DOCJA_BULLSEYE_DET_REPO', '')).strip()
        path_cfg = None
        if hf_repo:
            try:
                import tempfile
                from pathlib import Path
                d = tempfile.mkdtemp(prefix='docja_bullseye_det_')
                p = Path(d) / 'cfg.yaml'
                p.write_text(f"hf_hub_repo: '{hf_repo}'\n", encoding='utf-8')
                path_cfg = str(p)
            except Exception:
                path_cfg = None
        if path_cfg is None:
            try:
                from pathlib import Path as _P
                local_weights = _P.cwd() / 'models' / 'bullseye' / 'det-dbnet-v2'
                if local_weights.exists():
                    import tempfile as _tmp
                    d = _tmp.mkdtemp(prefix='docja_bullseye_det_')
                    cfgp = _P(d) / 'cfg.yaml'
                    cfgp.write_text(f"hf_hub_repo: '{str(local_weights)}'\n", encoding='utf-8')
                    path_cfg = str(cfgp)
            except Exception:
                pass

        self._det = _BZTextDetector(
            model_name=self.model_name,
            device='cuda' if (self.device == 'cuda') else 'cpu',
            from_pretrained=True,
            path_cfg=path_cfg,
            infer_onnx=self.infer_onnx,
        )

    @property
    def name(self) -> str:
        return f"{self.provider_label}:{self.model_name}"

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        # RGB -> BGR
        if image_rgb.ndim == 2:
            img_bgr = np.stack([image_rgb] * 3, axis=-1)
        else:
            img_bgr = image_rgb[..., ::-1].copy()

        try:
            outputs, _vis = self._det(img_bgr)
        except Exception as e:
            logger.warning(f"bullseye DBNet detect failed: {e}")
            return []

        points = getattr(outputs, 'points', None) or getattr(outputs, 'contents', None) or []
        scores = getattr(outputs, 'scores', [])
        regions: List[Dict[str, Any]] = []
        for i, pts in enumerate(points):
            poly = pts
            if isinstance(poly, (list, tuple)) and poly and not isinstance(poly[0], (list, tuple)):
                arr = list(poly)
                poly = [[float(arr[j]), float(arr[j + 1])] for j in range(0, len(arr), 2)]
            else:
                poly = [[float(x), float(y)] for x, y in poly]
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            conf = float(scores[i]) if i < len(scores) else 0.0
            regions.append({
                'polygon': poly,
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'type': 'text_line'
            })
        return regions
