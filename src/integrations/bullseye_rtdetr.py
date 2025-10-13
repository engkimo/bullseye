import os
import sys
import logging
from typing import List, Dict, Any

import numpy as np


logger = logging.getLogger(__name__)


class BullseyeRtDetrLayout:
    """Adapter for RT-DETRv2 layout detector branded as bullseye.

    - Uses transformers HF path when available unless disabled by env.
    - Local fallback via bullseye LayoutParser when configured.
    """

    def __init__(self, model_id: str, device: str = 'cuda', no_hf: bool = False):
        self.model_id = model_id
        self.device = device
        self.provider_label = 'bullseye-layout'
        self.model_label = model_id or 'rtdetrv2'
        self.hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self._local_parser = None
        self.id2label = {}
        env_no_hf = no_hf or (os.getenv('DOCJA_NO_HF', '0') == '1')

        if env_no_hf:
            try:
                import importlib.util
                from pathlib import Path
                local_dir = (os.getenv('DOCJA_BULLSEYE_LOCAL_DIR', '')).strip()
                if not local_dir:
                    # Prefer bullseye/src
                    guess_b = Path.cwd() / 'bullseye' / 'src'
                    if (guess_b / 'bullseye' / 'layout_parser.py').exists():
                        local_dir = str(guess_b)
                if not local_dir:
                    raise RuntimeError('DOCJA_BULLSEYE_LOCAL_DIR not set and local bullseye not found')
                import types
                pkg_path = str(Path(local_dir) / 'bullseye')
                if 'bullseye' not in sys.modules:
                    b = types.ModuleType('bullseye')
                    b.__path__ = [pkg_path]
                    sys.modules['bullseye'] = b
                file_path = Path(local_dir) / 'bullseye' / 'layout_parser.py'
                spec = importlib.util.spec_from_file_location('bullseye.layout_parser', file_path)
                if not (spec and spec.loader):
                    raise ImportError('spec loader not available')
                m = importlib.util.module_from_spec(spec)
                sys.modules['bullseye.layout_parser'] = m
                spec.loader.exec_module(m)  # type: ignore
                try:
                    from ..logging_filters import sanitize_bullseye_loggers, install_bullseye_log_filter
                    install_bullseye_log_filter()
                    sanitize_bullseye_loggers()
                except Exception:
                    pass
                _LayoutParser = getattr(m, 'LayoutParser')

                # Prefer explicit HF repo id
                hf_repo = (os.getenv('DOCJA_BULLSEYE_LAYOUT_REPO', '')).strip()
                path_cfg = None
                if hf_repo:
                    import tempfile
                    d = tempfile.mkdtemp(prefix='docja_bullseye_layout_')
                    cfgp = Path(d) / 'cfg.yaml'
                    cfgp.write_text(f"hf_hub_repo: '{hf_repo}'\n", encoding='utf-8')
                    path_cfg = str(cfgp)
                # Force local weights if no explicit repo is given
                if path_cfg is None:
                    try:
                        from pathlib import Path as _P
                        local_weights = _P.cwd() / 'models' / 'bullseye' / 'layout-rtdetrv2-v2'
                        if local_weights.exists():
                            import tempfile as _tmp
                            d = _tmp.mkdtemp(prefix='docja_bullseye_layout_')
                            cfgp = _P(d) / 'cfg.yaml'
                            cfgp.write_text(f"hf_hub_repo: '{str(local_weights)}'\n", encoding='utf-8')
                            path_cfg = str(cfgp)
                    except Exception:
                        pass
                self._local_parser = _LayoutParser(
                    model_name='rtdetrv2v2',
                    device='cuda' if (self.device == 'cuda') else 'cpu',
                    from_pretrained=True,
                    path_cfg=path_cfg,
                )
                import torch
                self.torch = torch
                self.model_label = 'local-rtdetrv2'
                return
            except Exception as e:
                raise RuntimeError(f'Local bullseye layout required but failed: {e}')

        # HF path
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            logger.info(f"Loading HF RT-DETRv2 layout model: {self.model_id}")
            common = dict(trust_remote_code=True, token=self.hf_token)
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, **common)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_id, **common)
            import torch
            self.torch = torch
            self.model = self.model.to(self.device if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
            self.model.eval()
            cfg = getattr(self.model, 'config', None)
            self.id2label = getattr(cfg, 'id2label', {}) or {}
            return
        except Exception as e:
            logger.warning(f"HF RT-DETRv2 load failed: {e}; trying local LayoutParser")
        # Local fallback
        try:
            import importlib.util
            from pathlib import Path
            local_dir = (os.getenv('DOCJA_BULLSEYE_LOCAL_DIR', '')).strip()
            if not local_dir:
                guess_b = Path.cwd() / 'bullseye' / 'src'
                if (guess_b / 'bullseye' / 'layout_parser.py').exists():
                    local_dir = str(guess_b)
            import types
            pkg_path = str(Path(local_dir) / 'bullseye')
            if 'bullseye' not in sys.modules:
                b = types.ModuleType('bullseye')
                b.__path__ = [pkg_path]
                sys.modules['bullseye'] = b
            file_path = Path(local_dir) / 'bullseye' / 'layout_parser.py'
            spec = importlib.util.spec_from_file_location('bullseye.layout_parser', file_path)
            if spec and spec.loader:
                m = importlib.util.module_from_spec(spec)
                sys.modules['bullseye.layout_parser'] = m
                spec.loader.exec_module(m)  # type: ignore
                try:
                    from ..logging_filters import sanitize_bullseye_loggers, install_bullseye_log_filter
                    install_bullseye_log_filter()
                    sanitize_bullseye_loggers()
                except Exception:
                    pass
                _LayoutParser = getattr(m, 'LayoutParser')
            else:
                raise ImportError('spec loader not available')
            hf_repo = (os.getenv('DOCJA_BULLSEYE_LAYOUT_REPO', '')).strip()
            path_cfg = None
            if hf_repo:
                import tempfile
                d = tempfile.mkdtemp(prefix='docja_bullseye_layout_')
                cfgp = Path(d) / 'cfg.yaml'
                cfgp.write_text(f"hf_hub_repo: '{hf_repo}'\n", encoding='utf-8')
                path_cfg = str(cfgp)
            if path_cfg is None:
                try:
                    from pathlib import Path as _P
                    local_weights = _P.cwd() / 'models' / 'bullseye' / 'layout-rtdetrv2-v2'
                    if local_weights.exists():
                        import tempfile as _tmp
                        d = _tmp.mkdtemp(prefix='docja_bullseye_layout_')
                        cfgp = _P(d) / 'cfg.yaml'
                        cfgp.write_text(f"hf_hub_repo: '{str(local_weights)}'\n", encoding='utf-8')
                        path_cfg = str(cfgp)
                except Exception:
                    pass
            self._local_parser = _LayoutParser(
                model_name='rtdetrv2v2',
                device='cuda' if (self.device == 'cuda') else 'cpu',
                from_pretrained=True,
                path_cfg=path_cfg,
            )
            import torch
            self.torch = torch
            self.model_label = 'local-rtdetrv2'
        except Exception as e:
            raise RuntimeError("Neither HF nor local bullseye layout could be loaded") from e

    @property
    def name(self) -> str:
        try:
            return f"{self.provider_label}:{self.model_label}"
        except Exception:
            return self.provider_label

    def detect(self, image: np.ndarray, conf_thresh: float = 0.3) -> List[Dict[str, Any]]:
        if self._local_parser is None:
            h, w = image.shape[:2]
            inputs = self.processor(images=image, return_tensors='pt')
            inputs = {k: (v.to(self.model.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            try:
                processed = self.processor.post_process_object_detection(outputs, threshold=conf_thresh, target_sizes=[(h, w)])
            except Exception:
                processed = self.processor.post_process(outputs, threshold=conf_thresh, target_sizes=[(h, w)])
            pr = processed[0]
            boxes = pr.get('boxes', [])
            scores = pr.get('scores', [])
            labels = pr.get('labels', [])
            res: List[Dict[str, Any]] = []
            for i in range(len(scores)):
                b = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])
                score = float(scores[i])
                lab_id = int(labels[i])
                lab = self.id2label.get(lab_id, str(lab_id))
                res.append({'bbox': b, 'label': lab, 'confidence': score})
            return res
        # local path (expects BGR)
        img_bgr = image[..., ::-1].copy()
        outputs, _vis = self._local_parser(img_bgr)
        res: List[Dict[str, Any]] = []
        for lab, elements in (('paragraph', getattr(outputs, 'paragraphs', [])),
                              ('table', getattr(outputs, 'tables', [])),
                              ('figure', getattr(outputs, 'figures', []))):
            for el in elements:
                box = el.get('box') if isinstance(el, dict) else getattr(el, 'box', None)
                score = el.get('score') if isinstance(el, dict) else getattr(el, 'score', 0.0)
                if box is not None:
                    res.append({'bbox': [float(v) for v in box], 'label': lab, 'confidence': float(score or 0.0)})
        return res
