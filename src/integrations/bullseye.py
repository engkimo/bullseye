import os
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import importlib.util
import types
from pathlib import Path


logger = logging.getLogger(__name__)


def self_property(func):
    return property(func)


class BullseyeParseqRecognizer:
    """Adapter for PARSeq recognizers branded as "bullseye".

    - Prefers local repo specified by DOCJA_BULLSEYE_LOCAL_DIR
    - Prefers local weights under models/bullseye
    - Falls back to Hugging Face if allowed
    """

    def __init__(self,
                 model_id: Optional[str] = None,
                 device: str = "cuda",
                 provider_label: str = "bullseye-parseq",
                 no_hf: bool = False,
                 fail_hard: bool = False):
        self.model_id = model_id
        self.device = device
        self.provider_label = provider_label

        try:
            from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoConfig
            self._AutoProcessor = AutoProcessor
            self._AutoTokenizer = AutoTokenizer
            self._AutoModel = AutoModel
            self._AutoConfig = AutoConfig
        except Exception as e:
            raise RuntimeError(
                "transformers is required for BullseyeParseqRecognizer. Install with: pip install transformers timm einops"
            ) from e

        # HF token & flags
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        # Honor both old/new flags
        self.no_hf = no_hf or (os.getenv('DOCJA_NO_HF', '0') == '1')
        self.fail_hard = fail_hard or (os.getenv('DOCJA_NO_INTERNAL_FALLBACK', '0') == '1')

        # Try local package first
        self._recognizer_local = None
        try:
            self._try_load_local_package()
        except Exception as e:
            logger.debug(f"Local bullseye load skipped: {e}")

        if self._recognizer_local is not None:
            logger.info("Using local bullseye TextRecognizer (PARSeq)")
            return

        if self.no_hf:
            if self.fail_hard:
                raise RuntimeError("HF path disabled and local bullseye recognizer not available")
            raise RuntimeError("Local bullseye recognizer not available and HF disabled")

        # HF fallback
        logger.info(f"Loading HF model: {self.model_id} (provider={self.provider_label})")
        common_kwargs = dict(trust_remote_code=True, token=self.hf_token)

        self.processor = None
        self.tokenizer = None
        try:
            self.processor = self._AutoProcessor.from_pretrained(self.model_id, **common_kwargs)
        except Exception:
            try:
                self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_id, **common_kwargs)
            except Exception:
                pass

        model_loaded = False
        try:
            self.model = self._AutoModel.from_pretrained(self.model_id, **common_kwargs)
            model_loaded = True
        except Exception as e:
            logger.warning(f"AutoModel load failed: {e}. Trying snapshot + dynamic import...")
            try:
                self._snapshot_and_import()
                model_loaded = True
            except Exception as ee:
                raise RuntimeError(f"Failed to load HF model {self.model_id} via AutoModel and snapshot fallback: {ee}") from ee

        try:
            import torch
            self.torch = torch
            self.model = self.model.to(self.device if torch.cuda.is_available() and self.device == 'cuda' else 'cpu')
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"PyTorch required for HF model inference: {e}") from e

    def _make_tmp_cfg(self, repo_or_dir: str) -> Optional[str]:
        try:
            import tempfile
            d = tempfile.mkdtemp(prefix='docja_bullseye_')
            p = Path(d) / 'cfg.yaml'
            p.write_text(f"hf_hub_repo: '{repo_or_dir}'\n", encoding='utf-8')
            return str(p)
        except Exception:
            return None

    def _try_load_local_package(self):
        import sys
        # detect local repo (new var first)
        local_dir = (os.getenv('DOCJA_BULLSEYE_LOCAL_DIR', '')).strip()
        if not local_dir:
            # Prefer ./bullseye/src if present
            guess_b = Path.cwd() / 'bullseye' / 'src'
            if (guess_b / 'bullseye' / 'text_recognizer.py').exists():
                local_dir = str(guess_b)
        if not local_dir or not os.path.isdir(local_dir):
            return
        try:
            import sys as _sys
            # Create stub packages for both names (map to whichever exists under local_dir)
            pkg_path = str(Path(local_dir) / 'bullseye')
            if 'bullseye' not in _sys.modules:
                b = types.ModuleType('bullseye')
                b.__path__ = [pkg_path]
                _sys.modules['bullseye'] = b
            file_path = Path(local_dir) / 'bullseye' / 'text_recognizer.py'
            spec = importlib.util.spec_from_file_location('bullseye.text_recognizer', file_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                _sys.modules['bullseye.text_recognizer'] = mod
                spec.loader.exec_module(mod)  # type: ignore
                try:
                    from ..logging_filters import sanitize_bullseye_loggers, install_bullseye_log_filter
                    install_bullseye_log_filter()
                    sanitize_bullseye_loggers()
                except Exception:
                    pass
                BZTextRecognizer = getattr(mod, 'TextRecognizer')
            else:
                raise ImportError("spec loader not available")
        except Exception:
            if local_dir not in sys.path:
                sys.path.insert(0, local_dir)
            from bullseye.text_recognizer import TextRecognizer as BZTextRecognizer  # type: ignore
        # Allow explicit override via ENV, else infer from model_id
        override_name = (os.getenv('DOCJA_BULLSEYE_REC_NAME', '')).strip()
        if override_name:
            model_name = override_name
        else:
            mid = (self.model_id or '')
            model_name = 'parseqv2' if ('middle-v2' in mid or 'v2' in mid.lower()) else 'parseq'
        # local weights override (new var first)
        # Prefer explicit HF repo id; local dir path is not accepted by HF validators
        hf_repo = (os.getenv('DOCJA_BULLSEYE_REC_REPO', '')).strip()
        path_cfg = self._make_tmp_cfg(hf_repo) if hf_repo else None
        self._recognizer_local = BZTextRecognizer(
            model_name=model_name,
            device='cuda' if (self.device == 'cuda') else 'cpu',
            from_pretrained=True,
            path_cfg=path_cfg,
        )

    def _snapshot_and_import(self):
        from huggingface_hub import snapshot_download
        import importlib.util
        import sys

        cache_dir = os.getenv('DOCJA_BULLSEYE_CACHE', '') or None
        local_override = (os.getenv('DOCJA_BULLSEYE_LOCAL_DIR', '')).strip()
        if local_override and os.path.isdir(local_override):
            local_dir = Path(local_override)
            logger.info(f"Using local repo for PARSeq: {local_dir}")
        else:
            local_dir = Path(snapshot_download(self.model_id, token=self.hf_token, local_dir=cache_dir))

        if str(local_dir) not in sys.path:
            sys.path.insert(0, str(local_dir))

        entry = (os.getenv('DOCJA_BULLSEYE_ENTRY', '')).strip()
        candidates = []
        if entry and ':' in entry:
            candidates.append(tuple(entry.split(':', 1)))
        candidates.extend([
            ('modeling_parseq', 'PARSeq'),
            ('parseq', 'PARSeq'),
            ('model', 'PARSeq'),
            ('inference', 'Model'),
            ('recognizer', 'PARSeq'),
            ('recognizer', 'Recognizer'),
        ])

        model = None
        last_err: Optional[Exception] = None
        for mod_name, cls_name in candidates:
            try:
                module_spec = None
                try:
                    module_spec = importlib.util.find_spec(mod_name)
                except Exception:
                    module_spec = None
                if module_spec is None:
                    for py in local_dir.rglob('*.py'):
                        if py.stem.lower() == mod_name.lower():
                            spec = importlib.util.spec_from_file_location(mod_name, py)
                            if spec and spec.loader:
                                m = importlib.util.module_from_spec(spec)
                                sys.modules[mod_name] = m
                                spec.loader.exec_module(m)  # type: ignore
                                module_spec = importlib.util.find_spec(mod_name)
                                break
                if module_spec is None:
                    continue
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                try:
                    model = cls.from_pretrained(str(local_dir))
                except Exception:
                    model = cls()
            except Exception as e:
                last_err = e
                continue
            if model is not None:
                break
        if model is None:
            raise RuntimeError(f"No suitable PARSeq entry found in repo. Last error: {last_err}")
        self.model = model

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image.astype(np.uint8)).convert('RGB')

    @self_property
    def name(self) -> str:  # type: ignore
        return f"{self.provider_label}:{self.model_id}"

    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        # Local fast path (expects BGR)
        if self._recognizer_local is not None:
            try:
                if image.ndim == 2:
                    img_bgr = np.stack([image]*3, axis=-1)
                else:
                    img_bgr = image[..., ::-1].copy()
                outputs, _vis = self._recognizer_local(img_bgr, points=None, vis=None)
                contents = getattr(outputs, 'contents', []) or []
                scores = getattr(outputs, 'scores', []) or []
                if contents:
                    text = contents[0]
                    conf = float(scores[0]) if scores else (1.0 if text else 0.0)
                    return text, conf
            except Exception as e:
                logger.warning(f"Local bullseye recognizer failed: {e}; trying HF path")

        pil = self._to_pil(image)

        try:
            for mname in ('predict', 'recognize', 'infer'):
                if hasattr(self.model, mname):
                    fn = getattr(self.model, mname)
                    out = fn(pil)
                    if isinstance(out, tuple) and len(out) >= 1:
                        text = str(out[0])
                        conf = float(out[1]) if len(out) > 1 else (1.0 if text else 0.0)
                        return text, conf
                    if isinstance(out, str):
                        return out, (1.0 if out else 0.0)
        except Exception:
            pass

        inputs = None
        try:
            if self.processor is not None:
                inputs = self.processor(images=pil, return_tensors="pt")
            elif self.tokenizer is not None:
                arr = np.array(pil).transpose(2, 0, 1)
                tensor = self.torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                inputs = {"pixel_values": tensor}
            else:
                raise RuntimeError("No processor/tokenizer available to prepare inputs")
        except Exception as e:
            logger.warning(f"HF processor failed to prepare inputs: {e}")
            return "", 0.0

        inputs = {k: (v.to(self.model.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

        try:
            if hasattr(self.model, 'generate'):
                gen_ids = self.model.generate(**inputs)
                if self.processor and hasattr(self.processor, 'batch_decode'):
                    text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                elif self.tokenizer is not None:
                    text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                else:
                    text = ""
                return text, (1.0 if text else 0.0)
            outputs = self.model(**inputs)
            text = ""
            conf = 0.0
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                prob = self.torch.softmax(logits, dim=-1)
                conf = float(prob.max(dim=-1).values.mean().item()) if prob is not None else 0.0
                ids = prob.argmax(dim=-1)
                if self.processor and hasattr(self.processor, 'batch_decode'):
                    text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
                elif self.tokenizer is not None:
                    text = self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            return text, conf
        except Exception as e:
            logger.warning(f"HF model inference failed: {e}")
            return "", 0.0
