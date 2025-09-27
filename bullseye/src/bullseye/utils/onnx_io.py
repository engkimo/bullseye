from __future__ import annotations

from typing import Optional, Sequence
import os

from ..exceptions import OnnxLoadError


def load_onnx_session(model_bytes: bytes, device: str = "cpu"):
    """
    Lazily import onnx/onnxruntime and build an InferenceSession from given model bytes.

    - Avoids importing heavy deps at module import time.
    - Selects CUDAExecutionProvider when `device` includes 'cuda' and CUDA is available.
    """
    import onnx  # type: ignore
    import onnxruntime  # type: ignore
    import torch

    try:
        model = onnx.load_model_from_string(model_bytes)
    except Exception as e:  # pragma: no cover - defensive
        raise OnnxLoadError(f"Failed to load ONNX model bytes: {e}")

    providers: Optional[Sequence[str]] = None
    # Allow forcing CPU to avoid noisy GPU discovery warnings
    force_cpu = os.getenv("DOCJA_ONNX_CPU", "0") == "1"
    if (not force_cpu) and ("cuda" in device) and torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]

    try:
        if providers is not None:
            return onnxruntime.InferenceSession(
                model.SerializeToString(), providers=providers
            )
        return onnxruntime.InferenceSession(model.SerializeToString())
    except Exception as e:  # pragma: no cover - defensive
        raise OnnxLoadError(f"Failed to create ONNX Runtime session: {e}")


def load_onnx_session_from_path(path_onnx: str, device: str = "cpu"):
    """Load an ONNX model from file path with provider selection by device."""
    try:
        with open(path_onnx, "rb") as f:
            model_bytes = f.read()
    except FileNotFoundError as e:  # pragma: no cover
        raise OnnxLoadError(f"ONNX file not found: {path_onnx}") from e
    return load_onnx_session(model_bytes, device=device)
