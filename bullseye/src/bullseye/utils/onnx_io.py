from __future__ import annotations

from typing import Optional, Sequence


def load_onnx_session(model_bytes: bytes, device: str = "cpu"):
    """
    Lazily import onnx/onnxruntime and build an InferenceSession from given model bytes.

    - Avoids importing heavy deps at module import time.
    - Selects CUDAExecutionProvider when `device` includes 'cuda' and CUDA is available.
    """
    import onnx  # type: ignore
    import onnxruntime  # type: ignore
    import torch

    model = onnx.load_model_from_string(model_bytes)

    providers: Optional[Sequence[str]] = None
    if "cuda" in device and torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]

    if providers is not None:
        return onnxruntime.InferenceSession(model.SerializeToString(), providers=providers)
    return onnxruntime.InferenceSession(model.SerializeToString())


def load_onnx_session_from_path(path_onnx: str, device: str = "cpu"):
    """Load an ONNX model from file path with provider selection by device."""
    with open(path_onnx, "rb") as f:
        model_bytes = f.read()
    return load_onnx_session(model_bytes, device=device)

