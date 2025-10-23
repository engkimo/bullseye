from __future__ import annotations

import os
from typing import Optional

from .providers.bullseye import BullseyeProvider
from .providers.base import ProviderConfig, BaseProvider
from .engine.pipeline import DocumentEngine


def get_provider(
    name: Optional[str] = None,
    device: str = "cuda",
    visualize: bool = False,
    infer_onnx: bool = False,
) -> BaseProvider:
    env_name = os.getenv("DOCJA_PROVIDER")
    provider_name = (name or env_name).lower()
    cfg = ProviderConfig(name=provider_name, device=device, visualize=visualize, infer_onnx=infer_onnx)

    if provider_name in ("bullseye"):
        return BullseyeProvider(cfg)
    raise ValueError(f"Unknown provider: {provider_name}")


def create_engine(provider: Optional[BaseProvider] = None, **provider_kwargs) -> DocumentEngine:
    prov = provider or get_provider(**provider_kwargs)
    return DocumentEngine(detector=prov.detector, recognizer=prov.recognizer, layout=prov.layout)

