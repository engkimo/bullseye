from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    name: str = "bullseye"
    device: str = "cuda"
    visualize: bool = False
    infer_onnx: bool = False


class BaseProvider:
    """Base class for provider adapters.

    Concrete providers must expose attributes:
      - detector, recognizer, layout, table, reading_order
    aligned with core.interfaces ABCs.
    """

    def __init__(self, cfg: Optional[ProviderConfig] = None) -> None:
        self.cfg = cfg or ProviderConfig()

