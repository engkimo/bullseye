from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class TextDetectorIF(ABC):
    @abstractmethod
    def __call__(self, img: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:  # TextDetectorSchema
        ...


class TextRecognizerIF(ABC):
    @abstractmethod
    def __call__(self, img: np.ndarray, polygons=None, vis=None) -> Tuple[Any, Optional[np.ndarray]]:  # TextRecognizerSchema
        ...


class LayoutAnalyzerIF(ABC):
    @abstractmethod
    def __call__(self, img: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:  # Layout schema
        ...


class TableStructureRecognizerIF(ABC):
    @abstractmethod
    def __call__(self, img: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:  # Table schema
        ...


class ReadingOrderIF(ABC):
    @abstractmethod
    def estimate(self, elements: Any, mode: str, image: Optional[np.ndarray] = None) -> None:
        ...


class ExporterIF(ABC):
    @abstractmethod
    def export(self, document: Any, out_path: str, **options: Any) -> str:
        ...


class LLMClientIF(ABC):
    @abstractmethod
    def run(self, task: str, prompt: str, timeout: float = 30.0, strict_json: bool = True) -> Any:
        ...

