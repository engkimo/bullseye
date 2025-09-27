from __future__ import annotations

from typing import Any, Dict, List, Tuple, TypedDict


class BBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class UDJElt(TypedDict, total=False):
    id: int
    type: str
    bbox: List[float]  # [x1,y1,x2,y2]
    conf: float
    refs: List[int]


class UDJPage(TypedDict, total=False):
    page_num: int
    size: Dict[str, int]
    dpi: int
    text_blocks: List[Dict[str, Any]]
    reading_order: List[int]
    layout_elements: List[UDJElt]
    tables: List[Dict[str, Any]]
    graphs: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]
    warnings: List[str]


class UDJDocument(TypedDict, total=False):
    document_id: str
    version: str
    providers: Dict[str, str]
    pages: List[UDJPage]
    metadata: Dict[str, Any]

