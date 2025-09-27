from importlib.metadata import version

from .document_analyzer import DocumentAnalyzer
from .engine.pipeline import DocumentEngine
from .factory import get_provider, create_engine
from .layout_analyzer import LayoutAnalyzer
from .layout_parser import LayoutParser
from .ocr import OCR
from .table_structure_recognizer import TableStructureRecognizer
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer

__all__ = [
    "OCR",
    "LayoutParser",
    "TableStructureRecognizer",
    "TextDetector",
    "TextRecognizer",
    "LayoutAnalyzer",
    "DocumentAnalyzer",
    "DocumentEngine",
    "get_provider",
    "create_engine",
]
__version__ = version(__package__)
