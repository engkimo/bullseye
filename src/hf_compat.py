"""
HuggingFace compatibility layer for bullseye-style API.
Provides a familiar interface while using our custom models internally.
"""

import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import logging

from .pipeline import DocumentProcessor, DocumentResult


logger = logging.getLogger(__name__)


class DocJaProcessor:
    """bullseye-compatible processor interface."""
    
    def __init__(self, processor: DocumentProcessor):
        self.processor = processor
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load processor in HuggingFace style."""
        model_path = Path(model_name_or_path)
        
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
        
        # Override with kwargs
        config.update(kwargs)
        
        # Initialize processor
        processor = DocumentProcessor(
            det_model=config.get('det_model', 'dbnet'),
            rec_model=config.get('rec_model', 'abinet'),
            layout_model=config.get('layout_model', 'yolo'),
            enable_table=config.get('enable_table', True),
            enable_reading_order=config.get('enable_reading_order', True),
            device=config.get('device', 'cuda'),
            weights_dir=str(model_path),
            lite_mode=config.get('lite_mode', False)
        )
        
        return cls(processor)
    
    def __call__(self, 
                 images: Union[str, List[str]], 
                 detect_layout: bool = True,
                 detect_tables: bool = True,
                 extract_reading_order: bool = True,
                 return_tensors: Optional[str] = None,
                 **kwargs) -> Union[DocumentResult, Dict[str, Any]]:
        """Process images with bullseye-style interface."""
        
        # Handle single image
        if isinstance(images, str):
            images = [images]
        
        results = []
        for image_path in images:
            result = self.processor.process(
                image_path,
                extract_figures=kwargs.get('extract_figures', False)
            )
            results.append(result)
        
        # Return single result if single input
        if len(results) == 1:
            return self._format_output(results[0], return_tensors)
        
        return [self._format_output(r, return_tensors) for r in results]
    
    def _format_output(self, result: DocumentResult, 
                      return_tensors: Optional[str]) -> Union[DocumentResult, Dict[str, Any]]:
        """Format output to match expected interface."""
        
        if return_tensors == "pt":
            # Return PyTorch-style output
            return self._to_pytorch_format(result)
        elif return_tensors == "np":
            # Return NumPy-style output
            return self._to_numpy_format(result)
        else:
            # Return as-is
            return result
    
    def _to_pytorch_format(self, result: DocumentResult) -> Dict[str, torch.Tensor]:
        """Convert result to PyTorch tensor format."""
        output = {}
        
        # Collect all text blocks
        all_texts = []
        all_bboxes = []
        all_labels = []
        
        for page in result.pages:
            for block in page.text_blocks:
                all_texts.append(block.text)
                all_bboxes.append(block.bbox)
                all_labels.append(block.block_type)
        
        # Convert to tensors
        if all_bboxes:
            output['boxes'] = torch.tensor(all_bboxes, dtype=torch.float32)
        
        output['texts'] = all_texts
        output['labels'] = all_labels
        
        return output
    
    def _to_numpy_format(self, result: DocumentResult) -> Dict[str, Any]:
        """Convert result to NumPy format."""
        import numpy as np
        
        output = {}
        
        # Similar to PyTorch format but with NumPy arrays
        all_bboxes = []
        all_texts = []
        all_labels = []
        
        for page in result.pages:
            for block in page.text_blocks:
                all_texts.append(block.text)
                all_bboxes.append(block.bbox)
                all_labels.append(block.block_type)
        
        if all_bboxes:
            output['boxes'] = np.array(all_bboxes, dtype=np.float32)
        
        output['texts'] = all_texts
        output['labels'] = all_labels
        
        return output


class LayoutLMv3Processor:
    """Compatibility wrapper for LayoutLMv3-style processing."""
    
    def __init__(self, processor: DocJaProcessor):
        self.processor = processor
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load processor in LayoutLMv3 style."""
        processor = DocJaProcessor.from_pretrained(model_name_or_path, **kwargs)
        return cls(processor)
    
    def __call__(self, images, text=None, boxes=None, word_labels=None, 
                 padding=True, truncation=True, return_tensors="pt", **kwargs):
        """Process with LayoutLMv3-style interface."""
        
        # If text/boxes not provided, extract from image
        if text is None or boxes is None:
            result = self.processor(images, return_tensors=None)
            
            # Extract text and boxes
            text = []
            boxes = []
            
            for page in result.pages:
                for block in page.text_blocks:
                    text.append(block.text)
                    boxes.append(block.bbox)
        
        # Format output
        encoding = {
            'input_ids': self._tokenize(text, padding, truncation),
            'bbox': torch.tensor(boxes, dtype=torch.float32) if return_tensors == "pt" else boxes,
            'attention_mask': torch.ones(len(text), dtype=torch.long) if return_tensors == "pt" else [1] * len(text)
        }
        
        if word_labels is not None:
            encoding['labels'] = torch.tensor(word_labels, dtype=torch.long) if return_tensors == "pt" else word_labels
        
        return encoding
    
    def _tokenize(self, texts: List[str], padding: bool, truncation: bool) -> torch.Tensor:
        """Simple tokenization (placeholder - would use proper tokenizer)."""
        # This is a placeholder - in real implementation would use proper tokenizer
        # For now, just return dummy token IDs
        token_ids = []
        for text in texts:
            # Simple character-level tokenization
            ids = [ord(c) for c in text[:512]]  # Truncate to 512 chars
            if padding:
                ids = ids + [0] * (512 - len(ids))
            token_ids.append(ids)
        
        return torch.tensor(token_ids, dtype=torch.long)


class DonutProcessor:
    """Compatibility wrapper for Donut-style processing."""
    
    def __init__(self, processor: DocJaProcessor):
        self.processor = processor
        self.image_size = (1280, 960)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load processor in Donut style."""
        processor = DocJaProcessor.from_pretrained(model_name_or_path, **kwargs)
        return cls(processor)
    
    def __call__(self, images, return_tensors="pt", **kwargs):
        """Process with Donut-style interface."""
        
        # Process document
        result = self.processor(images, return_tensors=None)
        
        # Convert to Donut format (simplified)
        # Donut expects pixel values and optional decoder inputs
        from PIL import Image
        import numpy as np
        
        if isinstance(images, str):
            image = Image.open(images).convert('RGB')
        else:
            image = images
        
        # Resize to expected size
        image = image.resize(self.image_size, Image.LANCZOS)
        
        # Convert to tensor
        pixel_values = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension
        
        encoding = {
            'pixel_values': pixel_values
        }
        
        # Add extracted text as decoder input if needed
        if kwargs.get('task', None) == 'ocr':
            all_text = []
            for page in result.pages:
                page_text = ' '.join([block.text for block in page.text_blocks])
                all_text.append(page_text)
            encoding['text'] = ' '.join(all_text)
        
        return encoding


# Model wrappers for completeness
class LayoutLMv3Model:
    """Placeholder for LayoutLMv3 model compatibility."""
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        logger.warning("LayoutLMv3Model is a compatibility wrapper. Using DocJA models internally.")
        return cls()
    
    def forward(self, **kwargs):
        raise NotImplementedError("Use DocJaProcessor for document processing")


class DonutModel:
    """Placeholder for Donut model compatibility."""
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        logger.warning("DonutModel is a compatibility wrapper. Using DocJA models internally.")
        return cls()
    
    def generate(self, **kwargs):
        raise NotImplementedError("Use DocJaProcessor for document processing")