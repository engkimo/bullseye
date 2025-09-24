"""
Data mixing utilities for LLM SFT training.
Mixes general instruction data with document-specific data.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict
from tqdm import tqdm

from .harmony_formatter import HarmonyFormatter, ReasoningLevel


logger = logging.getLogger(__name__)


class DataMixer:
    """Mix general and document-specific instruction data."""
    
    def __init__(self,
                 general_ratio: float = 0.7,
                 doc_ratio: float = 0.3,
                 na_injection_rate: float = 0.25,
                 seed: int = 42):
        
        self.general_ratio = general_ratio
        self.doc_ratio = doc_ratio
        self.na_injection_rate = na_injection_rate
        self.formatter = HarmonyFormatter()
        
        random.seed(seed)
        
        # Task type weights for document data
        self.doc_task_weights = {
            'qa': 0.35,
            'json_extraction': 0.25,
            'summary': 0.20,
            'table_qa': 0.10,
            'layout_understanding': 0.10
        }
    
    def mix_datasets(self,
                    general_data: List[Dict[str, str]],
                    doc_data: List[Dict[str, Any]],
                    output_size: Optional[int] = None) -> List[Dict[str, str]]:
        """Mix general and document-specific datasets."""
        
        if output_size is None:
            output_size = len(general_data) + len(doc_data)
        
        # Calculate split
        n_general = int(output_size * self.general_ratio)
        n_doc = output_size - n_general
        
        logger.info(f"Mixing {n_general} general + {n_doc} document examples")
        
        # Sample from datasets
        mixed_data = []
        
        # Add general instruction data
        if len(general_data) >= n_general:
            general_sample = random.sample(general_data, n_general)
        else:
            # Repeat if not enough
            general_sample = general_data * (n_general // len(general_data) + 1)
            general_sample = general_sample[:n_general]
        
        mixed_data.extend(general_sample)
        
        # Add document-specific data
        doc_sample = self._sample_doc_data(doc_data, n_doc)
        mixed_data.extend(doc_sample)
        
        # Shuffle
        random.shuffle(mixed_data)
        
        return mixed_data
    
    def _sample_doc_data(self, doc_data: List[Dict[str, Any]], n_samples: int) -> List[Dict[str, str]]:
        """Sample and format document-specific data."""
        
        formatted_samples = []
        
        # Group by task type
        by_task = defaultdict(list)
        for item in doc_data:
            task_type = item.get('task_type', 'qa')
            by_task[task_type].append(item)
        
        # Sample according to weights
        for task_type, weight in self.doc_task_weights.items():
            n_task = int(n_samples * weight)
            if task_type not in by_task:
                continue
            
            task_data = by_task[task_type]
            if len(task_data) >= n_task:
                samples = random.sample(task_data, n_task)
            else:
                samples = task_data * (n_task // len(task_data) + 1)
                samples = samples[:n_task]
            
            # Format samples
            for sample in samples:
                formatted = self._format_doc_sample(sample, task_type)
                if formatted:
                    formatted_samples.append(formatted)
        
        # Fill remaining with QA if needed
        while len(formatted_samples) < n_samples and 'qa' in by_task:
            sample = random.choice(by_task['qa'])
            formatted = self._format_doc_sample(sample, 'qa')
            if formatted:
                formatted_samples.append(formatted)
        
        return formatted_samples[:n_samples]
    
    def _format_doc_sample(self, sample: Dict[str, Any], task_type: str) -> Optional[Dict[str, str]]:
        """Format a document sample into instruction format."""
        
        try:
            if task_type == 'qa':
                return self._format_qa_sample(sample)
            elif task_type == 'json_extraction':
                return self._format_json_sample(sample)
            elif task_type == 'summary':
                return self._format_summary_sample(sample)
            elif task_type == 'table_qa':
                return self._format_table_qa_sample(sample)
            elif task_type == 'layout_understanding':
                return self._format_layout_sample(sample)
            else:
                return self._format_qa_sample(sample)
        except Exception as e:
            logger.warning(f"Error formatting sample: {e}")
            return None
    
    def _format_qa_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format QA sample."""
        
        document = sample.get('context', sample.get('document', ''))
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        
        # Determine reasoning level
        if len(answer) > 100:
            reasoning = ReasoningLevel.HIGH
        elif len(answer) > 50:
            reasoning = ReasoningLevel.MEDIUM
        else:
            reasoning = ReasoningLevel.LOW
        
        # Apply N/A injection
        prompt = self.formatter.format_prompt(document, question, reasoning)
        prompt, is_na = self.formatter.inject_na_response(prompt, self.na_injection_rate)
        
        if is_na:
            answer = "N/A"
        
        response = self.formatter.format_response(answer, reasoning)
        
        return {
            'prompt': prompt,
            'response': response
        }
    
    def _format_json_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format JSON extraction sample."""
        
        document = sample.get('document', '')
        schema = sample.get('schema', {})
        extracted = sample.get('extracted', {})
        
        prompt = self.formatter.format_json_extraction_prompt(document, schema)
        response = json.dumps(extracted, ensure_ascii=False)
        
        return {
            'prompt': prompt,
            'response': response
        }
    
    def _format_summary_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format summarization sample."""
        
        document = sample.get('document', '')
        summary = sample.get('summary', '')
        max_length = sample.get('max_length', None)
        
        prompt = self.formatter.format_summary_prompt(document, max_length)
        response = self.formatter.format_response(summary, ReasoningLevel.MEDIUM)
        
        return {
            'prompt': prompt,
            'response': response
        }
    
    def _format_table_qa_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format table QA sample."""
        
        # Convert table to text representation
        table = sample.get('table', [])
        table_text = self._table_to_text(table)
        
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        
        prompt = self.formatter.format_prompt(table_text, question, ReasoningLevel.LOW)
        response = answer
        
        return {
            'prompt': prompt,
            'response': response
        }
    
    def _format_layout_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format layout understanding sample."""
        
        # Create text representation of layout
        layout_desc = self._layout_to_text(sample.get('layout', {}))
        question = sample.get('question', 'このドキュメントのレイアウトを説明してください')
        answer = sample.get('answer', layout_desc)
        
        prompt = self.formatter.format_prompt(layout_desc, question, ReasoningLevel.MEDIUM)
        response = answer
        
        return {
            'prompt': prompt,
            'response': response
        }
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table to text representation."""
        
        if not table:
            return "（表データなし）"
        
        lines = []
        for i, row in enumerate(table):
            if i == 0:
                # Header row
                lines.append("ヘッダー: " + " | ".join(row))
            else:
                lines.append(f"行{i}: " + " | ".join(row))
        
        return "\n".join(lines)
    
    def _layout_to_text(self, layout: Dict[str, Any]) -> str:
        """Convert layout info to text representation."""
        
        elements = []
        
        if 'title' in layout:
            elements.append(f"タイトル: {layout['title']}")
        
        if 'sections' in layout:
            for i, section in enumerate(layout['sections']):
                elements.append(f"セクション{i+1}: {section.get('heading', '無題')}")
        
        if 'tables' in layout:
            elements.append(f"表の数: {layout['tables']}")
        
        if 'figures' in layout:
            elements.append(f"図の数: {layout['figures']}")
        
        return "\n".join(elements) if elements else "（レイアウト情報なし）"
    
    def create_validation_split(self,
                              mixed_data: List[Dict[str, str]],
                              val_ratio: float = 0.1) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Create train/validation split."""
        
        n_val = int(len(mixed_data) * val_ratio)
        
        # Shuffle and split
        indices = list(range(len(mixed_data)))
        random.shuffle(indices)
        
        val_indices = set(indices[:n_val])
        
        train_data = []
        val_data = []
        
        for i, item in enumerate(mixed_data):
            if i in val_indices:
                val_data.append(item)
            else:
                train_data.append(item)
        
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data


def main():
    """CLI for data mixing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mix instruction datasets")
    parser.add_argument("--general", type=str, required=True, help="General instruction data (JSONL)")
    parser.add_argument("--doc", type=str, required=True, help="Document instruction data (JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output file (JSONL)")
    parser.add_argument("--size", type=int, help="Output size")
    parser.add_argument("--general-ratio", type=float, default=0.7)
    parser.add_argument("--na-rate", type=float, default=0.25)
    parser.add_argument("--val-split", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Load data
    general_data = []
    with open(args.general, encoding='utf-8') as f:
        for line in f:
            general_data.append(json.loads(line))
    
    doc_data = []
    with open(args.doc, encoding='utf-8') as f:
        for line in f:
            doc_data.append(json.loads(line))
    
    # Mix
    mixer = DataMixer(
        general_ratio=args.general_ratio,
        doc_ratio=1 - args.general_ratio,
        na_injection_rate=args.na_rate
    )
    
    mixed = mixer.mix_datasets(general_data, doc_data, args.size)
    
    # Split
    train_data, val_data = mixer.create_validation_split(mixed, args.val_split)
    
    # Save
    output_path = Path(args.output)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    val_path = output_path.with_suffix('.val.jsonl')
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(train_data)} training examples to {output_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")


if __name__ == "__main__":
    main()