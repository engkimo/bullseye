"""
JSON extraction evaluation script.
"""

import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import defaultdict
import re

from .harmony_formatter import HarmonyFormatter
from .eval_jsquad import JSQuADEvaluator


logger = logging.getLogger(__name__)


class JSONExtractEvaluator(JSQuADEvaluator):
    """Evaluator for JSON extraction tasks."""
    
    def evaluate_json_extraction(self, test_file: str) -> Dict[str, float]:
        """Evaluate JSON extraction performance."""
        
        # Load test data
        logger.info(f"Loading test data from {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        # Evaluation results
        results = {
            'valid_json': 0,
            'invalid_json': 0,
            'exact_match': 0,
            'field_precision': [],
            'field_recall': [],
            'field_f1': [],
            'value_accuracy': [],
            'by_schema_type': defaultdict(lambda: {
                'total': 0,
                'valid': 0,
                'exact_match': 0,
                'field_f1': []
            })
        }
        
        for example in tqdm(test_data, desc="Evaluating JSON extraction"):
            # Get document and schema
            document = example['document']
            schema = example['schema']
            expected = example['expected']
            schema_type = example.get('schema_type', 'general')
            
            # Generate extraction prompt
            prompt = self.formatter.format_json_extraction_prompt(
                document, schema, examples=example.get('examples')
            )
            
            # Generate response
            response = self.generate_answer(prompt, max_length=200)
            
            # Validate and evaluate
            is_valid, extracted, error = self.formatter.validate_json_response(response, schema)
            
            # Update statistics
            results['by_schema_type'][schema_type]['total'] += 1
            
            if is_valid:
                results['valid_json'] += 1
                results['by_schema_type'][schema_type]['valid'] += 1
                
                # Evaluate extraction quality
                metrics = self._evaluate_extraction(extracted, expected, schema)
                
                results['field_precision'].append(metrics['precision'])
                results['field_recall'].append(metrics['recall'])
                results['field_f1'].append(metrics['f1'])
                results['value_accuracy'].append(metrics['value_accuracy'])
                results['by_schema_type'][schema_type]['field_f1'].append(metrics['f1'])
                
                if metrics['exact_match']:
                    results['exact_match'] += 1
                    results['by_schema_type'][schema_type]['exact_match'] += 1
            else:
                results['invalid_json'] += 1
                logger.debug(f"Invalid JSON: {error}")
        
        # Calculate final metrics
        total = results['valid_json'] + results['invalid_json']
        
        final_metrics = {
            'total_examples': total,
            'valid_json_rate': (results['valid_json'] / total * 100) if total > 0 else 0,
            'exact_match_rate': (results['exact_match'] / total * 100) if total > 0 else 0,
            'field_precision': np.mean(results['field_precision']) * 100 if results['field_precision'] else 0,
            'field_recall': np.mean(results['field_recall']) * 100 if results['field_recall'] else 0,
            'field_f1': np.mean(results['field_f1']) * 100 if results['field_f1'] else 0,
            'value_accuracy': np.mean(results['value_accuracy']) * 100 if results['value_accuracy'] else 0,
            'by_schema_type': {}
        }
        
        # Per-schema-type metrics
        for schema_type, type_results in results['by_schema_type'].items():
            if type_results['total'] > 0:
                final_metrics['by_schema_type'][schema_type] = {
                    'total': type_results['total'],
                    'valid_rate': type_results['valid'] / type_results['total'] * 100,
                    'exact_match_rate': type_results['exact_match'] / type_results['total'] * 100,
                    'field_f1': np.mean(type_results['field_f1']) * 100 if type_results['field_f1'] else 0
                }
        
        return final_metrics
    
    def _evaluate_extraction(self, extracted: Dict, expected: Dict, schema: Dict) -> Dict[str, float]:
        """Evaluate extraction quality."""
        
        # Check exact match
        exact_match = extracted == expected
        
        # Field-level evaluation
        expected_fields = set(expected.keys())
        extracted_fields = set(extracted.keys())
        
        correct_fields = expected_fields & extracted_fields
        
        precision = len(correct_fields) / len(extracted_fields) if extracted_fields else 0
        recall = len(correct_fields) / len(expected_fields) if expected_fields else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Value accuracy (for correctly extracted fields)
        correct_values = 0
        for field in correct_fields:
            if self._values_match(extracted[field], expected[field], schema.get(field)):
                correct_values += 1
        
        value_accuracy = correct_values / len(correct_fields) if correct_fields else 0
        
        return {
            'exact_match': exact_match,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'value_accuracy': value_accuracy
        }
    
    def _values_match(self, extracted_value: Any, expected_value: Any, field_type: Optional[str]) -> bool:
        """Check if extracted value matches expected value."""
        
        # Handle None/null
        if extracted_value is None and expected_value is None:
            return True
        
        if extracted_value is None or expected_value is None:
            return False
        
        # Type-specific comparison
        if field_type == "number":
            try:
                # Convert to float for comparison
                extracted_num = float(extracted_value)
                expected_num = float(expected_value)
                # Allow small floating point differences
                return abs(extracted_num - expected_num) < 0.01
            except (ValueError, TypeError):
                return False
        
        elif field_type == "date":
            # Normalize date formats
            extracted_date = self._normalize_date(str(extracted_value))
            expected_date = self._normalize_date(str(expected_value))
            return extracted_date == expected_date
        
        else:
            # String comparison (case-insensitive, normalized)
            return self._normalize_string(str(extracted_value)) == self._normalize_string(str(expected_value))
    
    def _normalize_string(self, text: str) -> str:
        """Normalize string for comparison."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove quotes
        text = text.strip('"\'')
        # Lowercase
        text = text.lower()
        return text
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date format."""
        # Try to extract YYYY-MM-DD pattern
        date_pattern = re.search(r'(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})', date_str)
        if date_pattern:
            year, month, day = date_pattern.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return date_str


def create_json_test_data(output_file: str, n_examples: int = 100):
    """Create test data for JSON extraction evaluation."""
    
    test_examples = []
    
    # Invoice extraction examples
    for i in range(n_examples // 3):
        doc = f"""
請求書
請求書番号: INV-2024-{1000+i}
発行日: 2024年3月{(i%28)+1}日
請求先: 株式会社サンプル{i}
金額: ¥{(i+1)*12345:,}
支払期限: 2024年4月30日
"""
        test_examples.append({
            'schema_type': 'invoice',
            'document': doc,
            'schema': {
                'invoice_number': 'string',
                'issue_date': 'date',
                'amount': 'number',
                'due_date': 'date',
                'customer': 'string'
            },
            'expected': {
                'invoice_number': f'INV-2024-{1000+i}',
                'issue_date': f'2024-03-{(i%28)+1:02d}',
                'amount': (i+1)*12345,
                'due_date': '2024-04-30',
                'customer': f'株式会社サンプル{i}'
            }
        })
    
    # Contract extraction examples
    for i in range(n_examples // 3):
        doc = f"""
契約書
契約番号: CTR-{2000+i}
契約日: 2024年1月{(i%28)+1}日
甲: 株式会社A{i}
乙: 株式会社B{i}
契約期間: 2024年2月1日から2025年1月31日まで
契約金額: {(i+1)*100000}円
"""
        test_examples.append({
            'schema_type': 'contract',
            'document': doc,
            'schema': {
                'contract_id': 'string',
                'contract_date': 'date',
                'party_a': 'string',
                'party_b': 'string',
                'start_date': 'date',
                'end_date': 'date',
                'amount': 'number'
            },
            'expected': {
                'contract_id': f'CTR-{2000+i}',
                'contract_date': f'2024-01-{(i%28)+1:02d}',
                'party_a': f'株式会社A{i}',
                'party_b': f'株式会社B{i}',
                'start_date': '2024-02-01',
                'end_date': '2025-01-31',
                'amount': (i+1)*100000
            }
        })
    
    # Report extraction with null values
    for i in range(n_examples // 3):
        doc = f"""
月次報告書
報告月: 2024年{(i%12)+1}月
売上: {(i+1)*500000}円
前月比: {'+' if i%2 else '-'}{(i%20)+5}%
特記事項: {'なし' if i%3 == 0 else f'イベント{i}実施'}
"""
        test_examples.append({
            'schema_type': 'report',
            'document': doc,
            'schema': {
                'report_month': 'string',
                'revenue': 'number',
                'growth_rate': 'string',
                'notes': 'string|null'
            },
            'expected': {
                'report_month': f'2024年{(i%12)+1}月',
                'revenue': (i+1)*500000,
                'growth_rate': f"{'+' if i%2 else '-'}{(i%20)+5}%",
                'notes': None if i%3 == 0 else f'イベント{i}実施'
            }
        })
    
    # Save test data
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Created {len(test_examples)} JSON extraction test examples in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JSON extraction")
    parser.add_argument("--model", type=str, help="Model path or LoRA adapter path")
    parser.add_argument("--base", type=str, help="Base model (if using LoRA)")
    parser.add_argument("--test-file", type=str, help="Test data file (JSONL)")
    parser.add_argument("--create-test", action="store_true", help="Create test data")
    parser.add_argument("--output", type=str, default="results/json_extract_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.create_test:
        test_file = args.test_file or "data/json_extract_test.jsonl"
        create_json_test_data(test_file)
        if not args.model:
            return
    
    # Create evaluator
    evaluator = JSONExtractEvaluator(args.model, args.base, args.device)
    
    # Evaluate
    test_file = args.test_file or "data/json_extract_test.jsonl"
    logger.info(f"Starting JSON extraction evaluation on {test_file}...")
    metrics = evaluator.evaluate_json_extraction(test_file)
    
    # Print results
    print("\n=== JSON Extraction Evaluation Results ===")
    print(f"\nTotal Examples: {metrics['total_examples']}")
    print(f"Valid JSON Rate: {metrics['valid_json_rate']:.2f}%")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.2f}%")
    print(f"\nField-level Metrics:")
    print(f"  Precision: {metrics['field_precision']:.2f}%")
    print(f"  Recall: {metrics['field_recall']:.2f}%")
    print(f"  F1: {metrics['field_f1']:.2f}%")
    print(f"  Value Accuracy: {metrics['value_accuracy']:.2f}%")
    
    if metrics['by_schema_type']:
        print("\nBy Schema Type:")
        for schema_type, type_metrics in metrics['by_schema_type'].items():
            print(f"\n  {schema_type}:")
            print(f"    Total: {type_metrics['total']}")
            print(f"    Valid Rate: {type_metrics['valid_rate']:.2f}%")
            print(f"    Exact Match: {type_metrics['exact_match_rate']:.2f}%")
            print(f"    Field F1: {type_metrics['field_f1']:.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()