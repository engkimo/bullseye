"""
Document QA evaluation script for document-specific questions.
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

from .harmony_formatter import HarmonyFormatter, ReasoningLevel
from .eval_jsquad import JSQuADEvaluator


logger = logging.getLogger(__name__)


class DocQAEvaluator(JSQuADEvaluator):
    """Evaluator for document-specific QA tasks."""
    
    def __init__(self, model_path: str, base_model: str = None, device: str = "cuda"):
        super().__init__(model_path, base_model, device)
        self.na_threshold = 0.5  # Threshold for N/A detection
    
    def evaluate_docqa(self, test_file: str) -> Dict[str, float]:
        """Evaluate on document QA dataset."""
        
        # Load test data
        logger.info(f"Loading test data from {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        # Group by task type
        by_type = defaultdict(list)
        predictions = []
        references = []
        na_predictions = []
        na_references = []
        
        for example in tqdm(test_data, desc="Evaluating"):
            task_type = example.get('task_type', 'general')
            
            # Format prompt based on task type
            if task_type == 'extraction':
                prompt = self._format_extraction_prompt(example)
            elif task_type == 'table_qa':
                prompt = self._format_table_qa_prompt(example)
            elif task_type == 'layout':
                prompt = self._format_layout_prompt(example)
            else:
                prompt = self._format_general_qa_prompt(example)
            
            # Generate answer
            pred_answer = self.generate_answer(prompt)
            
            # Get reference answer
            ref_answer = example.get('answer', '')
            is_na = ref_answer.lower() in ['n/a', 'na', '該当なし', 'なし']
            
            # Store results
            predictions.append(pred_answer)
            references.append([ref_answer])
            
            # Track N/A performance
            pred_is_na = self._is_na_answer(pred_answer)
            na_predictions.append(pred_is_na)
            na_references.append(is_na)
            
            # Store by type
            by_type[task_type].append({
                'prediction': pred_answer,
                'reference': ref_answer,
                'is_na': is_na,
                'pred_is_na': pred_is_na
            })
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(predictions, references)
        
        # Calculate N/A metrics
        na_metrics = self._calculate_na_metrics(na_predictions, na_references)
        
        # Calculate per-type metrics
        type_metrics = {}
        for task_type, results in by_type.items():
            preds = [r['prediction'] for r in results]
            refs = [[r['reference']] for r in results]
            type_metrics[task_type] = self.calculate_metrics(preds, refs)
        
        # Combine all metrics
        metrics = {
            'overall': overall_metrics,
            'na_detection': na_metrics,
            'by_task_type': type_metrics
        }
        
        return metrics
    
    def _format_extraction_prompt(self, example: Dict) -> str:
        """Format extraction task prompt."""
        document = example['document']
        fields = example.get('fields', {})
        
        # Create extraction instruction
        field_desc = ", ".join([f"{k}: {v}" for k, v in fields.items()])
        question = f"次の情報を抽出してください: {field_desc}"
        
        return self.formatter.format_prompt(document, question, ReasoningLevel.LOW)
    
    def _format_table_qa_prompt(self, example: Dict) -> str:
        """Format table QA prompt."""
        table_text = example.get('table_text', '')
        question = example['question']
        
        return self.formatter.format_prompt(table_text, question, ReasoningLevel.LOW)
    
    def _format_layout_prompt(self, example: Dict) -> str:
        """Format layout understanding prompt."""
        layout_desc = example.get('layout_description', '')
        question = example['question']
        
        return self.formatter.format_prompt(layout_desc, question, ReasoningLevel.MEDIUM)
    
    def _format_general_qa_prompt(self, example: Dict) -> str:
        """Format general QA prompt."""
        document = example.get('context', example.get('document', ''))
        question = example['question']
        
        # Determine reasoning level based on question complexity
        if any(keyword in question for keyword in ['なぜ', '理由', '説明']):
            reasoning = ReasoningLevel.MEDIUM
        else:
            reasoning = ReasoningLevel.LOW
        
        return self.formatter.format_prompt(document, question, reasoning)
    
    def _is_na_answer(self, answer: str) -> bool:
        """Check if answer indicates N/A."""
        na_patterns = [
            'n/a', 'na', '該当なし', 'なし', '不明', '記載なし',
            '情報がありません', '見つかりません', '確認できません'
        ]
        
        answer_lower = answer.lower()
        return any(pattern in answer_lower for pattern in na_patterns)
    
    def _calculate_na_metrics(self, predictions: List[bool], references: List[bool]) -> Dict[str, float]:
        """Calculate N/A detection metrics."""
        
        predictions = np.array(predictions)
        references = np.array(references)
        
        # True positives, false positives, etc.
        tp = np.sum((predictions == True) & (references == True))
        fp = np.sum((predictions == True) & (references == False))
        tn = np.sum((predictions == False) & (references == False))
        fn = np.sum((predictions == False) & (references == True))
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
        
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'accuracy': accuracy * 100,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }


def create_test_data(output_file: str, n_examples: int = 100):
    """Create sample test data for evaluation."""
    
    test_examples = []
    
    # General QA examples
    for i in range(n_examples // 4):
        test_examples.append({
            'task_type': 'general',
            'document': f'株式会社サンプル{i}は2020年に設立された。本社は東京都港区にある。',
            'question': '設立年はいつですか？',
            'answer': '2020年'
        })
    
    # Extraction examples
    for i in range(n_examples // 4):
        test_examples.append({
            'task_type': 'extraction',
            'document': f'請求書番号: INV-{1000+i}\n金額: {(i+1)*10000}円\n支払期限: 2024年3月31日',
            'fields': {'invoice_number': 'string', 'amount': 'number'},
            'question': '請求書番号と金額を抽出してください',
            'answer': f'請求書番号: INV-{1000+i}, 金額: {(i+1)*10000}'
        })
    
    # Table QA examples
    for i in range(n_examples // 4):
        test_examples.append({
            'task_type': 'table_qa',
            'table_text': f'商品名 | 価格 | 在庫\nノートPC | 120000 | {10+i}\nマウス | 3000 | {50+i}',
            'question': 'ノートPCの在庫数は？',
            'answer': str(10+i)
        })
    
    # N/A examples
    for i in range(n_examples // 4):
        test_examples.append({
            'task_type': 'general',
            'document': '売上報告書: 第1四半期の売上は前年比10%増加した。',
            'question': '第2四半期の売上は？',
            'answer': 'N/A'
        })
    
    # Save test data
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Created {len(test_examples)} test examples in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Document QA")
    parser.add_argument("--model", type=str, help="Model path or LoRA adapter path")
    parser.add_argument("--base", type=str, help="Base model (if using LoRA)")
    parser.add_argument("--test-file", type=str, help="Test data file (JSONL)")
    parser.add_argument("--create-test", action="store_true", help="Create test data")
    parser.add_argument("--output", type=str, default="results/docqa_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.create_test:
        test_file = args.test_file or "data/docqa_test.jsonl"
        create_test_data(test_file)
        if not args.model:
            return
    
    # Create evaluator
    evaluator = DocQAEvaluator(args.model, args.base, args.device)
    
    # Evaluate
    test_file = args.test_file or "data/docqa_test.jsonl"
    logger.info(f"Starting evaluation on {test_file}...")
    metrics = evaluator.evaluate_docqa(test_file)
    
    # Print results
    print("\n=== Document QA Evaluation Results ===")
    print("\nOverall Metrics:")
    print(f"  Exact Match: {metrics['overall']['exact_match']:.2f}%")
    print(f"  F1 Score: {metrics['overall']['f1']:.2f}%")
    
    print("\nN/A Detection:")
    print(f"  Precision: {metrics['na_detection']['precision']:.2f}%")
    print(f"  Recall: {metrics['na_detection']['recall']:.2f}%")
    print(f"  F1: {metrics['na_detection']['f1']:.2f}%")
    
    print("\nBy Task Type:")
    for task_type, type_metrics in metrics['by_task_type'].items():
        print(f"\n  {task_type}:")
        print(f"    EM: {type_metrics['exact_match']:.2f}%")
        print(f"    F1: {type_metrics['f1']:.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()