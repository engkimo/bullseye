"""
Summarization evaluation script using ROUGE metrics.
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
from rouge_score import rouge_scorer
import re

from .harmony_formatter import HarmonyFormatter, ReasoningLevel
from .eval_jsquad import JSQuADEvaluator


logger = logging.getLogger(__name__)


class SummaryEvaluator(JSQuADEvaluator):
    """Evaluator for summarization tasks."""
    
    def __init__(self, model_path: str, base_model: str = None, device: str = "cuda"):
        super().__init__(model_path, base_model, device)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    def evaluate_summarization(self, test_file: str) -> Dict[str, float]:
        """Evaluate summarization performance."""
        
        # Load test data
        logger.info(f"Loading test data from {test_file}")
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        # Collect scores
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        length_compliance = []
        by_type = {}
        
        for example in tqdm(test_data, desc="Evaluating summarization"):
            document = example['document']
            reference_summary = example['summary']
            max_length = example.get('max_length')
            doc_type = example.get('doc_type', 'general')
            
            # Generate summary
            prompt = self.formatter.format_summary_prompt(
                document,
                max_length=max_length,
                focus_points=example.get('focus_points')
            )
            
            generated_summary = self.generate_answer(prompt, max_length=max_length or 200)
            
            # Calculate ROUGE scores
            scores = self.rouge_scorer.score(reference_summary, generated_summary)
            
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                rouge_scores[metric].append(scores[metric].fmeasure)
            
            # Check length compliance
            if max_length:
                actual_length = len(generated_summary)
                compliance = actual_length <= max_length
                length_compliance.append(compliance)
            
            # Store by document type
            if doc_type not in by_type:
                by_type[doc_type] = {
                    'rouge1': [],
                    'rouge2': [],
                    'rougeL': [],
                    'count': 0
                }
            
            by_type[doc_type]['rouge1'].append(scores['rouge1'].fmeasure)
            by_type[doc_type]['rouge2'].append(scores['rouge2'].fmeasure)
            by_type[doc_type]['rougeL'].append(scores['rougeL'].fmeasure)
            by_type[doc_type]['count'] += 1
        
        # Calculate final metrics
        metrics = {
            'total_examples': len(test_data),
            'rouge1': np.mean(rouge_scores['rouge1']) * 100,
            'rouge2': np.mean(rouge_scores['rouge2']) * 100,
            'rougeL': np.mean(rouge_scores['rougeL']) * 100,
            'length_compliance_rate': np.mean(length_compliance) * 100 if length_compliance else None,
            'by_document_type': {}
        }
        
        # Per-type metrics
        for doc_type, type_scores in by_type.items():
            metrics['by_document_type'][doc_type] = {
                'count': type_scores['count'],
                'rouge1': np.mean(type_scores['rouge1']) * 100,
                'rouge2': np.mean(type_scores['rouge2']) * 100,
                'rougeL': np.mean(type_scores['rougeL']) * 100
            }
        
        # Additional analysis
        metrics['summary_statistics'] = self._analyze_summaries(test_data, rouge_scores)
        
        return metrics
    
    def _analyze_summaries(self, test_data: List[Dict], rouge_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze summary characteristics."""
        
        # Find best and worst performing examples
        rougeL_scores = rouge_scores['rougeL']
        best_idx = np.argmax(rougeL_scores)
        worst_idx = np.argmin(rougeL_scores)
        
        analysis = {
            'best_example': {
                'rougeL': rougeL_scores[best_idx] * 100,
                'doc_type': test_data[best_idx].get('doc_type', 'unknown')
            },
            'worst_example': {
                'rougeL': rougeL_scores[worst_idx] * 100,
                'doc_type': test_data[worst_idx].get('doc_type', 'unknown')
            },
            'score_distribution': {
                'min': np.min(rougeL_scores) * 100,
                'max': np.max(rougeL_scores) * 100,
                'std': np.std(rougeL_scores) * 100,
                'percentiles': {
                    '25': np.percentile(rougeL_scores, 25) * 100,
                    '50': np.percentile(rougeL_scores, 50) * 100,
                    '75': np.percentile(rougeL_scores, 75) * 100
                }
            }
        }
        
        return analysis


def create_summary_test_data(output_file: str, n_examples: int = 50):
    """Create test data for summarization evaluation."""
    
    test_examples = []
    
    # Financial report summaries
    for i in range(n_examples // 3):
        doc = f"""
株式会社サンプル{i} 2024年第{(i%4)+1}四半期決算報告

売上高: {(i+1)*1000}百万円（前年同期比{(i%20)+5}%増）
営業利益: {(i+1)*100}百万円（前年同期比{(i%15)+3}%増）
純利益: {(i+1)*50}百万円（前年同期比{(i%10)+2}%増）

主要な要因:
- 新製品の好調な売上により、主力事業が成長
- コスト削減施策により利益率が改善
- 海外展開が順調に進展し、グローバル売上が拡大

今後の見通し:
通期予想を上方修正し、売上高{(i+1)*4500}百万円、営業利益{(i+1)*450}百万円を見込む。
"""
        
        summary = f"株式会社サンプル{i}の第{(i%4)+1}四半期は、売上高{(i+1)*1000}百万円（前年比{(i%20)+5}%増）、営業利益{(i+1)*100}百万円を達成。新製品好調とコスト削減により増収増益。通期予想を上方修正。"
        
        test_examples.append({
            'doc_type': 'financial',
            'document': doc,
            'summary': summary,
            'max_length': 150
        })
    
    # Contract summaries
    for i in range(n_examples // 3):
        doc = f"""
業務委託契約書

契約当事者:
甲: 株式会社クライアント{i}（以下「甲」）
乙: 株式会社サービス{i}（以下「乙」）

契約内容:
1. 業務内容: システム開発および保守運用
2. 契約期間: 2024年4月1日から2025年3月31日まで
3. 契約金額: 月額{(i+1)*100}万円（税別）
4. 支払条件: 月末締め翌月末払い
5. 秘密保持: 双方は業務上知り得た情報を第三者に開示してはならない
6. 解約条件: 3ヶ月前の書面による通知により解約可能

特記事項:
- 成果物の著作権は甲に帰属する
- 乙は類似業務を他社に提供することができる
"""
        
        summary = f"甲（株式会社クライアント{i}）と乙（株式会社サービス{i}）のシステム開発・保守契約。期間は2024年4月から1年間、月額{(i+1)*100}万円。3ヶ月前通知で解約可。著作権は甲に帰属。"
        
        test_examples.append({
            'doc_type': 'contract',
            'document': doc,
            'summary': summary,
            'max_length': 100,
            'focus_points': ['契約期間', '金額', '解約条件']
        })
    
    # Meeting minutes summaries
    for i in range(n_examples // 3 + n_examples % 3):
        doc = f"""
会議議事録

日時: 2024年3月{(i%28)+1}日 14:00-15:30
場所: 会議室A
参加者: 山田部長、田中課長、鈴木主任、佐藤

議題:
1. プロジェクト{i}の進捗確認
   - 現在の進捗率: {60+(i%30)}%
   - 課題: リソース不足により一部タスクに遅延
   - 対策: 外部リソースの活用を検討

2. 予算の見直し
   - 当初予算: {(i+1)*500}万円
   - 現在の消化率: {40+(i%40)}%
   - 追加予算の必要性について議論

3. 次回のマイルストーン
   - 4月15日: プロトタイプ完成
   - 5月1日: テスト開始
   - 5月31日: リリース予定

決定事項:
- 外部リソースを2名追加投入
- 予算を{(i+1)*100}万円追加申請
- 週次での進捗確認会議を実施

次回会議: 2024年4月{(i%28)+1}日 14:00
"""
        
        summary = f"プロジェクト{i}進捗会議。進捗率{60+(i%30)}%だがリソース不足で遅延。外部2名追加と予算{(i+1)*100}万円追加を決定。4月プロトタイプ、5月末リリース予定。"
        
        test_examples.append({
            'doc_type': 'minutes',
            'document': doc,
            'summary': summary
        })
    
    # Save test data
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    logger.info(f"Created {len(test_examples)} summarization test examples in {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization")
    parser.add_argument("--model", type=str, help="Model path or LoRA adapter path")
    parser.add_argument("--base", type=str, help="Base model (if using LoRA)")
    parser.add_argument("--test-file", type=str, help="Test data file (JSONL)")
    parser.add_argument("--create-test", action="store_true", help="Create test data")
    parser.add_argument("--output", type=str, default="results/summary_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create test data if requested
    if args.create_test:
        test_file = args.test_file or "data/summary_test.jsonl"
        create_summary_test_data(test_file)
        if not args.model:
            return
    
    # Create evaluator
    evaluator = SummaryEvaluator(args.model, args.base, args.device)
    
    # Evaluate
    test_file = args.test_file or "data/summary_test.jsonl"
    logger.info(f"Starting summarization evaluation on {test_file}...")
    metrics = evaluator.evaluate_summarization(test_file)
    
    # Print results
    print("\n=== Summarization Evaluation Results ===")
    print(f"\nTotal Examples: {metrics['total_examples']}")
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {metrics['rouge1']:.2f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.2f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.2f}")
    
    if metrics['length_compliance_rate'] is not None:
        print(f"\nLength Compliance Rate: {metrics['length_compliance_rate']:.2f}%")
    
    if metrics['by_document_type']:
        print("\nBy Document Type:")
        for doc_type, type_metrics in metrics['by_document_type'].items():
            print(f"\n  {doc_type} (n={type_metrics['count']}):")
            print(f"    ROUGE-L: {type_metrics['rougeL']:.2f}")
    
    if 'summary_statistics' in metrics:
        stats = metrics['summary_statistics']
        print("\nScore Distribution:")
        print(f"  Min: {stats['score_distribution']['min']:.2f}")
        print(f"  Median: {stats['score_distribution']['percentiles']['50']:.2f}")
        print(f"  Max: {stats['score_distribution']['max']:.2f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()