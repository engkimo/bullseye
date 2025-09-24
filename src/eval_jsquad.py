"""
JSQuAD evaluation script for Japanese question answering.
"""

import json
import argparse
from typing import Dict, List, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from collections import Counter
import re

from .harmony_formatter import HarmonyFormatter, ReasoningLevel


logger = logging.getLogger(__name__)


class JSQuADEvaluator:
    """Evaluator for JSQuAD dataset."""
    
    def __init__(self, model_path: str, base_model: str = None, device: str = "cuda"):
        self.device = device
        self.formatter = HarmonyFormatter()
        
        # Load model and tokenizer
        self.load_model(model_path, base_model)
    
    def load_model(self, model_path: str, base_model: str = None):
        """Load model with LoRA adapter if specified."""
        
        if base_model:
            # Load base model with LoRA adapter
            from peft import PeftModel
            
            logger.info(f"Loading base model: {base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            logger.info(f"Loading LoRA adapter: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            # Load full model
            logger.info(f"Loading model: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def evaluate(self, dataset_name: str = "shunk031/JGLUE", subset: str = "JSQuAD") -> Dict[str, float]:
        """Evaluate on JSQuAD dataset."""
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}/{subset}")
        dataset = load_dataset(dataset_name, subset, split="validation")
        
        # Evaluate
        predictions = []
        references = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            # Format prompt
            context = example['context']
            question = example['question']
            prompt = self.formatter.format_prompt(context, question, ReasoningLevel.LOW)
            
            # Generate answer
            pred_answer = self.generate_answer(prompt)
            
            # Get reference answers
            ref_answers = example['answers']['text']
            
            predictions.append(pred_answer)
            references.append(ref_answers)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, references)
        
        return metrics
    
    def generate_answer(self, prompt: str, max_length: int = 100) -> str:
        """Generate answer for given prompt."""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = response[len(prompt):].strip()
        
        # Clean up answer
        answer = self._clean_answer(answer)
        
        return answer
    
    def _clean_answer(self, answer: str) -> str:
        """Clean generated answer."""
        # Remove common prefixes
        prefixes = ['答え:', '回答:', 'A:', 'Answer:']
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Take first sentence if multiple
        if '。' in answer:
            answer = answer.split('。')[0]
        
        return answer.strip()
    
    def calculate_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Calculate EM and F1 scores."""
        
        em_scores = []
        f1_scores = []
        
        for pred, refs in zip(predictions, references):
            # EM score
            em = max([self._exact_match(pred, ref) for ref in refs])
            em_scores.append(em)
            
            # F1 score
            f1 = max([self._f1_score(pred, ref) for ref in refs])
            f1_scores.append(f1)
        
        metrics = {
            'exact_match': np.mean(em_scores) * 100,
            'f1': np.mean(f1_scores) * 100,
            'total_examples': len(predictions)
        }
        
        return metrics
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for evaluation."""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[。、！？\s]', '', text)
        
        # Normalize numbers
        text = re.sub(r'[０-９]', lambda m: chr(ord(m.group(0)) - ord('０') + ord('0')), text)
        
        return text
    
    def _exact_match(self, prediction: str, reference: str) -> float:
        """Calculate exact match score."""
        return float(self._normalize_answer(prediction) == self._normalize_answer(reference))
    
    def _f1_score(self, prediction: str, reference: str) -> float:
        """Calculate F1 score."""
        pred_tokens = self._tokenize_ja(self._normalize_answer(prediction))
        ref_tokens = self._tokenize_ja(self._normalize_answer(reference))
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def _tokenize_ja(self, text: str) -> List[str]:
        """Simple Japanese tokenization."""
        # Character-level tokenization for simplicity
        # In production, use proper morphological analyzer
        return list(text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on JSQuAD")
    parser.add_argument("--model", type=str, required=True, help="Model path or LoRA adapter path")
    parser.add_argument("--base", type=str, help="Base model (if using LoRA)")
    parser.add_argument("--output", type=str, default="results/jsquad_results.json")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = JSQuADEvaluator(args.model, args.base, args.device)
    
    # Evaluate
    logger.info("Starting evaluation on JSQuAD...")
    metrics = evaluator.evaluate()
    
    # Print results
    print("\n=== JSQuAD Evaluation Results ===")
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    print(f"Total Examples: {metrics['total_examples']}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()