#!/usr/bin/env python3
"""
Unsloth-based LoRA fine-tuning for gpt-oss-20B
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from unsloth import FastLanguageModel
from trl import SFTTrainer
import wandb
from tqdm import tqdm
import numpy as np


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class TrainingConfig:
    model_name: str = "augmxnt/shisa-v2-qwen2.5-72b-instruct-gguf"
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_in_4bit: bool = True
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    output_dir: str = "checkpoints/llm_lora"
    resume_from_checkpoint: Optional[str] = None
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


class DocumentAIDataset:
    """Dataset handler for document AI tasks"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.harmony_template = """<|im_start|>system
You are a helpful AI assistant specialized in Japanese document analysis and information extraction.
<|im_end|>
<|im_start|>user
{instruction}

Document:
{document}
<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
    
    def load_data(self) -> Dataset:
        """Load and prepare dataset"""
        if self.data_path.suffix == '.jsonl':
            data = self._load_jsonl()
        elif self.data_path.suffix == '.json':
            data = self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Convert to Harmony format
        formatted_data = []
        for item in data:
            text = self.harmony_template.format(
                instruction=item.get('instruction', 'Extract information from the document.'),
                document=item['document'],
                response=item['response']
            )
            formatted_data.append({'text': text})
        
        return Dataset.from_list(formatted_data)
    
    def _load_jsonl(self) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _load_json(self) -> List[Dict]:
        """Load JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class UnslothTrainer:
    """Unsloth-based trainer for efficient LoRA fine-tuning"""
    
    def __init__(self, config: TrainingConfig, lora_config: LoRAConfig):
        self.config = config
        self.lora_config = lora_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb if available
        if "WANDB_API_KEY" in os.environ:
            wandb.init(
                project="document-ai-llm",
                name=f"lora_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "training": self.config.__dict__,
                    "lora": self.lora_config.__dict__
                }
            )
    
    def setup_model(self) -> Tuple[FastLanguageModel, AutoTokenizer]:
        """Setup model with Unsloth optimizations"""
        print("Loading model with Unsloth...")
        
        # BitsAndBytes config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=self.config.load_in_4bit,
            quantization_config=bnb_config if self.config.load_in_4bit else None
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_config.r,
            target_modules=self.lora_config.target_modules,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
            random_state=42,
            use_rslora=False,
            loftq_config=None
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Run training"""
        model, tokenizer = self.setup_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            report_to=["wandb"] if "WANDB_API_KEY" in os.environ else [],
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
            resume_from_checkpoint=self.config.resume_from_checkpoint
        )
        
        # Setup trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            args=training_args,
            packing=False  # Disable packing for simplicity
        )
        
        # Memory optimization callback
        def memory_callback(trainer, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        trainer.add_callback(type('MemoryCallback', (), {
            'on_step_end': memory_callback,
            'on_epoch_end': memory_callback
        })())
        
        # Start training
        print(f"Starting training on {len(train_dataset)} samples...")
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        
        # Save final model
        print("Saving final model...")
        model.save_pretrained(f"{self.config.output_dir}/final")
        tokenizer.save_pretrained(f"{self.config.output_dir}/final")
        
        # Save LoRA weights separately
        model.save_pretrained_merged(
            f"{self.config.output_dir}/merged",
            tokenizer,
            save_method="lora"
        )
        
        print("Training completed!")
        return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    eval_dataset: Dataset,
    max_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model on test set"""
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            if i >= max_samples:
                break
            
            inputs = tokenizer(
                sample['text'],
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            
            total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity
    }


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for document AI")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    parser.add_argument("--model-name", type=str, default="augmxnt/shisa-v2-qwen2.5-72b-instruct-gguf")
    parser.add_argument("--output-dir", type=str, default="checkpoints/llm_lora")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-model-id", type=str, help="HuggingFace Hub model ID")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint for evaluation")
    
    args = parser.parse_args()
    
    # Setup configs
    training_config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        resume_from_checkpoint=args.resume,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Create trainer
    trainer = UnslothTrainer(training_config, lora_config)
    
    if args.evaluate_only:
        # Load model for evaluation
        model, tokenizer = FastLanguageModel.from_pretrained(
            args.checkpoint or args.output_dir + "/final",
            max_seq_length=args.max_seq_length,
            dtype=torch.float16,
            load_in_4bit=True
        )
        
        # Load eval data
        dataset_handler = DocumentAIDataset(args.eval_data, tokenizer)
        eval_dataset = dataset_handler.load_data()
        
        # Evaluate
        metrics = evaluate_model(model, tokenizer, eval_dataset)
        print(f"Evaluation metrics: {metrics}")
    else:
        # Load datasets
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_handler = DocumentAIDataset(args.train_data, tokenizer)
        train_dataset = train_handler.load_data()
        
        eval_dataset = None
        if args.eval_data:
            eval_handler = DocumentAIDataset(args.eval_data, tokenizer)
            eval_dataset = eval_handler.load_data()
        
        # Train
        model, tokenizer = trainer.train(train_dataset, eval_dataset)
        
        # Final evaluation
        if eval_dataset:
            metrics = evaluate_model(model, tokenizer, eval_dataset)
            print(f"Final evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()