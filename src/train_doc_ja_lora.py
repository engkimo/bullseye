"""
LoRA training script for gpt-oss-20B using Unsloth and QLoRA.
Optimized for L4 24GB GPU with automatic OOM recovery.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: Unsloth not available. Training will be slower.")


logger = logging.getLogger(__name__)


class DocJALoRATrainer:
    """LoRA trainer for document-specific Japanese LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # OOM recovery parameters
        self.oom_retries = 0
        self.max_oom_retries = 3
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('output_dir', 'outputs')) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with Unsloth optimization."""
        
        model_name = self.config['model_name']
        
        if UNSLOTH_AVAILABLE:
            logger.info("Loading model with Unsloth optimization")
            
            # Unsloth optimized loading
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=self.config.get('max_seq_length', 8192),
                dtype=torch.float16,
                load_in_4bit=True,  # QLoRA 4-bit
                device_map="auto"
            )
            
            # Add LoRA adapters with Unsloth
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.get('lora_r', 32),
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_alpha=self.config.get('lora_alpha', 32),
                lora_dropout=self.config.get('lora_dropout', 0),
                bias="none",
                use_gradient_checkpointing=True,
                random_state=3407,
                use_rslora=False,
                loftq_config=None
            )
        else:
            # Standard loading (slower)
            logger.info("Loading model without Unsloth optimization")
            
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            # QLoRA config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="right",
                use_fast=True
            )
            
            # Add LoRA
            lora_config = LoraConfig(
                r=self.config.get('lora_r', 32),
                lora_alpha=self.config.get('lora_alpha', 32),
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=self.config.get('lora_dropout', 0),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Trainable parameters: {self.count_parameters()}")
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable: {trainable:,} / Total: {total:,} ({trainable/total*100:.2f}%)")
        return trainable
    
    def load_dataset(self) -> Dataset:
        """Load and prepare dataset."""
        
        data_path = self.config['data_path']
        logger.info(f"Loading dataset from {data_path}")
        
        # Load JSONL data
        dataset = load_dataset('json', data_files=data_path, split='train')
        
        # Format for SFT
        def format_example(example):
            # Harmony format
            text = f"{example['prompt']}\n{example['response']}"
            return {"text": text}
        
        dataset = dataset.map(format_example)
        
        # Split train/validation if needed
        if self.config.get('validation_split', 0.1) > 0:
            split = dataset.train_test_split(
                test_size=self.config['validation_split'],
                seed=42
            )
            train_dataset = split['train']
            val_dataset = split['test']
        else:
            train_dataset = dataset
            val_dataset = None
        
        logger.info(f"Train examples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation examples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def create_training_args(self) -> TrainingArguments:
        """Create training arguments with OOM-safe defaults."""
        
        return TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 1),
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 16),
            gradient_checkpointing=True,
            optim="adamw_8bit",
            learning_rate=self.config.get('learning_rate', 2e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            max_grad_norm=self.config.get('max_grad_norm', 0.3),
            warmup_ratio=self.config.get('warmup_ratio', 0.03),
            group_by_length=True,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 200),
            save_total_limit=3,
            evaluation_strategy="steps" if self.config.get('do_eval', False) else "no",
            eval_steps=self.config.get('eval_steps', 200),
            load_best_model_at_end=self.config.get('do_eval', False),
            report_to=self.config.get('report_to', ['tensorboard']),
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            fp16=not is_bfloat16_supported() if UNSLOTH_AVAILABLE else True,
            bf16=is_bfloat16_supported() if UNSLOTH_AVAILABLE else False,
            push_to_hub=False,
            resume_from_checkpoint=self.config.get('resume_from_checkpoint', True)
        )
    
    def train(self):
        """Run training with OOM recovery."""
        
        try:
            self._train_internal()
        except RuntimeError as e:
            if "out of memory" in str(e) and self.oom_retries < self.max_oom_retries:
                logger.warning(f"OOM detected. Attempting recovery (retry {self.oom_retries + 1}/{self.max_oom_retries})")
                self._handle_oom()
                self.train()  # Retry
            else:
                raise
    
    def _train_internal(self):
        """Internal training function."""
        
        # Load model and data
        if self.model is None:
            self.load_model_and_tokenizer()
        
        train_dataset, val_dataset = self.load_dataset()
        
        # Create trainer
        training_args = self.create_training_args()
        
        if UNSLOTH_AVAILABLE:
            # Use Unsloth trainer
            from unsloth import UnslothTrainer as Trainer
        else:
            from transformers import Trainer
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.get('max_seq_length', 8192),
            packing=False,  # Disable for stability
            formatting_func=None,  # Already formatted
        )
        
        # Start training
        logger.info("Starting training...")
        
        # Check for checkpoint
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint_dir = Path(training_args.output_dir)
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            if checkpoints:
                checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[1])))
                logger.info(f"Resuming from checkpoint: {checkpoint}")
        
        # Train
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        
        # Save training results
        with open(Path(training_args.output_dir) / "train_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info("Training completed!")
    
    def _handle_oom(self):
        """Handle OOM by adjusting parameters."""
        
        self.oom_retries += 1
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Adjust parameters
        adjustments = {
            1: {
                'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 16) * 2,
                'max_seq_length': min(self.config.get('max_seq_length', 8192), 4096)
            },
            2: {
                'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 16) * 4,
                'max_seq_length': min(self.config.get('max_seq_length', 8192), 2048),
                'lora_r': max(self.config.get('lora_r', 32) // 2, 16)
            },
            3: {
                'batch_size': 1,
                'gradient_accumulation_steps': 64,
                'max_seq_length': 1024,
                'lora_r': 16
            }
        }
        
        if self.oom_retries in adjustments:
            self.config.update(adjustments[self.oom_retries])
            logger.info(f"Adjusted parameters: {adjustments[self.oom_retries]}")
        
        # Reset model to free memory
        del self.model
        del self.tokenizer
        if self.trainer:
            del self.trainer
        
        torch.cuda.empty_cache()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None


def main():
    """Main training script."""
    
    parser = argparse.ArgumentParser(description="Train LoRA adapter for DocJA")
    parser.add_argument("--config", type=str, default="configs/training_lora.yaml")
    parser.add_argument("--data", type=str, help="Override data path")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    if args.config.endswith('.yaml'):
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        with open(args.config) as f:
            config = json.load(f)
    
    # Override with command line args
    if args.data:
        config['data_path'] = args.data
    if args.output:
        config['output_dir'] = args.output
    if args.resume:
        config['resume_from_checkpoint'] = True
    
    # Set environment variables for optimal performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Create trainer and run
    trainer = DocJALoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()