"""
Fine-tuning script for Qwen2-VL-2B on JSON prediction task.

Uses Hugging Face transformers and PEFT (LoRA) for efficient fine-tuning.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """Custom callback to print training metrics."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Print loss and learning rate
            if 'loss' in logs:
                print(f"\nStep {state.global_step}: Loss = {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                print(f"  Learning Rate = {logs['learning_rate']:.2e}")
            if 'eval_loss' in logs:
                print(f"  Eval Loss = {logs['eval_loss']:.4f}")

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Improved_representations.vlm_finetuning.dataset import QwenVLDataset


def setup_model_and_processor(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1
):
    """Setup Qwen2-VL model and processor."""
    print(f"Loading model: {model_name}")
    
    # Get HuggingFace token if available
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    # Load processor
    processor = Qwen2VLProcessor.from_pretrained(
        model_name,
        token=hf_token
    )
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=hf_token
    )
    
    # Setup LoRA if requested
    if use_lora:
        print("Setting up LoRA for efficient fine-tuning...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    # TODO: Implement proper metrics (JSON accuracy, per-square accuracy, etc.)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Qwen2-VL-2B on JSON prediction'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='Improved_representations/data/vlm_dataset',
        help='Directory containing train.json and val.json'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2-VL-2B-Instruct',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Improved_representations/checkpoints/qwen2vl_json',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--image_base_dir',
        type=str,
        default='data/hf_chess_puzzles',
        help='Base directory for images'
    )
    parser.add_argument(
        '--use_lora',
        action='store_true',
        default=True,
        help='Use LoRA for efficient fine-tuning'
    )
    parser.add_argument(
        '--lora_r',
        type=int,
        default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=500,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Evaluate every N steps'
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=50,
        help='Log every N steps'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup model and processor
    model, processor = setup_model_and_processor(
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Load datasets
    train_dataset = QwenVLDataset(
        data_path=Path(args.data_dir) / 'train.json',
        image_base_dir=args.image_base_dir,
        processor=processor,
        max_length=args.max_length
    )
    
    val_dataset = QwenVLDataset(
        data_path=Path(args.data_dir) / 'val.json',
        image_base_dir=args.image_base_dir,
        processor=processor,
        max_length=args.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        dataloader_num_workers=0,  # Set to 0 for Windows compatibility
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard
        logging_first_step=True,  # Log the first step
        logging_nan_inf_filter=False,  # Don't filter NaN/Inf
    )
    
    # Data collator - use Seq2Seq collator for generation tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8  # For efficiency
    )
    
    # Trainer with logging callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()],  # Add custom logging callback
    )
    
    # Train
    print("Starting training...")
    print(f"Total training steps: {len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs}")
    print(f"Logging every {args.logging_steps} steps")
    print(f"Evaluating every {args.eval_steps} steps")
    print("-" * 60)
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    print("Training complete!")


if __name__ == '__main__':
    main()

