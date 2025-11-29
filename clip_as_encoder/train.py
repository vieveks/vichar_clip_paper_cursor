"""
Training script for ChessLLaVA models.
Fine-tunes projection layer (and optionally language model) on chess QA data.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from pathlib import Path
import logging

from model import ChessLLaVA
from dataset import ChessQADataset, create_qa_dataset_from_benchmark

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        
        # Format prompts for LLaVA
        prompts = []
        for q in questions:
            prompts.append(f"USER: <image>\n{q}\nASSISTANT:")
        
        # Use processor to format inputs
        inputs = model.processor(
            text=prompts,
            images=[images[i] for i in range(images.size(0))],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Create labels (answers)
        answer_texts = [f"{ans}\n" for ans in answers]
        labels = model.processor(
            text=answer_texts,
            return_tensors="pt",
            padding=True
        ).to(device)["input_ids"]
        
        # Forward pass
        outputs = model.language_model(
            **inputs,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            
            # Format prompts
            prompts = []
            for q in questions:
                prompts.append(f"USER: <image>\n{q}\nASSISTANT:")
            
            inputs = model.processor(
                text=prompts,
                images=[images[i] for i in range(images.size(0))],
                return_tensors="pt",
                padding=True
            ).to(device)
            
            answer_texts = [f"{ans}\n" for ans in answers]
            labels = model.processor(
                text=answer_texts,
                return_tensors="pt",
                padding=True
            ).to(device)["input_ids"]
            
            outputs = model.language_model(**inputs, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train ChessLLaVA model")
    
    # Model arguments
    parser.add_argument(
        "--vision_encoder_type",
        type=str,
        choices=["generic", "chess_finetuned"],
        default="generic",
        help="Type of vision encoder to use"
    )
    parser.add_argument(
        "--chess_clip_checkpoint",
        type=str,
        default=None,
        help="Path to chess-finetuned CLIP checkpoint (required if chess_finetuned)"
    )
    parser.add_argument(
        "--language_model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="LLaVA language model name"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Path to dataset CSV with image paths and FENs"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing chess board images"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train/val split ratio"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm"
    )
    parser.add_argument(
        "--train_language_model",
        action="store_true",
        help="Also train language model (not just projection)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/chess_llava",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load questions
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "benchmarking"))
    from questions import get_scoring_questions
    
    questions = get_scoring_questions()
    logger.info(f"Loaded {len(questions)} question types")
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_qa_dataset_from_benchmark(
        dataset_csv=args.dataset_csv,
        images_dir=args.images_dir,
        questions=questions,
        num_samples=args.num_samples
    )
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Load model
    logger.info("Loading model...")
    model = ChessLLaVA(
        vision_encoder_type=args.vision_encoder_type,
        chess_clip_checkpoint=args.chess_clip_checkpoint,
        language_model_name=args.language_model,
        device=device
    )
    
    # Set trainable components
    model.set_trainable_components(
        train_language_model=args.train_language_model,
        train_projection=True
    )
    
    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    logger.info(f"Training {sum(p.numel() for p in trainable_params)} parameters")
    
    # Training loop
    best_val_loss = float("inf")
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "epochs": []
    }
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.max_grad_norm)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save history
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["epochs"].append(epoch + 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "args": vars(args)
            }, checkpoint_path)
            logger.info(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_history": training_history,
            "args": vars(args)
        }, checkpoint_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info("\n[OK] Training completed!")


if __name__ == "__main__":
    main()

