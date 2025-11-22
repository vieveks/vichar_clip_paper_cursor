"""
Training script for CLIP model using Hugging Face chess-puzzles-images-mini dataset.

This script provides:
- Proper train/validation/test splits (using dataset's native splits or custom splits)
- Comprehensive logging and metrics tracking
- Checkpoint saving with best model selection
- Support for mixed precision training
- Detailed documentation for paper publication
"""

import argparse
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.amp import autocast, GradScaler
from torchvision import transforms
import open_clip
from tqdm import tqdm
import logging
from datetime import datetime

# Import our dataset loader
import sys
sys.path.append(str(Path(__file__).parent / "utils"))
from hf_chess_dataset_loader import load_hf_dataset_splits, create_custom_splits

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)


def create_transforms():
    """Create image transformation pipeline matching CLIP's preprocessing."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


def train_epoch(model, train_loader, optimizer, scaler, device, fp16, tokenizer, loss_fn):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, texts in pbar:
        images = images.to(device)
        texts = tokenizer(texts).to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type="cuda", enabled=fp16 and device.type == "cuda"):
            image_features, text_features, logit_scale = model(images, texts)
            loss = loss_fn(image_features, text_features, logit_scale)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, device, fp16, tokenizer, loss_fn):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, texts in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            texts = tokenizer(texts).to(device)
            
            with autocast(device_type="cuda", enabled=fp16 and device.type == "cuda"):
                image_features, text_features, logit_scale = model(images, texts)
                loss = loss_fn(image_features, text_features, logit_scale)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_training_metadata(args, output_dir, train_size, val_size, test_size):
    """Save training configuration and dataset information for documentation."""
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset': {
            'name': 'bingbangboom/chess-puzzles-images-mini',
            'source': 'https://huggingface.co/datasets/bingbangboom/chess-puzzles-images-mini',
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'total_size': train_size + val_size + test_size,
            'split_method': args.split_method
        },
        'model': {
            'architecture': args.model,
            'pretrained': args.pretrained
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'fp16': args.fp16,
            'optimizer': 'AdamW',
            'loss_function': 'CLIP Loss'
        },
        'hardware': {
            'device': str(args.device),
            'num_workers': args.num_workers
        }
    }
    
    metadata_path = Path(output_dir) / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Training metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train CLIP model on Hugging Face chess puzzles dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing train.csv, validation.csv, test.csv"
    )
    parser.add_argument(
        "--split_method",
        type=str,
        choices=["native", "custom"],
        default="native",
        help="Use dataset's native splits (100k/12.5k/12.5k) or create custom splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (only for custom splits)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (only for custom splits)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (only for custom splits)"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for custom splits"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="CLIP model architecture"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2B-s34B-b79K",
        help="Pretrained weights"
    )
    
    # Training arguments
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/clip_hf_chess",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logging.info(f"Using device: {device}")
    
    # Create transforms
    transform = create_transforms()
    
    # Load datasets
    logging.info(f"Loading dataset from {args.data_dir}")
    if args.split_method == "native":
        train_dataset, val_dataset, test_dataset = load_hf_dataset_splits(
            args.data_dir, transform=transform
        )
        logging.info("Using dataset's native splits")
    else:
        train_dataset, val_dataset, test_dataset = create_custom_splits(
            args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            transform=transform,
            seed=args.split_seed
        )
        logging.info(f"Created custom splits: {args.train_ratio:.1%}/{args.val_ratio:.1%}/{args.test_ratio:.1%}")
    
    logging.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Load model
    logging.info(f"Loading model: {args.model} with pretrained weights: {args.pretrained}")
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.model,
        pretrained=args.pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    from open_clip.loss import ClipLoss
    loss_fn = ClipLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(
        device=device if device.type == "cuda" else "cpu",
        enabled=args.fp16 and device.type == "cuda"
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        logging.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        logging.info(f"Resumed from epoch {start_epoch}")
    
    # Save training metadata
    save_training_metadata(
        args, output_dir,
        len(train_dataset), len(val_dataset), len(test_dataset)
    )
    
    # Training loop
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    logging.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, args.fp16, tokenizer, loss_fn
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, args.fp16, tokenizer, loss_fn)
        
        logging.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['epochs'].append(epoch + 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, best_model_path)
            logging.info(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every epoch
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "training_history": training_history
        }, checkpoint_path)
    
    # Save final training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logging.info(f"\n✅ Training complete!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

