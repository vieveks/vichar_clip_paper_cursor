"""
Fast training script for CLIP model - optimized for quick iteration.

Uses subset of data, fewer epochs, larger batch size for faster training.
"""

import argparse
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
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
from hf_chess_dataset_loader import load_hf_dataset_splits

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_fast.log'),
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


def train_epoch(model, train_loader, optimizer, scaler, device, fp16, tokenizer, loss_fn, max_grad_norm=1.0):
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
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss detected! Skipping batch.")
            continue
        
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


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


def main():
    parser = argparse.ArgumentParser(
        description="Fast training script for CLIP model on chess dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/hf_chess_puzzles",
        help="Directory containing train.csv, validation.csv, test.csv"
    )
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=10000,
        help="Number of training samples to use (subset for faster training)"
    )
    parser.add_argument(
        "--val_subset_size",
        type=int,
        default=2000,
        help="Number of validation samples to use"
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
        default="runs/clip_hf_chess_fast",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (larger for more VRAM)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5, lower for stability)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (default: 1.0)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
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
    logging.info(f"Using device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create transforms
    transform = create_transforms()
    
    # Load datasets
    logging.info(f"Loading dataset from {args.data_dir}")
    train_dataset, val_dataset, test_dataset = load_hf_dataset_splits(
        args.data_dir, transform=transform
    )
    
    # Create subsets for faster training
    logging.info(f"Creating subsets: Train={args.train_subset_size}, Val={args.val_subset_size}")
    train_subset = Subset(train_dataset, range(min(args.train_subset_size, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(args.val_subset_size, len(val_dataset))))
    
    logging.info(f"Dataset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
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
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset': {
            'name': 'bingbangboom/chess-puzzles-images-mini',
            'train_size': len(train_subset),
            'val_size': len(val_subset),
            'subset': True
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
            'max_grad_norm': args.max_grad_norm
        },
        'hardware': {
            'device': str(device),
            'num_workers': args.num_workers
        }
    }
    
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Training metadata saved to {metadata_path}")
    logging.info(f"\n{'='*60}")
    logging.info("Starting fast training...")
    logging.info(f"{'='*60}")
    
    # Training loop
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, args.fp16, tokenizer, loss_fn, args.max_grad_norm
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
            logging.info(f"[SUCCESS] Saved best model (val_loss: {val_loss:.4f})")
        
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
    
    logging.info(f"\n[SUCCESS] Training complete!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

