"""
Training script for grid predictor model.
"""

import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from .model import SpatialGridPredictor
from .dataset import GridPredictionDataset, create_transforms


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, grid_labels, _ in pbar:
        images = images.to(device)
        grid_labels = grid_labels.to(device)  # [B, 64]
        
        # Forward pass
        grid_logits = model(images)  # [B, 64, 13]
        
        # Reshape for loss: [B, 64, 13] -> [B*64, 13], [B, 64] -> [B*64]
        logits_flat = grid_logits.reshape(-1, grid_logits.size(-1))
        labels_flat = grid_labels.reshape(-1)
        
        # Compute loss
        loss = criterion(logits_flat, labels_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_squares = 0
    total_squares = 0
    
    with torch.no_grad():
        for images, grid_labels, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            grid_labels = grid_labels.to(device)
            
            # Forward pass
            grid_logits = model(images)
            
            # Compute loss
            logits_flat = grid_logits.reshape(-1, grid_logits.size(-1))
            labels_flat = grid_labels.reshape(-1)
            loss = criterion(logits_flat, labels_flat)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Compute accuracy
            predictions = grid_logits.argmax(dim=-1)  # [B, 64]
            correct_squares += (predictions == grid_labels).sum().item()
            total_squares += grid_labels.numel()
    
    avg_loss = total_loss / num_batches
    accuracy = correct_squares / total_squares if total_squares > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Grid Predictor")
    parser.add_argument("--train_json", type=str, required=True,
                       help="Path to enriched training JSON file")
    parser.add_argument("--val_json", type=str, required=True,
                       help="Path to enriched validation JSON file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/grid_predictor",
                       help="Directory to save checkpoints")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                       help="Path to fine-tuned CLIP encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--freeze_encoder", action="store_true",
                       help="Freeze CLIP encoder (only train spatial aligner and classifier)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per split (for testing)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = checkpoint_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_path}")
    
    # Create datasets
    transform = create_transforms()
    train_dataset = GridPredictionDataset(args.train_json, transform=transform)
    val_dataset = GridPredictionDataset(args.val_json, transform=transform)
    
    # Limit samples if specified
    if args.max_samples:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(args.max_samples, len(val_dataset))))
        logging.info(f"Limited to {args.max_samples} samples per split")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = SpatialGridPredictor(
        encoder_checkpoint=args.encoder_checkpoint,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        logging.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'train_loss': train_loss
        }
        
        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest.pt')
        
        # Save best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(checkpoint, checkpoint_dir / 'best.pt')
            logging.info(f"Saved best model (accuracy: {val_accuracy:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    logging.info(f"\nTraining complete!")
    logging.info(f"Best Val Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    logging.info(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

