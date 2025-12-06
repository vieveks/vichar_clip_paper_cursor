"""
Training script for JSON Predictor model.

Trains the CLIP-based model to predict:
1. Grid: 64x13 classification (piece type per square)
2. Metadata: to_move (binary), castling rights (4 binary)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from json_predictor.model import JSONPredictorModel
from json_predictor.dataset import JSONDataset


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(grid_logits, grid_targets, to_move_logits, to_move_targets, castling_logits, castling_targets):
    """Compute accuracy metrics."""
    # Grid accuracy (per-square)
    grid_preds = grid_logits.argmax(dim=-1)  # [B, 64]
    grid_correct = (grid_preds == grid_targets).float()
    per_square_acc = grid_correct.mean().item()
    
    # Exact board match
    exact_match = (grid_preds == grid_targets).all(dim=-1).float().mean().item()
    
    # To-move accuracy
    to_move_preds = to_move_logits.argmax(dim=-1)
    to_move_acc = (to_move_preds == to_move_targets).float().mean().item()
    
    # Castling accuracy (per-flag)
    castling_preds = (torch.sigmoid(castling_logits) > 0.5).float()
    castling_acc = (castling_preds == castling_targets).float().mean().item()
    
    return {
        'per_square_acc': per_square_acc,
        'exact_board_match': exact_match,
        'to_move_acc': to_move_acc,
        'castling_acc': castling_acc
    }


def train_epoch(model, dataloader, optimizer, device, epoch, log_interval=50):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    grid_loss_meter = AverageMeter()
    meta_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # Loss functions
    grid_criterion = nn.CrossEntropyLoss()
    to_move_criterion = nn.CrossEntropyLoss()
    castling_criterion = nn.BCEWithLogitsLoss()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        grid_targets = batch['grid_target'].to(device)
        to_move_targets = batch['to_move'].to(device)
        castling_targets = batch['castling'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute losses
        # Grid loss: reshape for cross-entropy
        grid_logits = outputs['grid_logits']  # [B, 64, 13]
        B, S, C = grid_logits.shape
        grid_loss = grid_criterion(
            grid_logits.view(B * S, C),
            grid_targets.view(B * S)
        )
        
        # Metadata losses
        to_move_loss = to_move_criterion(outputs['to_move_logits'], to_move_targets)
        castling_loss = castling_criterion(outputs['castling_logits'], castling_targets)
        
        # Total loss (weighted)
        meta_loss = to_move_loss + castling_loss
        total_loss = grid_loss + 0.5 * meta_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(
                grid_logits, grid_targets,
                outputs['to_move_logits'], to_move_targets,
                outputs['castling_logits'], castling_targets
            )
        
        # Update meters
        loss_meter.update(total_loss.item(), B)
        grid_loss_meter.update(grid_loss.item(), B)
        meta_loss_meter.update(meta_loss.item(), B)
        acc_meter.update(metrics['per_square_acc'], B)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'acc': f"{acc_meter.avg:.4f}",
            'exact': f"{metrics['exact_board_match']:.4f}"
        })
    
    return {
        'loss': loss_meter.avg,
        'grid_loss': grid_loss_meter.avg,
        'meta_loss': meta_loss_meter.avg,
        'per_square_acc': acc_meter.avg
    }


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    exact_match_meter = AverageMeter()
    to_move_meter = AverageMeter()
    castling_meter = AverageMeter()
    
    grid_criterion = nn.CrossEntropyLoss()
    to_move_criterion = nn.CrossEntropyLoss()
    castling_criterion = nn.BCEWithLogitsLoss()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        images = batch['image'].to(device)
        grid_targets = batch['grid_target'].to(device)
        to_move_targets = batch['to_move'].to(device)
        castling_targets = batch['castling'].to(device)
        
        outputs = model(images)
        
        # Compute loss
        grid_logits = outputs['grid_logits']
        B, S, C = grid_logits.shape
        grid_loss = grid_criterion(
            grid_logits.view(B * S, C),
            grid_targets.view(B * S)
        )
        to_move_loss = to_move_criterion(outputs['to_move_logits'], to_move_targets)
        castling_loss = castling_criterion(outputs['castling_logits'], castling_targets)
        total_loss = grid_loss + 0.5 * (to_move_loss + castling_loss)
        
        # Compute metrics
        metrics = compute_metrics(
            grid_logits, grid_targets,
            outputs['to_move_logits'], to_move_targets,
            outputs['castling_logits'], castling_targets
        )
        
        # Update meters
        loss_meter.update(total_loss.item(), B)
        acc_meter.update(metrics['per_square_acc'], B)
        exact_match_meter.update(metrics['exact_board_match'], B)
        to_move_meter.update(metrics['to_move_acc'], B)
        castling_meter.update(metrics['castling_acc'], B)
        
        pbar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'acc': f"{acc_meter.avg:.4f}",
            'exact': f"{exact_match_meter.avg:.4f}"
        })
    
    return {
        'loss': loss_meter.avg,
        'per_square_acc': acc_meter.avg,
        'exact_board_match': exact_match_meter.avg,
        'to_move_acc': to_move_meter.avg,
        'castling_acc': castling_meter.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Train JSON Predictor')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train.json, val.json, test.json')
    parser.add_argument('--image_base_dir', type=str, default=None,
                       help='Base directory for image paths')
    
    # Model arguments
    parser.add_argument('--encoder_checkpoint', type=str, default=None,
                       help='Path to fine-tuned CLIP checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze the visual encoder')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/json_predictor',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Determine image base directory
    if args.image_base_dir is None:
        args.image_base_dir = str(Path(args.data_dir).parent.parent.parent)
    
    # Create datasets (support both .json and .jsonl formats)
    print("Loading datasets...")
    data_dir = Path(args.data_dir)
    
    # Find train file
    train_path = data_dir / 'train.jsonl'
    if not train_path.exists():
        train_path = data_dir / 'train.json'
    
    # Find val file  
    val_path = data_dir / 'val.jsonl'
    if not val_path.exists():
        val_path = data_dir / 'val.json'
    
    train_dataset = JSONDataset(
        str(train_path),
        image_base_dir=args.image_base_dir
    )
    val_dataset = JSONDataset(
        str(val_path),
        image_base_dir=args.image_base_dir
    )
    
    # Create dataloaders
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("Creating model...")
    model = JSONPredictorModel(
        encoder_checkpoint=args.encoder_checkpoint,
        hidden_dim=args.hidden_dim,
        freeze_encoder=args.freeze_encoder
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train': [], 'val': []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['per_square_acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['per_square_acc']:.4f}, "
              f"Exact: {val_metrics['exact_board_match']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_metrics['per_square_acc'] > best_val_acc:
            best_val_acc = val_metrics['per_square_acc']
            checkpoint_path = Path(args.checkpoint_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Saved best model (acc: {best_val_acc:.4f})")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / 'final_model.pt'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'args': vars(args)
    }, final_path)
    
    # Save history
    history_path = Path(args.log_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"History saved to: {history_path}")


if __name__ == '__main__':
    main()

