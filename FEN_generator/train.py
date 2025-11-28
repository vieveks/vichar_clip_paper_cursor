
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from FEN_generator.model import ChessFENGenerator
from FEN_generator.tokenizer import FENTokenizer
from FEN_generator.dataset import create_fen_dataloaders
from train_clip_hf_dataset import create_transforms





def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer, scaler, debug=False):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for i, (images, tgt_tokens) in enumerate(pbar):
        if debug and i >= 10:
            break
            
        images = images.to(device)
        tgt_tokens = tgt_tokens.to(device)
        
        # Input to decoder: <SOS> ... <last_token>
        # Target: ... <last_token> <EOS>
        decoder_input = tgt_tokens[:, :-1]
        target = tgt_tokens[:, 1:]
        
        # Create masks
        tgt_seq_len = decoder_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_padding_mask = (decoder_input == tokenizer.pad_token_id).to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=True):
            logits = model(images, decoder_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
            
            # Standard cross-entropy loss
            ce_loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            # Length penalty: penalize if predicted EOS is too early
            # Find actual target lengths (excluding padding)
            target_lens = (tgt_tokens != tokenizer.pad_token_id).sum(dim=1).float()  # includes SOS
            
            # Find predicted EOS positions
            # Get argmax predictions for each position
            preds = logits.argmax(dim=-1)  # [batch_size, seq_len]
            
            # Find first EOS position (or seq_len if no EOS)
            eos_mask = (preds == tokenizer.eos_token_id)
            eos_positions = torch.where(eos_mask.any(dim=1), 
                                       eos_mask.float().argmax(dim=1),
                                       torch.full((eos_mask.size(0),), eos_mask.size(1), device=device))
            pred_lens = eos_positions.float() + 1  # +1 for SOS token
            
            # Penalty for sequences shorter than target
            length_penalty = torch.relu(target_lens - pred_lens).mean()
            
            # Combined loss
            loss = ce_loss + 0.1 * length_penalty
        
        # Check for NaN
        if torch.isnan(loss):
            logging.warning("Loss is NaN! Skipping step.")
            optimizer.zero_grad()
            continue
            
        scaler.scale(loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    num_batches = 10 if debug else len(dataloader)
    return total_loss / num_batches

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, tokenizer, debug=False):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i, (images, tgt_tokens) in enumerate(tqdm(dataloader, desc="Validating")):
            if debug and i >= 10:
                break
                
            images = images.to(device)
            tgt_tokens = tgt_tokens.to(device)
            
            decoder_input = tgt_tokens[:, :-1]
            target = tgt_tokens[:, 1:]
            
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
            tgt_padding_mask = (decoder_input == tokenizer.pad_token_id).to(device)
            
            with autocast(enabled=True):
                logits = model(images, decoder_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            total_loss += loss.item()
            
    num_batches = 10 if debug else len(dataloader)
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description="Train FEN Generator")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pretrained CLIP checkpoint")
    parser.add_argument("--out_dir", type=str, default="runs/fen_generator", help="Output directory")
    parser.add_argument("--epochs_stage1", type=int, default=5, help="Epochs for Stage 1 (Decoder only)")
    parser.add_argument("--epochs_stage2", type=int, default=10, help="Epochs for Stage 2 (Fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr_encoder", type=float, default=1e-6, help="Learning rate for encoder in Stage 2")
    parser.add_argument("--lr_decoder", type=float, default=3e-4, help="Learning rate for decoder")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer epochs/batches)")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(args.out_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=True # Force reconfiguration
    )
    logging.info(f"Logging to {log_path}")
    
    # Tokenizer
    tokenizer = FENTokenizer()
    
    # Data
    transform = create_transforms()
    train_loader, val_loader, test_loader = create_fen_dataloaders(
        args.data_dir, tokenizer, transform, batch_size=args.batch_size
    )
    
    if args.debug:
        logging.info("Debug mode: reducing dataset size")
        # Just take a few batches
        # (Not easy to slice loader, but we can break loops early)
    
    # Model
    logging.info("Initializing model...")
    model = ChessFENGenerator(vocab_size=len(tokenizer)).to(device)
    
    # Load pretrained CLIP weights
    logging.info(f"Loading pretrained CLIP weights from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle state dict keys
    # If checkpoint saves full model, keys might be 'model_state_dict'
    # And inside, keys might be 'visual.xxx' or just 'xxx' depending on how it was saved
    state_dict = checkpoint['model_state_dict']
    
    # Filter for visual encoder
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('visual.'):
            encoder_state_dict[k.replace('visual.', '')] = v
        elif not k.startswith('text') and not k.startswith('logit_scale'): 
             # If it was saved as just the visual model? Unlikely based on train_clip.py
             # train_clip.py saves 'model_state_dict' of the full CLIP model
             pass
             
    # If we found keys starting with 'visual.', good.
    if encoder_state_dict:
        msg = model.encoder.load_state_dict(encoder_state_dict, strict=False)
        logging.info(f"Loaded encoder weights: {msg}")
    else:
        logging.warning("Could not find 'visual.' keys in checkpoint. Trying to load as-is (risky).")
        # Fallback logic if needed
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # --- STAGE 1: FREEZE ENCODER ---
    logging.info("STAGE 1: Training Decoder Only")
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_decoder
    )
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    epochs_s1 = 1 if args.debug else args.epochs_stage1
    
    for epoch in range(epochs_s1):
        logging.info(f"Stage 1 - Epoch {epoch+1}/{epochs_s1}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, tokenizer, scaler, debug=args.debug)
        val_loss = validate(model, val_loader, criterion, device, tokenizer, debug=args.debug)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.out_dir}/best_model_stage1.pt")
            
    # --- STAGE 2: UNFREEZE ENCODER ---
    logging.info("STAGE 2: Fine-tuning Full Model")
    
    # Unfreeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = True
        
    # Differential learning rates
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
        {'params': model.decoder.parameters(), 'lr': args.lr_decoder},
        {'params': model.fc_out.parameters(), 'lr': args.lr_decoder},
        {'params': model.embedding.parameters(), 'lr': args.lr_decoder},
        {'params': model.encoder_proj.parameters(), 'lr': args.lr_decoder}, # If exists
    ])
    
    # Scheduler for Stage 2
    # Warmup for 1 epoch, then cosine decay
    epochs_s2 = 1 if args.debug else args.epochs_stage2
    num_training_steps = epochs_s2 * len(train_loader)
    num_warmup_steps = len(train_loader) # 1 epoch warmup
    
    # Simple linear warmup + cosine decay implementation
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(epochs_s2):
        logging.info(f"Stage 2 - Epoch {epoch+1}/{epochs_s2}")
        
        # Train loop with scheduler step
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        
        for i, (images, tgt_tokens) in enumerate(pbar):
            if args.debug and i >= 10:
                break
                
            images = images.to(device)
            tgt_tokens = tgt_tokens.to(device)
            
            decoder_input = tgt_tokens[:, :-1]
            target = tgt_tokens[:, 1:]
            
            tgt_seq_len = decoder_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
            tgt_padding_mask = (decoder_input == tokenizer.pad_token_id).to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=True):
                logits = model(images, decoder_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            if torch.isnan(loss):
                logging.warning("Loss is NaN! Skipping step.")
                optimizer.zero_grad()
                continue
                
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
            
        train_loss = total_loss / (10 if args.debug else len(train_loader))
        
        val_loss = validate(model, val_loader, criterion, device, tokenizer, debug=args.debug)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.out_dir}/best_model_stage2.pt")
            
    logging.info("Training complete!")

if __name__ == "__main__":
    main()
