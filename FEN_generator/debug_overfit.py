
import logging
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from FEN_generator.model import ChessFENGenerator
from FEN_generator.tokenizer import FENTokenizer
from FEN_generator.dataset import FENGenerationDataset
from train_clip_hf_dataset import create_transforms

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_overfit_single_batch():
    logging.info("Testing overfitting on a single batch...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = FENTokenizer()
    transform = create_transforms()
    
    # Load a tiny dataset (just train.csv)
    data_dir = Path("data/hf_chess_puzzles")
    train_csv = data_dir / "train.csv"
    
    dataset = FENGenerationDataset(train_csv, tokenizer, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get one batch
    images, tgt_tokens = next(iter(dataloader))
    images = images.to(device)
    tgt_tokens = tgt_tokens.to(device)
    
    logging.info(f"Batch shapes - Images: {images.shape}, Tokens: {tgt_tokens.shape}")
    logging.info(f"Sample FEN: {tokenizer.decode(tgt_tokens[0].tolist())}")
    
    # Model
    model = ChessFENGenerator(vocab_size=len(tokenizer)).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Train loop
    model.train()
    for i in range(100):
        decoder_input = tgt_tokens[:, :-1]
        target = tgt_tokens[:, 1:]
        
        tgt_seq_len = decoder_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_padding_mask = (decoder_input == tokenizer.pad_token_id).to(device)
        
        optimizer.zero_grad()
        logits = model(images, decoder_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            logging.info(f"Iter {i+1}: Loss = {loss.item():.4f}")
            
    # Check generation
    model.eval()
    with torch.no_grad():
        generated = model.generate(images, tokenizer, device=device)
        logging.info(f"Original: {tokenizer.decode(tgt_tokens[0].tolist())}")
        logging.info(f"Generated: {tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    test_overfit_single_batch()
