
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.hf_chess_dataset_loader import HFChessDataset

def expand_fen(fen):
    """
    Converts standard FEN notation to expanded FEN format.
    Replaces numbers with repeated '1' tokens (representing empty squares).
    
    Example:
        'r3k2r' -> 'r111k11r'
        '8' -> '11111111'
        'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR' 
        -> 'rnbqkbnr/pppppppp/11111111/11111111/1111P111/11111111/PPPP1PPP/RNBQKBNR'
    
    Args:
        fen: Standard FEN string (board placement part only)
    
    Returns:
        Expanded FEN string where numbers are replaced with '1' tokens
    """
    # Take only the board placement part if full FEN is provided
    board_fen = fen.split()[0] if ' ' in fen else fen
    
    rows = board_fen.split('/')
    expanded_rows = []
    
    for row in rows:
        expanded_row = ""
        for char in row:
            if char.isdigit():
                # Replace digit with that many '1's (empty square tokens)
                expanded_row += '1' * int(char)
            else:
                expanded_row += char
        expanded_rows.append(expanded_row)
    
    return "/".join(expanded_rows)

def collapse_fen(expanded_fen):
    """
    Converts expanded FEN back to standard FEN notation.
    Replaces consecutive '1's with their count.
    
    Example:
        'r111k11r' -> 'r3k2r'
        '11111111' -> '8'
    
    Args:
        expanded_fen: Expanded FEN string with '1' tokens
    
    Returns:
        Standard FEN string with numbers
    """
    rows = expanded_fen.split('/')
    collapsed_rows = []
    
    for row in rows:
        collapsed_row = ""
        count = 0
        for char in row:
            if char == '1':
                count += 1
            else:
                if count > 0:
                    collapsed_row += str(count)
                    count = 0
                collapsed_row += char
        # Handle trailing empty squares
        if count > 0:
            collapsed_row += str(count)
        collapsed_rows.append(collapsed_row)
    
    return "/".join(collapsed_rows)

class FENGenerationDataset(HFChessDataset):
    """
    Dataset for FEN generation task.
    Inherits from HFChessDataset but returns tokenized FENs.
    """
    def __init__(self, csv_file, tokenizer, transform=None, max_len=80):
        super().__init__(csv_file, transform=transform, use_full_fen=False)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):
        image, fen = super().__getitem__(idx)
        
        # Expand FEN: convert numbers to repeated '1' tokens
        # This makes every row exactly 8 tokens (64 squares + 7 slashes = 71 tokens total)
        expanded_fen = expand_fen(fen)
        
        # Tokenize expanded FEN
        token_ids = self.tokenizer.encode(expanded_fen)
        
        # Pad or truncate
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
            token_ids[-1] = self.tokenizer.eos_token_id
        else:
            token_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(token_ids))
            
        return image, torch.tensor(token_ids, dtype=torch.long)

def create_fen_dataloaders(data_dir, tokenizer, transform, batch_size=32, num_workers=4):
    """
    Create dataloaders for FEN generation.
    Reuses the split logic from hf_chess_dataset_loader but wraps with FENGenerationDataset.
    """
    from utils.hf_chess_dataset_loader import load_hf_dataset_splits
    
    # We need to manually create the datasets because load_hf_dataset_splits returns HFChessDataset
    # So we will replicate the logic but use FENGenerationDataset
    
    data_path = Path(data_dir)
    train_csv = data_path / "train.csv"
    val_csv = data_path / "validation.csv"
    test_csv = data_path / "test.csv"
    
    train_dataset = FENGenerationDataset(train_csv, tokenizer, transform=transform)
    val_dataset = FENGenerationDataset(val_csv, tokenizer, transform=transform)
    test_dataset = FENGenerationDataset(test_csv, tokenizer, transform=transform)
    
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
