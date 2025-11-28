
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.hf_chess_dataset_loader import HFChessDataset

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
        
        # Tokenize FEN
        token_ids = self.tokenizer.encode(fen)
        
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
