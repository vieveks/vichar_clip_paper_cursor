"""
Dataset loader for Hugging Face chess-puzzles-images-mini dataset.

This loader supports:
- Loading from pre-split train/val/test CSV files
- Using the dataset's native splits (100k/12.5k/12.5k)
- Custom train/val/test splits for experimentation
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import json


class HFChessDataset(Dataset):
    """
    Dataset loader for Hugging Face chess puzzles dataset.
    
    Supports loading from CSV files with image paths and FEN strings.
    Can use either the dataset's native splits or custom splits.
    """
    
    def __init__(self, csv_file: str, transform=None, use_full_fen: bool = False):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file with columns: image_path, fen, and optionally
                     active_color, castling_rights, en_passant_target_square, best_continuation
            transform: Image transformation pipeline
            use_full_fen: If True, construct full FEN from components. If False, use board_state as-is.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_full_fen = use_full_fen
        
        # Validate required columns
        assert 'image_path' in self.df.columns, "CSV must have 'image_path' column"
        assert 'fen' in self.df.columns, "CSV must have 'fen' column"
        
        # Get base directory for relative paths
        csv_path = Path(csv_file)
        self.base_dir = csv_path.parent
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path (handle both absolute and relative paths)
        img_path = self.df.iloc[idx]['image_path']
        if not Path(img_path).is_absolute():
            img_path = self.base_dir / img_path
        else:
            img_path = Path(img_path)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get FEN string
        if self.use_full_fen and all(col in self.df.columns for col in 
                                     ['active_color', 'castling_rights', 'en_passant_target_square']):
            # Construct full FEN from components
            board_state = self.df.iloc[idx]['fen']
            active_color = self.df.iloc[idx]['active_color']
            castling = self.df.iloc[idx]['castling_rights']
            en_passant = self.df.iloc[idx]['en_passant_target_square']
            # Note: Full FEN would need halfmove and fullmove counters, but dataset doesn't provide them
            # For now, we'll use the board_state as-is since it's already a valid FEN representation
            fen = board_state
        else:
            fen = self.df.iloc[idx]['fen']
        
        return image, fen
    
    def get_metadata(self, idx):
        """Get additional metadata for a sample."""
        row = self.df.iloc[idx]
        metadata = {
            'fen': row['fen'],
            'active_color': row.get('active_color', ''),
            'castling_rights': row.get('castling_rights', ''),
            'en_passant_target_square': row.get('en_passant_target_square', ''),
            'best_continuation': row.get('best_continuation', '')
        }
        return metadata


def load_hf_dataset_splits(data_dir: str, transform=None):
    """
    Load the dataset using its native train/validation/test splits.
    
    Args:
        data_dir: Directory containing train.csv, validation.csv, test.csv
        transform: Image transformation pipeline
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir)
    
    train_csv = data_path / "train.csv"
    val_csv = data_path / "validation.csv"
    test_csv = data_path / "test.csv"
    
    assert train_csv.exists(), f"Train CSV not found: {train_csv}"
    assert val_csv.exists(), f"Validation CSV not found: {val_csv}"
    assert test_csv.exists(), f"Test CSV not found: {test_csv}"
    
    train_dataset = HFChessDataset(train_csv, transform=transform)
    val_dataset = HFChessDataset(val_csv, transform=transform)
    test_dataset = HFChessDataset(test_csv, transform=transform)
    
    return train_dataset, val_dataset, test_dataset


def create_custom_splits(data_dir: str, train_ratio: float = 0.8, 
                        val_ratio: float = 0.1, test_ratio: float = 0.1,
                        transform=None, seed: int = 42):
    """
    Create custom train/val/test splits from the combined dataset.
    
    Args:
        data_dir: Directory containing the dataset CSVs
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set (must sum to 1.0)
        transform: Image transformation pipeline
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    data_path = Path(data_dir)
    
    # Load all splits and combine
    all_dataframes = []
    for split in ['train', 'validation', 'test']:
        csv_path = data_path / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dataframes.append(df)
    
    if not all_dataframes:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Shuffle with seed
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split
    total = len(combined_df)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    train_df = combined_df[:train_size]
    val_df = combined_df[train_size:train_size + val_size]
    test_df = combined_df[train_size + val_size:]
    
    # Save temporary CSVs
    temp_dir = data_path / "custom_splits"
    temp_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(temp_dir / "train.csv", index=False)
    val_df.to_csv(temp_dir / "val.csv", index=False)
    test_df.to_csv(temp_dir / "test.csv", index=False)
    
    # Create datasets
    train_dataset = HFChessDataset(temp_dir / "train.csv", transform=transform)
    val_dataset = HFChessDataset(temp_dir / "val.csv", transform=transform)
    test_dataset = HFChessDataset(temp_dir / "test.csv", transform=transform)
    
    return train_dataset, val_dataset, test_dataset

