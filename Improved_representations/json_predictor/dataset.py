"""
Dataset loader for JSON predictor training.

Loads JSON dataset and provides:
- Image tensors
- Grid targets (64x13 one-hot)
- Metadata targets (to_move, castling)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


# Piece type to index mapping (0 = empty)
PIECE_TO_IDX = {
    'empty': 0,
    'white_pawn': 1, 'white_knight': 2, 'white_bishop': 3,
    'white_rook': 4, 'white_queen': 5, 'white_king': 6,
    'black_pawn': 7, 'black_knight': 8, 'black_bishop': 9,
    'black_rook': 10, 'black_queen': 11, 'black_king': 12
}

IDX_TO_PIECE = {v: k for k, v in PIECE_TO_IDX.items()}

# Square name to index (a1=0, b1=1, ..., h8=63)
def square_name_to_idx(square_name: str) -> int:
    """Convert square name to index (0-63)."""
    file_idx = ord(square_name[0]) - ord('a')  # 0-7
    rank_idx = int(square_name[1]) - 1  # 0-7
    return rank_idx * 8 + file_idx


def idx_to_square_name(idx: int) -> str:
    """Convert index (0-63) to square name."""
    file_idx = idx % 8
    rank_idx = idx // 8
    return chr(ord('a') + file_idx) + str(rank_idx + 1)


class JSONDataset(Dataset):
    """
    Dataset for JSON predictor training.
    
    Loads images and converts JSON representation to:
    - grid_target: [64] tensor of piece indices (0-12)
    - to_move: 0 (white) or 1 (black)
    - castling: [4] binary tensor [K, Q, k, q]
    """
    
    def __init__(
        self,
        data_path: str,
        image_base_dir: Optional[str] = None,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_path: Path to JSON dataset file
            image_base_dir: Base directory for resolving image paths
            transform: Image transforms (default: standard CLIP preprocessing)
            max_samples: Maximum samples to load
        """
        self.data_path = Path(data_path)
        self.image_base_dir = Path(image_base_dir) if image_base_dir else self.data_path.parent
        self.transform = transform
        
        # Load dataset (support both JSON and JSONL formats)
        data_path_str = str(data_path)
        if data_path_str.endswith('.jsonl'):
            # JSONL format: one JSON object per line
            self.data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        else:
            # Regular JSON format
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Load image
        image_path = sample['image_path']
        if not Path(image_path).is_absolute():
            image_path = self.image_base_dir / image_path
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize to 224x224 and normalize
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            image = default_transform(image)
        
        # Parse JSON representation to grid target
        json_repr = sample['json_repr']
        grid_target = self._json_to_grid_target(json_repr)
        
        # Parse metadata
        metadata = json_repr.get('metadata', {})
        to_move = 0 if metadata.get('to_move', 'white') == 'white' else 1
        
        castling_rights = metadata.get('castling_rights', {'white': [], 'black': []})
        castling = torch.tensor([
            1 if 'K' in castling_rights.get('white', []) else 0,
            1 if 'Q' in castling_rights.get('white', []) else 0,
            1 if 'k' in castling_rights.get('black', []) else 0,
            1 if 'q' in castling_rights.get('black', []) else 0
        ], dtype=torch.float32)
        
        return {
            'image': image,
            'grid_target': grid_target,
            'to_move': torch.tensor(to_move, dtype=torch.long),
            'castling': castling,
            'fen': sample['fen'],
            'image_path': sample['image_path']  # Include for prediction export
        }
    
    def _json_to_grid_target(self, json_repr: Dict) -> torch.Tensor:
        """Convert JSON representation to grid target tensor."""
        # Initialize empty grid
        grid = torch.zeros(64, dtype=torch.long)  # All squares empty (0)
        
        # Place pieces
        for piece_info in json_repr['pieces']:
            square = piece_info['square']
            piece_name = piece_info['piece']
            
            square_idx = square_name_to_idx(square)
            piece_idx = PIECE_TO_IDX.get(piece_name, 0)
            
            grid[square_idx] = piece_idx
        
        return grid


def create_json_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing train.json, val.json, test.json
        batch_size: Batch size
        num_workers: Number of data loading workers
        transform: Image transforms
        max_samples: Maximum samples per split
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    # Get parent directory for image resolution
    image_base_dir = data_dir.parent.parent.parent  # Goes up to project root
    
    loaders = []
    for split in ['train.json', 'val.json', 'test.json']:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"Warning: {split_path} not found, using test_sample.json")
            split_path = data_dir / 'test_sample.json'
        
        dataset = JSONDataset(
            str(split_path),
            image_base_dir=str(image_base_dir),
            transform=transform,
            max_samples=max_samples
        )
        
        shuffle = 'train' in str(split)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders.append(loader)
    
    return tuple(loaders)


# Utility function to convert grid prediction back to JSON
def grid_to_json(
    grid_preds: torch.Tensor,
    to_move: int = 0,
    castling: Optional[torch.Tensor] = None
) -> Dict:
    """
    Convert grid predictions to JSON representation.
    
    Args:
        grid_preds: [64] tensor of piece indices
        to_move: 0 (white) or 1 (black)
        castling: [4] tensor of castling rights
    
    Returns:
        JSON representation dict
    """
    pieces = []
    white_material = 0
    black_material = 0
    
    piece_values = {'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 5, 'queen': 9, 'king': 0}
    
    for idx in range(64):
        piece_idx = grid_preds[idx].item() if isinstance(grid_preds[idx], torch.Tensor) else grid_preds[idx]
        
        if piece_idx > 0:  # Not empty
            piece_name = IDX_TO_PIECE[piece_idx]
            square_name = idx_to_square_name(idx)
            
            # Parse color and type
            parts = piece_name.split('_')
            color = parts[0]
            piece_type = parts[1]
            value = piece_values.get(piece_type, 0)
            
            pieces.append({
                'piece': piece_name,
                'square': square_name,
                'color': color,
                'type': piece_type,
                'value': value
            })
            
            # Track material
            if color == 'white':
                white_material += value
            else:
                black_material += value
    
    # Build metadata
    metadata = {
        'to_move': 'white' if to_move == 0 else 'black',
        'material': {'white': white_material, 'black': black_material},
        'material_balance': white_material - black_material
    }
    
    # Add castling if provided
    if castling is not None:
        white_castling = []
        black_castling = []
        if castling[0] > 0.5:
            white_castling.append('K')
        if castling[1] > 0.5:
            white_castling.append('Q')
        if castling[2] > 0.5:
            black_castling.append('k')
        if castling[3] > 0.5:
            black_castling.append('q')
        metadata['castling_rights'] = {'white': white_castling, 'black': black_castling}
    
    return {
        'pieces': pieces,
        'metadata': metadata
    }


# Test function
def test_dataset():
    """Test the dataset loading."""
    import os
    
    test_path = Path(__file__).parent.parent / 'data' / 'json_dataset' / 'test_sample.json'
    
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return
    
    dataset = JSONDataset(str(test_path), max_samples=5)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test first sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Grid target shape: {sample['grid_target'].shape}")
    print(f"To move: {sample['to_move']}")
    print(f"Castling: {sample['castling']}")
    print(f"FEN: {sample['fen']}")
    
    # Show non-empty squares
    non_empty = (sample['grid_target'] > 0).nonzero().squeeze()
    print(f"\nNon-empty squares ({len(non_empty)}):")
    for idx in non_empty[:5]:
        piece_idx = sample['grid_target'][idx].item()
        print(f"  {idx_to_square_name(idx.item())}: {IDX_TO_PIECE[piece_idx]}")


if __name__ == '__main__':
    test_dataset()

