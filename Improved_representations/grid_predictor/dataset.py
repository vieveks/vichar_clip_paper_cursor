"""
Dataset for grid prediction task.

Loads enriched dataset JSON files and returns images with grid labels.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms


def create_transforms():
    """Create image transforms for training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class GridPredictionDataset(Dataset):
    """
    Dataset for grid prediction task.
    
    Loads from enriched JSON files that contain:
    - image_path: Path to chess board image
    - grid: 8×8 list of integers (0-12 for piece types)
    """
    def __init__(self, json_path, transform=None, image_base_dir=None):
        """
        Args:
            json_path: Path to enriched dataset JSON file
            transform: Image transformation pipeline
            image_base_dir: Base directory for image paths (if relative)
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.transform = transform if transform is not None else create_transforms()
        
        # Determine image base directory
        if image_base_dir is None:
            json_path_obj = Path(json_path)
            # Try to find images relative to JSON file
            self.image_base_dir = json_path_obj.parent.parent.parent / 'data' / 'hf_chess_puzzles'
        else:
            self.image_base_dir = Path(image_base_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image_path = sample['image_path']
        if not Path(image_path).is_absolute():
            image_path = self.image_base_dir / image_path
        else:
            image_path = Path(image_path)
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get grid label
        grid = sample['grid']  # 8×8 list of lists
        # Convert to tensor: [8, 8] -> [64]
        grid_flat = [item for row in grid for item in row]
        grid_tensor = torch.tensor(grid_flat, dtype=torch.long)
        
        return image, grid_tensor, sample['fen']  # Also return FEN for evaluation

