import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import clip

class ChessDataset(Dataset):
    """
    A PyTorch Dataset for loading chess board images and their corresponding FEN notations.
    """
    def __init__(self, data_dir: str, preprocess):
        """
        Args:
            data_dir (str): Path to the dataset directory which contains 'images' and 'texts' subdirectories.
            preprocess: The image preprocessing function from CLIP.
        """
        self.data_dir = Path(data_dir)
        self.image_paths = sorted(list((self.data_dir / "images").glob("*.png")))
        self.text_paths = sorted(list((self.data_dir / "texts").glob("*.txt")))
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)

        # Load text
        with open(self.text_paths[idx], "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Tokenize text
        tokenized_text = clip.tokenize([text])[0]

        return image, tokenized_text
