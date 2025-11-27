"""
Utility to extract FEN representation from chess board images using the finetuned CLIP model.
"""

import torch
import open_clip
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
from typing import List, Optional
import chess


class CLIPFENExtractor:
    """Extracts FEN from chess board images using finetuned CLIP model."""
    
    def __init__(self, checkpoint_path: str, model_name: str = "ViT-B-32", device: str = None):
        """
        Initialize the CLIP FEN extractor.
        
        Args:
            checkpoint_path: Path to the trained CLIP model checkpoint
            model_name: CLIP model architecture name
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model, self.tokenizer = self._load_model()
        self.transform = self._get_image_transform()
        
    def _load_model(self):
        """Load the CLIP model from checkpoint."""
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")
        
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained="laion2B-s34B-b79K",
            device=self.device
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print(f"[OK] CLIP model loaded from {self.checkpoint_path}")
        return model, tokenizer
    
    def _get_image_transform(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
    
    def extract_fen_from_image(self, image_path: str, fen_candidates: Optional[List[str]] = None, top_k: int = 1) -> str:
        """
        Extract FEN from a chess board image.
        
        Args:
            image_path: Path to chess board image
            fen_candidates: Optional list of candidate FENs to choose from.
                          If None, will generate candidates from common positions.
            top_k: Number of top predictions to return
            
        Returns:
            Best matching FEN string
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        # Generate candidates if not provided
        if fen_candidates is None:
            fen_candidates = self._generate_fen_candidates()
        
        if not fen_candidates:
            raise ValueError("No FEN candidates provided")
        
        # Encode image and text
        text_tokens = self.tokenizer(fen_candidates).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get top-k predictions
        k = min(top_k, len(fen_candidates))
        values, indices = torch.topk(similarity.squeeze(0), k=k)
        
        if top_k == 1:
            return fen_candidates[indices[0].item()]
        else:
            return [fen_candidates[indices[i].item()] for i in range(k)]
    
    def _generate_fen_candidates(self) -> List[str]:
        """
        Generate a set of common FEN candidates for matching.
        This is a fallback - ideally candidates should be provided.
        """
        # Common starting positions and variations
        candidates = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",  # e4
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",  # d4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # e4 e5
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",  # e4 c5
        ]
        return candidates
    
    def extract_fen_from_image_with_candidates_file(self, image_path: str, candidates_csv: str, top_k: int = 1) -> str:
        """
        Extract FEN from image using candidates from a CSV file.
        
        Args:
            image_path: Path to chess board image
            candidates_csv: Path to CSV file with 'fen' column
            top_k: Number of top predictions to return
            
        Returns:
            Best matching FEN string
        """
        try:
            df = pd.read_csv(candidates_csv)
            fen_candidates = df['fen'].dropna().tolist()
        except Exception as e:
            raise ValueError(f"Error reading candidates CSV {candidates_csv}: {e}")
        
        return self.extract_fen_from_image(image_path, fen_candidates, top_k)

