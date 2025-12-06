"""
JSON Predictor Model

CLIP-based model that predicts:
1. Grid: 64x13 classification (piece type per square)
2. Metadata: to_move (binary), castling rights (4 binary)

The grid prediction is then deterministically converted to JSON and FEN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    print("Warning: open_clip not available, using torchvision resnet")


class SpatialAligner(nn.Module):
    """
    Aligns CLIP visual features from 7x7 patches to 8x8 chess board.
    
    Uses learnable upsampling to preserve spatial information.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        input_spatial_size: int = 7,
        output_spatial_size: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_spatial_size = input_spatial_size
        self.output_spatial_size = output_spatial_size
        
        # Project to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable upsampling (7x7 -> 8x8)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Position embeddings for 8x8 grid
        self.position_embeddings = nn.Parameter(
            torch.randn(1, output_spatial_size * output_spatial_size, hidden_dim) * 0.02
        )
    
    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: [B, 49, 768] from CLIP visual encoder
        
        Returns:
            aligned_features: [B, 64, hidden_dim] aligned to 8x8 board
        """
        B = spatial_features.shape[0]
        
        # Project
        x = self.input_proj(spatial_features)  # [B, 49, hidden_dim]
        
        # Reshape to 2D
        x = x.transpose(1, 2)  # [B, hidden_dim, 49]
        x = x.view(B, self.hidden_dim, self.input_spatial_size, self.input_spatial_size)  # [B, hidden_dim, 7, 7]
        
        # Upsample to 8x8
        x = self.upsample(x)  # [B, hidden_dim, 8, 8]
        
        # Flatten
        x = x.view(B, self.hidden_dim, -1)  # [B, hidden_dim, 64]
        x = x.transpose(1, 2)  # [B, 64, hidden_dim]
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        return x


class JSONPredictorModel(nn.Module):
    """
    Main model for predicting JSON representation from chess board images.
    
    Architecture:
    1. CLIP ViT-B/32 visual encoder (frozen or fine-tuned)
    2. Spatial aligner (7x7 -> 8x8)
    3. Per-square classifier (64 x 13-way classification)
    4. Metadata predictor (to_move, castling)
    
    Output is converted deterministically to JSON and FEN.
    """
    
    def __init__(
        self,
        encoder_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        encoder_checkpoint: Optional[str] = None,
        hidden_dim: int = 512,
        num_piece_types: int = 13,
        freeze_encoder: bool = True
    ):
        """
        Args:
            encoder_name: CLIP model name
            pretrained: Pretrained weights name
            encoder_checkpoint: Path to fine-tuned CLIP checkpoint
            hidden_dim: Hidden dimension for classifier
            num_piece_types: Number of piece classes (13: empty + 6 white + 6 black)
            freeze_encoder: Whether to freeze the visual encoder
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.hidden_dim = hidden_dim
        self.num_piece_types = num_piece_types
        
        # Load CLIP visual encoder
        if HAS_OPEN_CLIP:
            clip_model, _, _ = open_clip.create_model_and_transforms(
                encoder_name, pretrained=pretrained
            )
            self.visual_encoder = clip_model.visual
            self.visual_dim = 768  # ViT-B/32
        else:
            # Fallback to ResNet
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)
            self.visual_encoder = nn.Sequential(*list(backbone.children())[:-2])
            self.visual_dim = 2048
        
        # Load fine-tuned checkpoint if provided
        if encoder_checkpoint:
            self._load_encoder_weights(encoder_checkpoint)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            print("Visual encoder frozen")
        
        # Spatial aligner
        self.spatial_aligner = SpatialAligner(
            input_dim=self.visual_dim,
            hidden_dim=hidden_dim
        )
        
        # Per-square classifier
        self.square_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_piece_types)
        )
        
        # Metadata predictors
        # Global feature for metadata (use [CLS] token)
        self.global_proj = nn.Linear(self.visual_dim, hidden_dim)
        
        # To-move predictor (binary: white=0, black=1)
        self.to_move_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Castling predictor (4 binary: K, Q, k, q)
        self.castling_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )
    
    def _load_encoder_weights(self, checkpoint_path: str):
        """Load weights from fine-tuned CLIP checkpoint."""
        print(f"Loading encoder weights from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Extract visual encoder weights
        visual_state_dict = {}
        for key, value in state_dict.items():
            if 'visual' in key or key.startswith('image_encoder'):
                new_key = key.replace('visual.', '').replace('image_encoder.', '')
                visual_state_dict[new_key] = value
        
        if visual_state_dict:
            # Try to load
            try:
                self.visual_encoder.load_state_dict(visual_state_dict, strict=False)
                print(f"Loaded {len(visual_state_dict)} visual encoder weights")
            except Exception as e:
                print(f"Warning: Could not load visual encoder weights: {e}")
    
    def _extract_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spatial and global features from images.
        
        Args:
            images: [B, 3, 224, 224] input images
        
        Returns:
            spatial_features: [B, 49, visual_dim] patch features
            global_features: [B, visual_dim] global (CLS) features
        """
        if HAS_OPEN_CLIP:
            # Get transformer outputs
            x = self.visual_encoder.conv1(images)  # [B, width, H, W]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, HW]
            x = x.permute(0, 2, 1)  # [B, HW, width]
            
            # Add CLS token
            cls_token = self.visual_encoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)  # [B, HW+1, width]
            
            # Add position embedding
            x = x + self.visual_encoder.positional_embedding.to(x.dtype)
            
            # Pre-norm
            x = self.visual_encoder.ln_pre(x)
            
            # Transformer
            x = x.permute(1, 0, 2)  # [HW+1, B, width]
            x = self.visual_encoder.transformer(x)
            x = x.permute(1, 0, 2)  # [B, HW+1, width]
            
            # Post-norm
            x = self.visual_encoder.ln_post(x)
            
            # Split CLS and patch tokens
            global_features = x[:, 0]  # [B, width] - keep at 768 for global_proj
            spatial_features = x[:, 1:]  # [B, 49, width]
            
            # Note: We don't apply the CLIP projection here to keep global_features at 768 dims
            # for our global_proj layer which expects 768 input
        else:
            # ResNet fallback
            features = self.visual_encoder(images)  # [B, 2048, 7, 7]
            B, C, H, W = features.shape
            spatial_features = features.view(B, C, -1).permute(0, 2, 1)  # [B, 49, 2048]
            global_features = features.mean(dim=[2, 3])  # [B, 2048]
        
        return spatial_features, global_features
    
    def forward(
        self,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, 224, 224] input images
        
        Returns:
            Dictionary with:
                - grid_logits: [B, 64, 13] per-square piece logits
                - to_move_logits: [B, 2] to-move logits
                - castling_logits: [B, 4] castling logits
        """
        # Extract features
        spatial_features, global_features = self._extract_features(images)
        
        # Align to 8x8 grid
        aligned_features = self.spatial_aligner(spatial_features)  # [B, 64, hidden_dim]
        
        # Per-square classification
        grid_logits = self.square_classifier(aligned_features)  # [B, 64, 13]
        
        # Metadata prediction from global features
        global_proj = self.global_proj(global_features)  # [B, hidden_dim]
        to_move_logits = self.to_move_classifier(global_proj)  # [B, 2]
        castling_logits = self.castling_classifier(global_proj)  # [B, 4]
        
        return {
            'grid_logits': grid_logits,
            'to_move_logits': to_move_logits,
            'castling_logits': castling_logits
        }
    
    def predict(
        self,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with probabilities.
        
        Returns:
            Dictionary with:
                - grid_preds: [B, 64] predicted piece indices
                - grid_probs: [B, 64, 13] piece probabilities
                - to_move: [B] predicted to-move (0=white, 1=black)
                - castling: [B, 4] castling predictions (binary)
        """
        outputs = self.forward(images)
        
        grid_probs = F.softmax(outputs['grid_logits'], dim=-1)
        grid_preds = grid_probs.argmax(dim=-1)
        
        to_move = outputs['to_move_logits'].argmax(dim=-1)
        castling = torch.sigmoid(outputs['castling_logits']) > 0.5
        
        return {
            'grid_preds': grid_preds,
            'grid_probs': grid_probs,
            'to_move': to_move,
            'castling': castling.float()
        }


def test_model():
    """Test model forward pass."""
    print("Testing JSONPredictorModel...")
    
    # Create model
    model = JSONPredictorModel(freeze_encoder=True)
    model.eval()
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        predictions = model.predict(images)
    
    print(f"\nModel outputs:")
    print(f"  grid_logits shape: {outputs['grid_logits'].shape}")
    print(f"  to_move_logits shape: {outputs['to_move_logits'].shape}")
    print(f"  castling_logits shape: {outputs['castling_logits'].shape}")
    
    print(f"\nPredictions:")
    print(f"  grid_preds shape: {predictions['grid_preds'].shape}")
    print(f"  grid_probs shape: {predictions['grid_probs'].shape}")
    print(f"  to_move shape: {predictions['to_move'].shape}")
    print(f"  castling shape: {predictions['castling'].shape}")
    
    # Check shapes
    assert outputs['grid_logits'].shape == (batch_size, 64, 13)
    assert outputs['to_move_logits'].shape == (batch_size, 2)
    assert outputs['castling_logits'].shape == (batch_size, 4)
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    test_model()

