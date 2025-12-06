"""
Spatial Grid Predictor Model for Level 1: Per-square piece classification.

This model predicts the piece type on each of the 64 chess squares independently,
avoiding the exposure bias problem of sequence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from einops import rearrange


class SpatialAligner(nn.Module):
    """
    Learnable spatial alignment from 7×7 ViT patches to 8×8 chess board squares.
    
    Uses learnable upsampling instead of simple bilinear interpolation.
    """
    def __init__(self, input_dim=768, hidden_dim=512, input_spatial_size=7, output_spatial_size=8):
        super().__init__()
        self.input_spatial_size = input_spatial_size
        self.output_spatial_size = output_spatial_size
        
        # Project to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable upsampling with transposed convolution
        self.upsample = nn.Sequential(
            # Reshape to spatial grid: [B, 49, hidden_dim] -> [B, hidden_dim, 7, 7]
            # Then upsample to 8×8
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(size=(output_spatial_size, output_spatial_size), mode='bilinear', align_corners=False),
            # Refine with convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([hidden_dim, output_spatial_size, output_spatial_size])
        )
        
        # Position embeddings for chess squares
        self.position_embeddings = nn.Parameter(
            torch.randn(1, output_spatial_size * output_spatial_size, hidden_dim) * 0.02
        )
    
    def forward(self, spatial_features):
        """
        Args:
            spatial_features: [B, 49, 768] from ViT (7×7 patches)
        
        Returns:
            board_features: [B, 64, hidden_dim] aligned to chess board
        """
        batch_size = spatial_features.size(0)
        
        # Project
        x = self.input_proj(spatial_features)  # [B, 49, hidden_dim]
        
        # Reshape to spatial: [B, 49, hidden_dim] -> [B, hidden_dim, 7, 7]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_spatial_size, w=self.input_spatial_size)
        
        # Upsample to 8×8
        x = self.upsample(x)  # [B, hidden_dim, 8, 8]
        
        # Back to sequence: [B, hidden_dim, 8, 8] -> [B, 64, hidden_dim]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        return x


class SpatialGridPredictor(nn.Module):
    """
    Grid-based piece classifier for chess positions.
    
    Architecture:
    1. CLIP ViT-B/32 encoder (can load fine-tuned weights)
    2. Spatial aligner (7×7 → 8×8)
    3. Per-square classifier (64 independent 13-way classifiers)
    """
    def __init__(
        self,
        encoder_name="ViT-B-32",
        pretrained="laion2B-s34B-b79K",
        encoder_checkpoint=None,
        hidden_dim=512,
        num_piece_types=13,
        freeze_encoder=False
    ):
        super().__init__()
        
        # 1. Vision Encoder (CLIP ViT-B/32)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=encoder_name,
            pretrained=pretrained
        )
        self.encoder = clip_model.visual
        self.encoder_dim = 768  # ViT-B/32 hidden dimension
        
        # Load fine-tuned weights if provided
        if encoder_checkpoint:
            self._load_encoder_weights(encoder_checkpoint)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 2. Spatial Aligner
        self.spatial_aligner = SpatialAligner(
            input_dim=self.encoder_dim,
            hidden_dim=hidden_dim,
            input_spatial_size=7,  # 7×7 patches from ViT
            output_spatial_size=8  # 8×8 chess board
        )
        
        # 3. Per-square Classifier
        # Each of the 64 squares gets an independent 13-way classifier
        self.grid_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_piece_types)
        )
        
        self.num_piece_types = num_piece_types
    
    def _load_encoder_weights(self, checkpoint_path):
        """Load fine-tuned CLIP encoder weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter for visual encoder keys
            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('visual.'):
                    encoder_state_dict[k.replace('visual.', '')] = v
                elif not k.startswith('text') and not k.startswith('logit_scale'):
                    # Might be direct encoder keys
                    encoder_state_dict[k] = v
            
            if encoder_state_dict:
                missing_keys, unexpected_keys = self.encoder.load_state_dict(
                    encoder_state_dict, strict=False
                )
                print(f"Loaded encoder weights from {checkpoint_path}")
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")
            else:
                print(f"Warning: Could not find encoder weights in {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load encoder weights: {e}")
    
    def forward(self, images):
        """
        Args:
            images: [B, 3, 224, 224] chess board images
        
        Returns:
            grid_logits: [B, 64, 13] logits for each square
        """
        # Extract spatial features from CLIP encoder
        encoder_output = self.encoder.forward_intermediates(images)
        
        # Get spatial features: [B, 768, 7, 7]
        spatial_features = encoder_output['image_intermediates'][-1]
        
        # Reshape to sequence: [B, 768, 7, 7] -> [B, 49, 768]
        batch_size = spatial_features.size(0)
        spatial_features = rearrange(spatial_features, 'b c h w -> b (h w) c')
        
        # Align to 8×8 board
        board_features = self.spatial_aligner(spatial_features)  # [B, 64, hidden_dim]
        
        # Classify each square
        grid_logits = self.grid_classifier(board_features)  # [B, 64, 13]
        
        return grid_logits
    
    def predict(self, images):
        """
        Predict piece types for each square.
        
        Args:
            images: [B, 3, 224, 224] chess board images
        
        Returns:
            grid_predictions: [B, 64] predicted piece indices
            grid_probs: [B, 64, 13] class probabilities
        """
        self.eval()
        with torch.no_grad():
            grid_logits = self.forward(images)
            grid_probs = F.softmax(grid_logits, dim=-1)
            grid_predictions = grid_logits.argmax(dim=-1)
        
        return grid_predictions, grid_probs

