"""
Hierarchical Multi-Representation Model.

Combines grid predictor with multiple representation decoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..grid_predictor.model import SpatialGridPredictor
from .decoders import FENDecoder, JSONDecoder, NLDecoder


class HierarchicalMultiRepresentationModel(nn.Module):
    """
    Complete hierarchical model for multi-representation chess position understanding.
    
    Architecture:
    1. Grid Predictor (Level 1)
    2. Multi-Representation Decoders (Level 2)
       - FEN Decoder
       - JSON Decoder
       - Natural Language Decoder
    """
    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        freeze_grid_predictor: bool = False,
        vocab_size_fen: int = 30,
        vocab_size_nl: int = 5000
    ):
        super().__init__()
        
        # Level 1: Grid Predictor
        self.grid_predictor = SpatialGridPredictor(
            encoder_checkpoint=encoder_checkpoint,
            freeze_encoder=False  # Will be controlled separately
        )
        
        if freeze_grid_predictor:
            for param in self.grid_predictor.parameters():
                param.requires_grad = False
        
        # Level 2: Decoders
        hidden_dim = 512
        
        self.fen_decoder = FENDecoder(
            input_dim=hidden_dim,
            vocab_size=vocab_size_fen,
            hidden_dim=hidden_dim
        )
        
        self.json_decoder = JSONDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        self.nl_decoder = NLDecoder(
            input_dim=hidden_dim,
            vocab_size=vocab_size_nl,
            hidden_dim=hidden_dim
        )
        
        # Confidence estimator (simple MLP)
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, images, output_representations='all'):
        """
        Args:
            images: [B, 3, 224, 224] chess board images
            output_representations: 'all' or list of ['grid', 'fen', 'json', 'nl']
        
        Returns:
            Dictionary with requested representations
        """
        # Level 1: Grid prediction
        grid_logits = self.grid_predictor(images)  # [B, 64, 13]
        grid_probs = F.softmax(grid_logits, dim=-1)
        
        # Get board features from grid predictor's spatial aligner
        # We need to extract these - for now, we'll recompute or cache them
        # In practice, we'd modify grid_predictor to return features too
        with torch.no_grad():
            encoder_output = self.grid_predictor.encoder.forward_intermediates(images)
            spatial_features = encoder_output['image_intermediates'][-1]
            from einops import rearrange
            spatial_features_seq = rearrange(spatial_features, 'b c h w -> b (h w) c')
            board_features = self.grid_predictor.spatial_aligner(spatial_features_seq)
        
        outputs = {
            'grid_logits': grid_logits,
            'grid_probs': grid_probs
        }
        
        # Estimate confidence for grid
        grid_confidence = self.confidence_net(board_features.mean(dim=1))  # [B, 1]
        outputs['confidence_grid'] = grid_confidence.squeeze(-1)
        
        # Level 2: Decoders
        if output_representations == 'all' or 'fen' in output_representations:
            fen_output = self.fen_decoder(board_features, grid_probs)
            outputs['fen'] = fen_output
        
        if output_representations == 'all' or 'json' in output_representations:
            json_output = self.json_decoder(board_features, grid_probs)
            outputs['json'] = json_output
            outputs['confidence_json'] = json_output.get('validity_score', 0.0)
        
        if output_representations == 'all' or 'nl' in output_representations:
            nl_output = self.nl_decoder(board_features, grid_probs)
            outputs['nl'] = nl_output
        
        return outputs

