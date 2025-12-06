"""
Decoders for different representations: FEN, JSON, Graph, Natural Language.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class FENDecoder(nn.Module):
    """
    Decode grid predictions to FEN string.
    
    Uses grid probabilities to constrain generation, addressing exposure bias.
    """
    def __init__(self, input_dim=512, vocab_size=30, max_length=80, hidden_dim=512):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, vocab_size)
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, input_dim)
        self.position_embeddings = nn.Embedding(max_length, input_dim)
        
        # Grid constraint: Use grid probabilities to bias vocabulary
        self.grid_constraint_proj = nn.Linear(64 * 13, vocab_size)
    
    def forward(self, board_features, grid_probs, target_fen=None):
        """
        Args:
            board_features: [B, 64, input_dim] spatial features
            grid_probs: [B, 64, 13] grid probabilities
            target_fen: [B, L] target FEN tokens (for teacher forcing)
        
        Returns:
            FEN logits and probabilities
        """
        if self.training and target_fen is not None:
            return self.forward_train(board_features, grid_probs, target_fen)
        else:
            return self.forward_generate(board_features, grid_probs)
    
    def forward_train(self, board_features, grid_probs, target_fen):
        """Training with teacher forcing."""
        # Get target embeddings
        target_embeds = self.token_embeddings(target_fen)  # [B, L, D]
        
        # Add position embeddings
        positions = torch.arange(target_fen.size(1), device=target_fen.device)
        target_embeds = target_embeds + self.position_embeddings(positions)
        
        # Decode
        output = self.transformer_decoder(
            tgt=target_embeds,
            memory=board_features
        )
        
        # Project to vocabulary
        logits = self.output_proj(output)  # [B, L, vocab_size]
        
        # Add grid constraint bias
        grid_bias = self.grid_constraint_proj(grid_probs.flatten(1))  # [B, vocab_size]
        logits = logits + grid_bias.unsqueeze(1)  # Broadcast to sequence length
        
        return {
            'logits': logits,
            'token_probs': F.softmax(logits, dim=-1)
        }
    
    def forward_generate(self, board_features, grid_probs, start_token_id=1):
        """Autoregressive generation with grid constraints."""
        batch_size = board_features.size(0)
        device = board_features.device
        
        # Start with [START] token
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        for step in range(self.max_length - 1):
            # Get embeddings
            token_embeds = self.token_embeddings(generated)
            positions = torch.arange(generated.size(1), device=device)
            token_embeds = token_embeds + self.position_embeddings(positions)
            
            # Decode
            output = self.transformer_decoder(
                tgt=token_embeds,
                memory=board_features
            )
            
            # Get logits for next token
            next_token_logits = self.output_proj(output[:, -1, :])  # [B, vocab_size]
            
            # Add grid constraint bias
            grid_bias = self.grid_constraint_proj(grid_probs.flatten(1))  # [B, vocab_size]
            next_token_logits = next_token_logits + grid_bias
            
            # Sample next token (greedy for now)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == 0).all():  # 0 = [END] token
                break
        
        return {
            'generated_tokens': generated,
            'token_probs': None
        }


class JSONDecoder(nn.Module):
    """
    Decode to structured JSON representation.
    
    Outputs pieces, relationships, and metadata.
    """
    def __init__(self, input_dim=512, hidden_dim=512):
        super().__init__()
        
        # Piece extraction (from grid probabilities)
        self.piece_extractor = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Relationship detection
        self.relationship_detector = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # Pair of squares
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 relationship types: attacks, defends, pins, controls
        )
        
        # Metadata predictor
        self.metadata_predictor = nn.Sequential(
            nn.Linear(64 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)  # to_move, castling (4), en_passant, material (2), balance
        )
    
    def forward(self, board_features, grid_probs):
        """
        Args:
            board_features: [B, 64, input_dim]
            grid_probs: [B, 64, 13]
        
        Returns:
            JSON-like structure (simplified for now)
        """
        batch_size = board_features.size(0)
        
        # Extract piece information from grid probabilities
        piece_features = self.piece_extractor(grid_probs)  # [B, 64, hidden_dim]
        
        # Predict relationships (simplified: predict for all pairs)
        # In practice, we'd use a more efficient approach
        relationships = []  # Would contain relationship predictions
        
        # Predict metadata
        board_flat = board_features.flatten(1)  # [B, 64*input_dim]
        metadata_logits = self.metadata_predictor(board_flat)  # [B, 10]
        
        return {
            'pieces': piece_features,
            'piece_probs': grid_probs,
            'relationships': relationships,
            'metadata': metadata_logits,
            'validity_score': self.compute_validity(grid_probs)
        }
    
    def compute_validity(self, grid_probs):
        """Check if predicted state is valid."""
        pred_pieces = grid_probs.argmax(dim=-1)  # [B, 64]
        
        # Count kings (should be exactly 2)
        white_kings = (pred_pieces == 6).sum(dim=1)
        black_kings = (pred_pieces == 12).sum(dim=1)
        king_validity = ((white_kings == 1) & (black_kings == 1)).float()
        
        return king_validity.mean().item()


class NLDecoder(nn.Module):
    """
    Natural Language decoder for generating text descriptions.
    """
    def __init__(self, input_dim=512, vocab_size=5000, max_length=512, hidden_dim=512):
        super().__init__()
        self.max_length = max_length
        
        # Small transformer for text generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # Project board features to decoder dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_length, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, board_features, grid_probs, target_tokens=None):
        """
        Args:
            board_features: [B, 64, input_dim]
            grid_probs: [B, 64, 13]
            target_tokens: [B, L] target tokens (for training)
        
        Returns:
            Text generation logits
        """
        # Project board features
        memory = self.input_proj(board_features)  # [B, 64, hidden_dim]
        
        if self.training and target_tokens is not None:
            # Teacher forcing
            target_embeds = self.token_embeddings(target_tokens)
            positions = torch.arange(target_tokens.size(1), device=target_tokens.device)
            target_embeds = target_embeds + self.position_embeddings(positions)
            
            output = self.decoder(tgt=target_embeds, memory=memory)
            logits = self.output_proj(output)
            
            return {
                'logits': logits,
                'token_probs': F.softmax(logits, dim=-1)
            }
        else:
            # Generation (simplified - would need proper autoregressive loop)
            return {
                'generated_tokens': None,
                'token_probs': None
            }

