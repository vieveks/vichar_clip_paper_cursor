"""
Symbolic Refinement Module (Experiment B)

Applies logical constraints to correct common neural errors in JSON predictions.
Target: Improve exact match from 0.008% to 8.3% (+103x improvement).
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Import shared utilities
import sys
from pathlib import Path
shared_path = Path(__file__).parent.parent / "shared"
sys.path.insert(0, str(shared_path))
from utils import grid_to_json, IDX_TO_PIECE, PIECE_TO_IDX, idx_to_square_name

# Also need square_name_to_idx for confidence calculation
from utils import square_name_to_idx


# Piece priority for keeping (higher = more important)
PIECE_PRIORITY = {
    'empty': 0,
    'white_king': 100, 'black_king': 100,  # Highest priority
    'white_queen': 9, 'black_queen': 9,
    'white_rook': 5, 'black_rook': 5,
    'white_bishop': 3, 'black_bishop': 3,
    'white_knight': 3, 'black_knight': 3,
    'white_pawn': 1, 'black_pawn': 1,
}


def refine_json_prediction(
    json_pred: Dict,
    grid_probs: Optional[torch.Tensor] = None,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Apply symbolic refinement to JSON prediction.
    
    Args:
        json_pred: JSON representation dict with 'pieces' and 'metadata'
        grid_probs: [64, 13] tensor of piece probabilities (optional, for confidence-based refinement)
        confidence_threshold: Minimum confidence for keeping pieces
    
    Returns:
        Refined JSON representation dict
    """
    # Make a deep copy to avoid modifying original
    refined = {
        'pieces': [p.copy() for p in json_pred['pieces']],
        'metadata': json_pred['metadata'].copy() if 'metadata' in json_pred else {}
    }
    
    # Apply constraints iteratively
    refined = _apply_king_uniqueness(refined, grid_probs, confidence_threshold)
    refined = _apply_piece_count_limit(refined, grid_probs, confidence_threshold)
    refined = _apply_pawn_placement_rules(refined, grid_probs, confidence_threshold)
    refined = _apply_castling_consistency(refined)
    
    return refined


def _apply_king_uniqueness(
    json_pred: Dict,
    grid_probs: Optional[torch.Tensor],
    confidence_threshold: float
) -> Dict:
    """Ensure exactly one king per color."""
    pieces = json_pred['pieces']
    
    # Count kings by color
    white_kings = [p for p in pieces if p['piece'] == 'white_king']
    black_kings = [p for p in pieces if p['piece'] == 'black_king']
    
    # Keep highest confidence king if multiple
    if len(white_kings) > 1:
        if grid_probs is not None:
            # Find highest confidence king
            best_king = max(white_kings, key=lambda p: _get_piece_confidence(
                p, grid_probs, 'white_king'
            ))
            pieces = [p for p in pieces if p['piece'] != 'white_king'] + [best_king]
        else:
            # Keep first one if no confidence info
            pieces = [p for p in pieces if p['piece'] != 'white_king'] + [white_kings[0]]
    
    if len(black_kings) > 1:
        if grid_probs is not None:
            best_king = max(black_kings, key=lambda p: _get_piece_confidence(
                p, grid_probs, 'black_king'
            ))
            pieces = [p for p in pieces if p['piece'] != 'black_king'] + [best_king]
        else:
            pieces = [p for p in pieces if p['piece'] != 'black_king'] + [black_kings[0]]
    
    # Ensure at least one king per color (if missing, try to infer from high-confidence pieces)
    if len(white_kings) == 0:
        # Look for high-confidence white pieces that might be king
        white_pieces = [p for p in pieces if p['piece'].startswith('white_')]
        if white_pieces and grid_probs is not None:
            # Find highest confidence white piece
            best_piece = max(white_pieces, key=lambda p: _get_piece_confidence(
                p, grid_probs, p['piece']
            ))
            if _get_piece_confidence(best_piece, grid_probs, best_piece['piece']) > confidence_threshold:
                # Convert to king (heuristic: if very high confidence, might be misclassified)
                best_piece['piece'] = 'white_king'
                best_piece['type'] = 'king'
    
    if len(black_kings) == 0:
        black_pieces = [p for p in pieces if p['piece'].startswith('black_')]
        if black_pieces and grid_probs is not None:
            best_piece = max(black_pieces, key=lambda p: _get_piece_confidence(
                p, grid_probs, p['piece']
            ))
            if _get_piece_confidence(best_piece, grid_probs, best_piece['piece']) > confidence_threshold:
                best_piece['piece'] = 'black_king'
                best_piece['type'] = 'king'
    
    json_pred['pieces'] = pieces
    return json_pred


def _apply_piece_count_limit(
    json_pred: Dict,
    grid_probs: Optional[torch.Tensor],
    confidence_threshold: float
) -> Dict:
    """Ensure total pieces <= 32 (16 per side)."""
    pieces = json_pred['pieces']
    
    # Count pieces by color
    white_pieces = [p for p in pieces if p['piece'].startswith('white_')]
    black_pieces = [p for p in pieces if p['piece'].startswith('black_')]
    
    # Remove low-confidence pieces if over limit
    if len(white_pieces) > 16:
        if grid_probs is not None:
            # Sort by confidence (lowest first) and remove excess
            white_pieces.sort(key=lambda p: _get_piece_confidence(
                p, grid_probs, p['piece']
            ))
            # Keep top 16 (highest confidence)
            white_pieces = white_pieces[-16:]
        else:
            # Remove lowest priority pieces
            white_pieces.sort(key=lambda p: PIECE_PRIORITY.get(p['piece'], 0))
            white_pieces = white_pieces[-16:]
    
    if len(black_pieces) > 16:
        if grid_probs is not None:
            black_pieces.sort(key=lambda p: _get_piece_confidence(
                p, grid_probs, p['piece']
            ))
            black_pieces = black_pieces[-16:]
        else:
            black_pieces.sort(key=lambda p: PIECE_PRIORITY.get(p['piece'], 0))
            black_pieces = black_pieces[-16:]
    
    # Reconstruct pieces list
    json_pred['pieces'] = white_pieces + black_pieces
    return json_pred


def _apply_pawn_placement_rules(
    json_pred: Dict,
    grid_probs: Optional[torch.Tensor],
    confidence_threshold: float
) -> Dict:
    """Remove pawns on rank 1 or 8 (invalid positions)."""
    pieces = json_pred['pieces']
    
    # Filter out pawns on invalid ranks
    valid_pieces = []
    for piece in pieces:
        if piece['type'] == 'pawn':
            rank = piece['square'][1]
            if rank in ['1', '8']:
                # Invalid pawn position - skip it
                continue
        valid_pieces.append(piece)
    
    json_pred['pieces'] = valid_pieces
    return json_pred


def _apply_castling_consistency(json_pred: Dict) -> Dict:
    """Ensure castling rights are consistent with king/rook positions."""
    pieces = json_pred['pieces']
    metadata = json_pred.get('metadata', {})
    castling_rights = metadata.get('castling_rights', {'white': [], 'black': []})
    
    # Check white king position
    white_king = next((p for p in pieces if p['piece'] == 'white_king'), None)
    if white_king:
        if white_king['square'] != 'e1':
            # King has moved - remove all castling rights
            castling_rights['white'] = []
        else:
            # Check rook positions
            white_rooks = [p for p in pieces if p['piece'] == 'white_rook']
            rook_squares = {p['square'] for p in white_rooks}
            
            # Remove kingside castling if rook not on h1
            if 'K' in castling_rights['white'] and 'h1' not in rook_squares:
                castling_rights['white'] = [r for r in castling_rights['white'] if r != 'K']
            
            # Remove queenside castling if rook not on a1
            if 'Q' in castling_rights['white'] and 'a1' not in rook_squares:
                castling_rights['white'] = [r for r in castling_rights['white'] if r != 'Q']
    
    # Check black king position
    black_king = next((p for p in pieces if p['piece'] == 'black_king'), None)
    if black_king:
        if black_king['square'] != 'e8':
            castling_rights['black'] = []
        else:
            black_rooks = [p for p in pieces if p['piece'] == 'black_rook']
            rook_squares = {p['square'] for p in black_rooks}
            
            if 'k' in castling_rights['black'] and 'h8' not in rook_squares:
                castling_rights['black'] = [r for r in castling_rights['black'] if r != 'k']
            
            if 'q' in castling_rights['black'] and 'a8' not in rook_squares:
                castling_rights['black'] = [r for r in castling_rights['black'] if r != 'q']
    
    metadata['castling_rights'] = castling_rights
    json_pred['metadata'] = metadata
    return json_pred


def _get_piece_confidence(
    piece: Dict,
    grid_probs: torch.Tensor,
    piece_name: str
) -> float:
    """Get confidence score for a piece from grid probabilities."""
    square = piece['square']
    square_idx = square_name_to_idx(square)
    piece_idx = PIECE_TO_IDX.get(piece_name, 0)
    
    if grid_probs is not None and square_idx < grid_probs.shape[0]:
        return grid_probs[square_idx, piece_idx].item()
    return 0.5  # Default confidence if not available


def refine_grid_prediction(
    grid_preds: torch.Tensor,
    grid_probs: torch.Tensor,
    to_move: int = 0,
    castling: Optional[torch.Tensor] = None,
    confidence_threshold: float = 0.5
) -> Tuple[torch.Tensor, Dict]:
    """
    Refine grid predictions by converting to JSON, applying constraints, and converting back.
    
    Args:
        grid_preds: [64] tensor of piece indices
        grid_probs: [64, 13] tensor of piece probabilities
        to_move: 0 (white) or 1 (black)
        castling: [4] tensor of castling rights
        confidence_threshold: Minimum confidence for keeping pieces
    
    Returns:
        Tuple of (refined_grid_preds, refined_json)
    """
    # Convert to JSON
    json_pred = grid_to_json(grid_preds, to_move, castling)
    
    # Apply refinement
    refined_json = refine_json_prediction(json_pred, grid_probs, confidence_threshold)
    
    # Convert back to grid (if needed)
    # For now, return JSON - grid conversion can be done separately if needed
    return grid_preds, refined_json  # Return original grid and refined JSON

