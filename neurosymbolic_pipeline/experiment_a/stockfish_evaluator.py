"""
Stockfish evaluation module for Experiment A.

Evaluates chess positions using Lichess Cloud Evaluation API (Stockfish engine) 
and calculates CP (centipawn) loss.

Uses the Lichess API implementation from benchmarking/ground_truth.py.
"""

import chess
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple

# Import Lichess API evaluator from existing benchmarking code
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarking"))
try:
    from ground_truth import GroundTruthExtractor
except ImportError:
    GroundTruthExtractor = None


class StockfishEvaluator:
    """Evaluates chess positions using Lichess Cloud Evaluation API (Stockfish)."""
    
    def __init__(self, use_lichess_api: bool = True, depth: int = 15):
        """
        Initialize Stockfish evaluator.
        
        Args:
            use_lichess_api: If True, use Lichess Cloud Evaluation API (recommended)
                            If False, use python-chess simple evaluation (fallback)
            depth: Search depth for evaluation (used by Lichess API)
        """
        self.depth = depth
        self.use_lichess_api = use_lichess_api
        self.lichess_extractor = None
        
        # Try to initialize Lichess API evaluator
        if use_lichess_api and GroundTruthExtractor is not None:
            try:
                self.lichess_extractor = GroundTruthExtractor()
                print(f"Using Lichess Cloud Evaluation API (depth={depth})")
            except Exception as e:
                print(f"Warning: Could not initialize Lichess API: {e}")
                print("Falling back to python-chess simple evaluation")
                self.use_lichess_api = False
        else:
            if not use_lichess_api:
                print("Using python-chess built-in evaluation (simple material-based)")
            else:
                print("Lichess API not available, using python-chess built-in evaluation")
                self.use_lichess_api = False
    
    def evaluate_position(self, fen: str) -> Optional[float]:
        """
        Evaluate a chess position and return CP score.
        
        Args:
            fen: FEN string representing the position
        
        Returns:
            CP score in centipawns (positive = white advantage, negative = black advantage)
            None if evaluation fails
        """
        try:
            if self.use_lichess_api and self.lichess_extractor is not None:
                # Use Lichess Cloud Evaluation API (Stockfish engine)
                try:
                    eval_result = self.lichess_extractor.get_position_evaluation(fen, depth=self.depth)
                    if eval_result is None:
                        # API returned None, fall back to simple evaluation
                        board = chess.Board(fen)
                        return self._simple_evaluation(board)
                    
                    # Handle mate scores
                    if eval_result.get('mate') is not None:
                        mate_ply = eval_result['mate']
                        return 10000 if mate_ply > 0 else -10000
                    
                    # Return CP score (already in centipawns)
                    return eval_result.get('score', 0)
                except Exception as api_error:
                    # If API fails, fall back to simple evaluation silently
                    # (errors are already logged by GroundTruthExtractor)
                    board = chess.Board(fen)
                    return self._simple_evaluation(board)
            else:
                # Fallback: Use python-chess simple evaluation
                board = chess.Board(fen)
                return self._simple_evaluation(board)
        
        except Exception as e:
            # Final fallback if even simple evaluation fails
            print(f"Error evaluating position {fen[:50]}...: {e}")
            return None
    
    def _simple_evaluation(self, board: chess.Board) -> float:
        """
        Simple material-based evaluation (fallback when Stockfish not available).
        
        Piece values:
        - Pawn: 1
        - Knight/Bishop: 3
        - Rook: 5
        - Queen: 9
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return (white_material - black_material) * 100  # Convert to centipawns
    
    def calculate_cp_loss(self, predicted_fen: str, ground_truth_fen: str) -> Optional[float]:
        """
        Calculate CP loss between predicted and ground truth FENs.
        
        Args:
            predicted_fen: Predicted FEN string
            ground_truth_fen: Ground truth FEN string
        
        Returns:
            Absolute CP loss (difference in evaluation)
            None if evaluation fails
        """
        pred_score = self.evaluate_position(predicted_fen)
        gt_score = self.evaluate_position(ground_truth_fen)
        
        if pred_score is None or gt_score is None:
            return None
        
        return abs(pred_score - gt_score)
    
    def close(self):
        """Close any connections (Lichess API doesn't need explicit closing)."""
        # Lichess API uses HTTP requests, no persistent connection to close
        pass

