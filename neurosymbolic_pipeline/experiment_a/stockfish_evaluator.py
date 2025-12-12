"""
Stockfish evaluation module for Experiment A.

Evaluates chess positions using Stockfish engine and calculates CP (centipawn) loss.

Priority order:
1. Local Stockfish binary (via python-chess) - Most accurate, recommended
2. Lichess Cloud Evaluation API - May be unavailable (404 errors reported)
3. Python-chess simple material evaluation - Fallback, less accurate

Note: Lichess cloud-eval endpoint appears to be deprecated/removed (returns 404).
For best results, install Stockfish binary locally.
"""

import chess
import chess.engine
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import shutil

# Import Lichess API evaluator from existing benchmarking code (optional)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarking"))
try:
    from ground_truth import GroundTruthExtractor
except ImportError:
    GroundTruthExtractor = None


class StockfishEvaluator:
    """Evaluates chess positions using Lichess Cloud Evaluation API (Stockfish)."""
    
    def __init__(self, stockfish_path: Optional[str] = None, use_lichess_api: bool = False, depth: int = 15):
        """
        Initialize Stockfish evaluator.
        
        Args:
            stockfish_path: Path to Stockfish executable (None = auto-detect or use fallback)
            use_lichess_api: If True, try Lichess API as fallback (may be unavailable)
            depth: Search depth for evaluation
        """
        self.depth = depth
        self.stockfish_path = stockfish_path
        self.engine = None
        self.use_lichess_api = use_lichess_api
        self.lichess_extractor = None
        
        # Priority 1: Try to initialize local Stockfish engine
        if stockfish_path and Path(stockfish_path).exists():
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Using local Stockfish binary at {stockfish_path} (depth={depth})")
                return
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish at {stockfish_path}: {e}")
        
        # Auto-detect Stockfish in PATH
        stockfish_cmd = shutil.which("stockfish")
        if stockfish_cmd:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_cmd)
                print(f"Using Stockfish from PATH: {stockfish_cmd} (depth={depth})")
                return
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish from PATH: {e}")
        
        # Priority 2: Try Lichess API (may be unavailable)
        if use_lichess_api and GroundTruthExtractor is not None:
            try:
                self.lichess_extractor = GroundTruthExtractor()
                print(f"Using Lichess Cloud Evaluation API (depth={depth})")
                print("Note: Lichess API may return 404 errors - endpoint may be deprecated")
                return
            except Exception as e:
                print(f"Warning: Could not initialize Lichess API: {e}")
        
        # Priority 3: Fallback to simple evaluation
        print("Using python-chess built-in evaluation (simple material-based)")
        print("For accurate CP loss, install Stockfish: https://stockfishchess.org/download/")
    
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
            board = chess.Board(fen)
            
            # Priority 1: Local Stockfish engine
            if self.engine is not None:
                try:
                    info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                    if 'score' in info:
                        score = info['score']
                        if score.is_mate():
                            # Convert mate score to large CP value
                            mate_ply = score.mate()
                            return 10000 if mate_ply > 0 else -10000
                        else:
                            return score.score()  # Already in centipawns
                except Exception as e:
                    print(f"Error with Stockfish engine: {e}")
                    # Fall through to next method
            
            # Priority 2: Lichess API (may be unavailable)
            if self.use_lichess_api and self.lichess_extractor is not None:
                try:
                    eval_result = self.lichess_extractor.get_position_evaluation(fen, depth=self.depth)
                    if eval_result is not None:
                        # Handle mate scores
                        if eval_result.get('mate') is not None:
                            mate_ply = eval_result['mate']
                            return 10000 if mate_ply > 0 else -10000
                        # Return CP score (already in centipawns)
                        return eval_result.get('score', 0)
                except Exception:
                    # API failed, fall through to simple evaluation
                    pass
            
            # Priority 3: Simple material-based evaluation
            return self._simple_evaluation(board)
        
        except Exception as e:
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
        """Close the engine connection."""
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass

