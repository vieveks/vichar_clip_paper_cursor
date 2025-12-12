"""
Stockfish evaluation module for chess positions.

Priority order:
1. Local Stockfish binary (most accurate)
2. Lichess Cloud Eval API (free, high depth, works great)
3. Python-chess simple material evaluation (fallback)

Note: Stockfish Online API is currently unreliable (returning 500 errors).
"""

import chess
import chess.engine
import requests
from pathlib import Path
from typing import Optional, Dict, Tuple
import shutil
import urllib.parse


class StockfishEvaluator:
    """Evaluates chess positions using Stockfish engine or Lichess API."""
    
    def __init__(
        self, 
        stockfish_path: Optional[str] = None, 
        depth: int = 15
    ):
        """
        Initialize Stockfish evaluator.
        
        Args:
            stockfish_path: Path to Stockfish executable (None = auto-detect)
            depth: Search depth for local evaluation (Lichess uses cloud depth)
        """
        self.depth = depth
        self.engine = None
        self._lichess_available = True  # Assume available until proven otherwise
        self._api_failures = 0
        
        # Try to initialize local Stockfish engine
        if stockfish_path and Path(stockfish_path).exists():
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"[OK] Using local Stockfish at {stockfish_path} (depth={depth})")
                return
            except Exception as e:
                print(f"[WARN] Could not initialize Stockfish at {stockfish_path}: {e}")
        
        # Auto-detect Stockfish in PATH
        stockfish_cmd = shutil.which("stockfish")
        if stockfish_cmd:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_cmd)
                print(f"[OK] Using Stockfish from PATH: {stockfish_cmd} (depth={depth})")
                return
            except Exception as e:
                print(f"[WARN] Could not initialize Stockfish from PATH: {e}")
        
        print("[INFO] Using Lichess Cloud Eval API (high-quality, depth 40-70)")
        print("       For local evaluation, install Stockfish: https://stockfishchess.org/download/")
    
    def _evaluate_via_lichess(self, fen: str) -> Optional[float]:
        """
        Evaluate position using Lichess Cloud Eval API.
        
        Args:
            fen: FEN string
            
        Returns:
            CP score in centipawns, or None if evaluation fails
        """
        if not self._lichess_available:
            return None
        
        try:
            # Properly encode the FEN
            encoded_fen = urllib.parse.quote(fen, safe='')
            url = f"https://lichess.org/api/cloud-eval?fen={encoded_fen}"
            
            response = requests.get(
                url,
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'ChessEvaluator/1.0'
                },
                timeout=10
            )
            
            if response.status_code == 429:
                print("[WARN] Lichess rate limited, waiting...")
                import time
                time.sleep(2)
                return None
            
            if response.status_code == 404:
                # Position not in cloud database - this is normal for uncommon positions
                return None
            
            if response.status_code != 200:
                self._api_failures += 1
                if self._api_failures > 5:
                    print("[WARN] Lichess API failing too often, disabling")
                    self._lichess_available = False
                return None
            
            data = response.json()
            
            # Lichess response format:
            # {"fen": "...", "knodes": 123, "depth": 40, "pvs": [{"moves": "e2e4 e7e5", "cp": 18}]}
            if 'pvs' in data and len(data['pvs']) > 0:
                pv = data['pvs'][0]
                
                # Check for mate
                if 'mate' in pv:
                    mate_in = pv['mate']
                    return 10000 if mate_in > 0 else -10000
                
                # Return centipawn score
                if 'cp' in pv:
                    return pv['cp']
            
            return None
            
        except requests.Timeout:
            print("[WARN] Lichess API timeout")
            return None
        except Exception as e:
            print(f"[WARN] Lichess API error: {e}")
            return None
    
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
            
            # Priority 1: Local Stockfish engine (most accurate)
            if self.engine is not None:
                try:
                    info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
                    if 'score' in info:
                        score = info['score'].white()  # Get score from white's perspective
                        if score.is_mate():
                            mate_ply = score.mate()
                            return 10000 if mate_ply > 0 else -10000
                        else:
                            return score.score()
                except Exception as e:
                    print(f"[WARN] Stockfish engine error: {e}")
            
            # Priority 2: Lichess Cloud Eval API
            lichess_score = self._evaluate_via_lichess(fen)
            if lichess_score is not None:
                return lichess_score
            
            # Priority 3: Simple material evaluation
            return self._simple_evaluation(board)
        
        except chess.InvalidFenError as e:
            print(f"[ERROR] Invalid FEN: {fen[:50]}... - {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Error evaluating position: {e}")
            return None
    
    def _simple_evaluation(self, board: chess.Board) -> float:
        """Simple material-based evaluation (fallback)."""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def calculate_cp_loss(self, predicted_fen: str, ground_truth_fen: str) -> Optional[float]:
        """Calculate CP loss between predicted and ground truth FENs."""
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
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Simple standalone function for quick evaluation
def evaluate_fen(fen: str) -> Optional[float]:
    """Quick evaluation of a single FEN position using Lichess API."""
    try:
        encoded_fen = urllib.parse.quote(fen, safe='')
        url = f"https://lichess.org/api/cloud-eval?fen={encoded_fen}"
        
        response = requests.get(url, timeout=10, headers={'Accept': 'application/json'})
        
        if response.status_code == 200:
            data = response.json()
            if 'pvs' in data and len(data['pvs']) > 0:
                pv = data['pvs'][0]
                if 'mate' in pv:
                    return 10000 if pv['mate'] > 0 else -10000
                if 'cp' in pv:
                    return pv['cp']
        
        return None
    except:
        return None


if __name__ == "__main__":
    # Test the evaluator
    test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    
    with StockfishEvaluator() as evaluator:
        score = evaluator.evaluate_position(test_fen)
        print(f"\nTest FEN: {test_fen}")
        if score is not None:
            print(f"Evaluation: {score} centipawns ({score/100:.2f} pawns)")
        else:
            print("Evaluation failed")
