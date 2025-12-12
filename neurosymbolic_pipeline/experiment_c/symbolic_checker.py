"""
Symbolic Checker for Experiment C.

Provides rule-based answers to logic-based chess questions using FEN strings.
"""

import chess
from typing import Dict, List, Optional


class SymbolicChecker:
    """Rule-based checker for chess logic questions."""
    
    def __init__(self):
        """Initialize symbolic checker."""
        pass
    
    def check_status(self, fen: str) -> Dict[str, bool]:
        """
        Determine check status for both kings.
        
        Args:
            fen: FEN string representing the position
        
        Returns:
            Dict with 'white_in_check', 'black_in_check', 'is_check'
        """
        try:
            board = chess.Board(fen)
            
            # Check if current side to move is in check
            is_check = board.is_check()
            
            # Determine which side is to move
            if board.turn == chess.WHITE:
                white_in_check = is_check
                # Check if black king is in check (by temporarily switching turn)
                board.turn = chess.BLACK
                black_in_check = board.is_check()
                board.turn = chess.WHITE  # Restore
            else:
                black_in_check = is_check
                # Check if white king is in check
                board.turn = chess.WHITE
                white_in_check = board.is_check()
                board.turn = chess.BLACK  # Restore
            
            return {
                'white_in_check': white_in_check,
                'black_in_check': black_in_check,
                'is_check': is_check
            }
        except Exception as e:
            print(f"Error checking status for FEN {fen[:50]}...: {e}")
            return {
                'white_in_check': False,
                'black_in_check': False,
                'is_check': False
            }
    
    def castling_rights(self, fen: str) -> Dict[str, bool]:
        """
        Get castling rights for both sides.
        
        Args:
            fen: FEN string
        
        Returns:
            Dict with castling rights for white and black
        """
        try:
            board = chess.Board(fen)
            
            return {
                'white_kingside': board.has_kingside_castling_rights(chess.WHITE),
                'white_queenside': board.has_queenside_castling_rights(chess.WHITE),
                'black_kingside': board.has_kingside_castling_rights(chess.BLACK),
                'black_queenside': board.has_queenside_castling_rights(chess.BLACK)
            }
        except Exception as e:
            print(f"Error getting castling rights for FEN {fen[:50]}...: {e}")
            return {
                'white_kingside': False,
                'white_queenside': False,
                'black_kingside': False,
                'black_queenside': False
            }
    
    def piece_location(self, fen: str, square: str) -> str:
        """
        Get piece on a specific square.
        
        Args:
            fen: FEN string
            square: Square name (e.g., 'e4')
        
        Returns:
            Piece description (e.g., 'White Knight', 'Black Pawn', 'Empty')
        """
        try:
            board = chess.Board(fen)
            square_obj = chess.parse_square(square)
            piece = board.piece_at(square_obj)
            
            if piece is None:
                return 'Empty'
            
            color = 'White' if piece.color == chess.WHITE else 'Black'
            piece_type = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            
            piece_names = {
                'P': 'Pawn', 'p': 'Pawn',
                'N': 'Knight', 'n': 'Knight',
                'B': 'Bishop', 'b': 'Bishop',
                'R': 'Rook', 'r': 'Rook',
                'Q': 'Queen', 'q': 'Queen',
                'K': 'King', 'k': 'King'
            }
            
            piece_name = piece_names.get(piece_type, 'Unknown')
            return f"{color} {piece_name}"
        
        except Exception as e:
            print(f"Error getting piece location for {square} in FEN {fen[:50]}...: {e}")
            return 'Unknown'
    
    def legal_moves(self, fen: str) -> List[str]:
        """
        Get all legal moves from the position.
        
        Args:
            fen: FEN string
        
        Returns:
            List of legal moves in UCI notation
        """
        try:
            board = chess.Board(fen)
            return [move.uci() for move in board.legal_moves]
        except Exception as e:
            print(f"Error getting legal moves for FEN {fen[:50]}...: {e}")
            return []
    
    def material_count(self, fen: str, color: Optional[str] = None) -> Dict[str, int]:
        """
        Get material count for both sides or a specific color.
        
        Args:
            fen: FEN string
            color: 'white', 'black', or None for both
        
        Returns:
            Material count dict
        """
        try:
            board = chess.Board(fen)
            
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
            
            if color == 'white':
                return {'white': white_material}
            elif color == 'black':
                return {'black': black_material}
            else:
                return {'white': white_material, 'black': black_material}
        
        except Exception as e:
            print(f"Error calculating material for FEN {fen[:50]}...: {e}")
            return {'white': 0, 'black': 0}

