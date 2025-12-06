"""
Deterministic converters between JSON and FEN representations.

This module provides:
- json_to_fen(): Convert JSON representation back to FEN string
- validate_json_position(): Validate JSON structure for chess rules
- round_trip_test(): Test FEN→JSON→FEN consistency
"""

import chess
from typing import Dict, List, Optional, Tuple
from .representations import board_to_json, fen_to_grid


# Reverse mapping: piece name to FEN character
PIECE_NAME_TO_FEN = {
    'white_pawn': 'P', 'white_knight': 'N', 'white_bishop': 'B',
    'white_rook': 'R', 'white_queen': 'Q', 'white_king': 'K',
    'black_pawn': 'p', 'black_knight': 'n', 'black_bishop': 'b',
    'black_rook': 'r', 'black_queen': 'q', 'black_king': 'k'
}

# Square name to index (a1=0, b1=1, ..., h8=63)
def square_name_to_index(square_name: str) -> Tuple[int, int]:
    """Convert square name (e.g., 'e4') to (file, rank) indices."""
    file_idx = ord(square_name[0]) - ord('a')  # 0-7
    rank_idx = int(square_name[1]) - 1  # 0-7
    return file_idx, rank_idx


def json_to_fen(json_repr: Dict) -> str:
    """
    Convert JSON representation back to FEN string.
    
    Args:
        json_repr: Dictionary with 'pieces' list and 'metadata'
    
    Returns:
        FEN string (board placement + active color + castling + en passant)
    
    Example:
        Input: {"pieces": [{"piece": "white_king", "square": "e1"}, ...], "metadata": {...}}
        Output: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    """
    # Initialize empty 8x8 board
    board = [[None for _ in range(8)] for _ in range(8)]
    
    # Place pieces on board
    for piece_info in json_repr['pieces']:
        piece_name = piece_info['piece']
        square = piece_info['square']
        
        file_idx, rank_idx = square_name_to_index(square)
        fen_char = PIECE_NAME_TO_FEN.get(piece_name)
        
        if fen_char:
            board[7 - rank_idx][file_idx] = fen_char  # Rank 8 is index 0
    
    # Convert board to FEN string
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        
        for cell in row:
            if cell is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_rows.append(fen_row)
    
    board_fen = "/".join(fen_rows)
    
    # Add metadata if available
    metadata = json_repr.get('metadata', {})
    
    # Active color
    to_move = metadata.get('to_move', 'white')
    active_color = 'w' if to_move == 'white' else 'b'
    
    # Castling rights
    castling_rights = metadata.get('castling_rights', {'white': [], 'black': []})
    castling_str = ""
    if 'K' in castling_rights.get('white', []):
        castling_str += 'K'
    if 'Q' in castling_rights.get('white', []):
        castling_str += 'Q'
    if 'k' in castling_rights.get('black', []):
        castling_str += 'k'
    if 'q' in castling_rights.get('black', []):
        castling_str += 'q'
    if not castling_str:
        castling_str = '-'
    
    # En passant
    en_passant = metadata.get('en_passant', None)
    en_passant_str = en_passant if en_passant else '-'
    
    # Construct full FEN (without halfmove and fullmove clocks)
    full_fen = f"{board_fen} {active_color} {castling_str} {en_passant_str}"
    
    return full_fen


def json_to_board_fen(json_repr: Dict) -> str:
    """
    Convert JSON representation to board-only FEN (no metadata).
    
    This is useful for comparing board positions only.
    """
    full_fen = json_to_fen(json_repr)
    return full_fen.split()[0]  # Return only board placement part


def validate_json_position(json_repr: Dict) -> Tuple[bool, List[str]]:
    """
    Validate JSON position against chess rules.
    
    Args:
        json_repr: Dictionary with 'pieces' list
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    pieces = json_repr.get('pieces', [])
    
    # Count pieces by type
    piece_counts = {}
    for piece_info in pieces:
        piece_name = piece_info['piece']
        piece_counts[piece_name] = piece_counts.get(piece_name, 0) + 1
    
    # Check for exactly one king per side
    white_kings = piece_counts.get('white_king', 0)
    black_kings = piece_counts.get('black_king', 0)
    
    if white_kings != 1:
        errors.append(f"Invalid number of white kings: {white_kings} (expected 1)")
    if black_kings != 1:
        errors.append(f"Invalid number of black kings: {black_kings} (expected 1)")
    
    # Check piece limits
    white_pawns = piece_counts.get('white_pawn', 0)
    black_pawns = piece_counts.get('black_pawn', 0)
    
    if white_pawns > 8:
        errors.append(f"Too many white pawns: {white_pawns} (max 8)")
    if black_pawns > 8:
        errors.append(f"Too many black pawns: {black_pawns} (max 8)")
    
    # Check total pieces per side (max 16)
    white_total = sum(v for k, v in piece_counts.items() if k.startswith('white_'))
    black_total = sum(v for k, v in piece_counts.items() if k.startswith('black_'))
    
    if white_total > 16:
        errors.append(f"Too many white pieces: {white_total} (max 16)")
    if black_total > 16:
        errors.append(f"Too many black pieces: {black_total} (max 16)")
    
    # Check for pawns on first/last rank
    for piece_info in pieces:
        if piece_info['type'] == 'pawn':
            rank = piece_info['square'][1]
            if rank in ['1', '8']:
                errors.append(f"Pawn on invalid rank: {piece_info['square']}")
    
    # Check for duplicate squares
    squares = [p['square'] for p in pieces]
    if len(squares) != len(set(squares)):
        errors.append("Duplicate pieces on same square")
    
    return len(errors) == 0, errors


def round_trip_test(fen_string: str, verbose: bool = False) -> Tuple[bool, str, str]:
    """
    Test FEN→JSON→FEN round-trip consistency.
    
    Args:
        fen_string: Original FEN string
        verbose: Print debug information
    
    Returns:
        Tuple of (passed, original_board_fen, reconstructed_board_fen)
    """
    # Extract board-only FEN (ignore metadata for comparison)
    original_board = fen_string.split()[0] if ' ' in fen_string else fen_string
    
    # Convert to JSON
    try:
        json_repr = board_to_json(fen_string)
    except Exception as e:
        if verbose:
            print(f"Error converting FEN to JSON: {e}")
        return False, original_board, f"ERROR: {e}"
    
    # Validate JSON
    is_valid, errors = validate_json_position(json_repr)
    if not is_valid and verbose:
        print(f"JSON validation errors: {errors}")
    
    # Convert back to FEN
    try:
        reconstructed_fen = json_to_fen(json_repr)
        reconstructed_board = reconstructed_fen.split()[0]
    except Exception as e:
        if verbose:
            print(f"Error converting JSON to FEN: {e}")
        return False, original_board, f"ERROR: {e}"
    
    # Compare board positions
    passed = original_board == reconstructed_board
    
    if verbose and not passed:
        print(f"Round-trip FAILED:")
        print(f"  Original:      {original_board}")
        print(f"  Reconstructed: {reconstructed_board}")
    
    return passed, original_board, reconstructed_board


def batch_round_trip_test(fen_list: List[str], verbose: bool = False) -> Dict:
    """
    Test round-trip on a batch of FEN strings.
    
    Returns:
        Dictionary with test results
    """
    results = {
        'total': len(fen_list),
        'passed': 0,
        'failed': 0,
        'failed_examples': []
    }
    
    for fen in fen_list:
        passed, original, reconstructed = round_trip_test(fen, verbose=False)
        
        if passed:
            results['passed'] += 1
        else:
            results['failed'] += 1
            if len(results['failed_examples']) < 10:  # Keep first 10 failures
                results['failed_examples'].append({
                    'original': original,
                    'reconstructed': reconstructed
                })
    
    results['accuracy'] = results['passed'] / results['total'] if results['total'] > 0 else 0.0
    
    if verbose:
        print(f"Round-trip test results:")
        print(f"  Total: {results['total']}")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    
    return results


# Test function
def test_converters():
    """Run basic tests on converters."""
    # Test cases
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",  # Starting position
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR",  # After 1.e4
        "r3k2r/ppb2p1p/2nqpp2/1B1p3b/Q2N4/7P/PP1N1PP1/R1B2RK1",  # Complex
        "8/8/4k3/1pB1p1n1/1P2P1p1/6K1/6P1/8",  # Endgame
    ]
    
    print("Testing FEN↔JSON converters...")
    print("="*50)
    
    for fen in test_fens:
        passed, original, reconstructed = round_trip_test(fen, verbose=True)
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {fen[:30]}...")
        print()
    
    print("="*50)
    print("All tests complete.")


if __name__ == "__main__":
    test_converters()

