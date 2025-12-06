"""
Representation conversion functions for chess positions.

This module provides functions to convert FEN strings to various representations:
- Grid: 8×8 integer matrix
- JSON: Structured format with pieces, relationships, metadata
- Graph: NetworkX graph representation
- Natural Language: Descriptive text
- Tactics: Tactical pattern analysis
"""

import chess
import chess.engine
import json
from typing import Dict, List, Optional, Tuple
import networkx as nx


# Piece value mapping
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Piece type to string mapping
PIECE_TYPES = {
    chess.PAWN: 'pawn',
    chess.KNIGHT: 'knight',
    chess.BISHOP: 'bishop',
    chess.ROOK: 'rook',
    chess.QUEEN: 'queen',
    chess.KING: 'king'
}

# Piece to index mapping for grid (0=empty, 1-6=white pieces, 7-12=black pieces)
PIECE_TO_INDEX = {
    None: 0,
    chess.Piece(chess.PAWN, chess.WHITE): 1,
    chess.Piece(chess.KNIGHT, chess.WHITE): 2,
    chess.Piece(chess.BISHOP, chess.WHITE): 3,
    chess.Piece(chess.ROOK, chess.WHITE): 4,
    chess.Piece(chess.QUEEN, chess.WHITE): 5,
    chess.Piece(chess.KING, chess.WHITE): 6,
    chess.Piece(chess.PAWN, chess.BLACK): 7,
    chess.Piece(chess.KNIGHT, chess.BLACK): 8,
    chess.Piece(chess.BISHOP, chess.BLACK): 9,
    chess.Piece(chess.ROOK, chess.BLACK): 10,
    chess.Piece(chess.QUEEN, chess.BLACK): 11,
    chess.Piece(chess.KING, chess.BLACK): 12,
}


def fen_to_grid(fen_string: str) -> List[List[int]]:
    """
    Convert FEN string to 8×8 integer matrix.
    
    Args:
        fen_string: FEN string (board placement part or full FEN)
    
    Returns:
        8×8 grid where each cell is:
        0 = empty
        1-6 = white pieces (pawn, knight, bishop, rook, queen, king)
        7-12 = black pieces (pawn, knight, bishop, rook, queen, king)
    """
    # Parse FEN (take only board placement part if full FEN provided)
    board_part = fen_string.split()[0] if ' ' in fen_string else fen_string
    board = chess.Board()
    board.set_board_fen(board_part)
    
    grid = []
    # Chess ranks go from 8 to 1 (top to bottom)
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):  # Files a-h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            
            if piece is None:
                row.append(0)
            else:
                # Map piece to index
                piece_key = chess.Piece(piece.piece_type, piece.color)
                row.append(PIECE_TO_INDEX.get(piece_key, 0))
        grid.append(row)
    
    return grid


def board_to_json(fen_string: str) -> Dict:
    """
    Convert FEN string to structured JSON representation.
    
    Args:
        fen_string: FEN string (can be full or board placement only)
    
    Returns:
        Dictionary with:
        - pieces: List of all pieces with their squares
        - relationships: List of piece relationships (attacks, defends, pins)
        - metadata: Material counts, castling rights, to_move, etc.
    """
    # Parse FEN
    board = chess.Board(fen_string)
    
    pieces = []
    relationships = []
    
    # Extract all pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            piece_type_str = PIECE_TYPES[piece.piece_type]
            color_str = 'white' if piece.color == chess.WHITE else 'black'
            piece_name = f"{color_str}_{piece_type_str}"
            value = PIECE_VALUES.get(piece.piece_type, 0)
            
            pieces.append({
                'piece': piece_name,
                'square': square_name,
                'color': color_str,
                'type': piece_type_str,
                'value': value
            })
    
    # Extract relationships
    # 1. Attacks (what pieces attack which squares)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            attacked_squares = list(board.attacks(square))
            
            for attacked_sq in attacked_squares:
                attacked_piece = board.piece_at(attacked_sq)
                if attacked_piece:
                    relationships.append({
                        'from_square': square_name,
                        'to_square': chess.square_name(attacked_sq),
                        'type': 'attacks',
                        'piece': f"{'white' if piece.color == chess.WHITE else 'black'}_{PIECE_TYPES[piece.piece_type]}",
                        'target': f"{'white' if attacked_piece.color == chess.WHITE else 'black'}_{PIECE_TYPES[attacked_piece.piece_type]}"
                    })
    
    # 2. Controls (squares controlled by pieces, even if empty)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            square_name = chess.square_name(square)
            controlled_squares = list(board.attacks(square))
            
            for controlled_sq in controlled_squares:
                if board.piece_at(controlled_sq) is None:  # Only empty squares
                    relationships.append({
                        'from_square': square_name,
                        'to_square': chess.square_name(controlled_sq),
                        'type': 'controls',
                        'piece': f"{'white' if piece.color == chess.WHITE else 'black'}_{PIECE_TYPES[piece.piece_type]}"
                    })
    
    # 3. Pins (detect pinned pieces)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Check if this piece is pinned
            king_square = board.king(piece.color)
            if king_square is not None:
                # Try removing this piece and see if king is attacked
                board_copy = board.copy()
                board_copy.remove_piece_at(square)
                
                if board_copy.is_attacked_by(not piece.color, king_square):
                    # This piece is pinned
                    pinning_pieces = board.attackers(not piece.color, king_square)
                    for pinning_sq in pinning_pieces:
                        pinning_piece = board.piece_at(pinning_sq)
                        if pinning_piece:
                            relationships.append({
                                'from_square': chess.square_name(pinning_sq),
                                'to_square': chess.square_name(square),
                                'to_king': chess.square_name(king_square),
                                'type': 'pins',
                                'pinning_piece': f"{'white' if pinning_piece.color == chess.WHITE else 'black'}_{PIECE_TYPES[pinning_piece.piece_type]}",
                                'pinned_piece': f"{'white' if piece.color == chess.WHITE else 'black'}_{PIECE_TYPES[piece.piece_type]}"
                            })
    
    # Calculate material
    white_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.WHITE and p.piece_type != chess.KING
    )
    
    black_material = sum(
        PIECE_VALUES.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.BLACK and p.piece_type != chess.KING
    )
    
    # Get castling rights
    castling_rights = {
        'white': [],
        'black': []
    }
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights['white'].append('K')
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights['white'].append('Q')
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights['black'].append('k')
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights['black'].append('q')
    
    # Get en passant
    en_passant = None
    if board.ep_square is not None:
        en_passant = chess.square_name(board.ep_square)
    
    return {
        'pieces': pieces,
        'relationships': relationships,
        'metadata': {
            'to_move': 'white' if board.turn == chess.WHITE else 'black',
            'castling_rights': castling_rights,
            'en_passant': en_passant,
            'material': {
                'white': white_material,
                'black': black_material
            },
            'material_balance': white_material - black_material
        }
    }


def board_to_graph(fen_string: str) -> Dict:
    """
    Convert FEN string to graph representation using NetworkX.
    
    Args:
        fen_string: FEN string
    
    Returns:
        Dictionary with:
        - nodes: List of node dictionaries (pieces)
        - edges: List of edge dictionaries (relationships)
        - metadata: Position metadata
    """
    # First get JSON representation
    json_repr = board_to_json(fen_string)
    
    # Build graph structure
    nodes = []
    edges = []
    
    # Create nodes from pieces
    for piece_info in json_repr['pieces']:
        nodes.append({
            'id': f"{piece_info['piece']}_{piece_info['square']}",
            'type': piece_info['type'],
            'color': piece_info['color'],
            'square': piece_info['square'],
            'value': piece_info['value']
        })
    
    # Create edges from relationships
    for rel in json_repr['relationships']:
        from_sq = rel['from_square']
        to_sq = rel['to_square']
        rel_type = rel['type']
        
        # Find node IDs
        from_node = next((n['id'] for n in nodes if n['square'] == from_sq), None)
        to_node = next((n['id'] for n in nodes if n['square'] == to_sq), None)
        
        if from_node and to_node:
            edge = {
                'from': from_node,
                'to': to_node,
                'type': rel_type
            }
            # Add additional info for pins
            if rel_type == 'pins' and 'to_king' in rel:
                edge['to_king'] = rel['to_king']
            
            edges.append(edge)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'metadata': json_repr['metadata']
    }


def board_to_natural_language(fen_string: str) -> str:
    """
    Convert FEN string to natural language description.
    
    Args:
        fen_string: FEN string
    
    Returns:
        Natural language description of the position
    """
    board = chess.Board(fen_string)
    json_repr = board_to_json(fen_string)
    
    # Build description
    description_parts = ["Chess Position:\n"]
    
    # White pieces
    white_pieces = [p for p in json_repr['pieces'] if p['color'] == 'white']
    if white_pieces:
        white_list = [f"{p['type']} on {p['square']}" for p in white_pieces]
        description_parts.append(f"White pieces: {', '.join(white_list)}.\n")
    
    # Black pieces
    black_pieces = [p for p in json_repr['pieces'] if p['color'] == 'black']
    if black_pieces:
        black_list = [f"{p['type']} on {p['square']}" for p in black_pieces]
        description_parts.append(f"Black pieces: {', '.join(black_list)}.\n")
    
    # To move
    description_parts.append(f"{json_repr['metadata']['to_move'].capitalize()} to move.\n")
    
    # Material
    material = json_repr['metadata']['material']
    balance = json_repr['metadata']['material_balance']
    if balance > 0:
        description_parts.append(f"White has a material advantage of {balance} points.\n")
    elif balance < 0:
        description_parts.append(f"Black has a material advantage of {abs(balance)} points.\n")
    else:
        description_parts.append("Material is equal.\n")
    
    # Check status
    if board.is_check():
        king_color = 'White' if board.turn == chess.WHITE else 'Black'
        description_parts.append(f"The {king_color.lower()} king is in check!\n")
    
    return "".join(description_parts)


def analyze_tactics(fen_string: str, engine_path: Optional[str] = None) -> Dict:
    """
    Analyze tactical patterns in the position.
    
    Args:
        fen_string: FEN string
        engine_path: Optional path to Stockfish engine for deeper analysis
    
    Returns:
        Dictionary with tactical patterns:
        - pins: List of pinned pieces
        - forks: List of fork opportunities
        - checks: Whether any king is in check
        - threats: List of threats
        - hanging_pieces: List of undefended pieces
    """
    board = chess.Board(fen_string)
    json_repr = board_to_json(fen_string)
    
    tactics = {
        'pins': [],
        'forks': [],
        'checks': board.is_check(),
        'threats': [],
        'hanging_pieces': []
    }
    
    # Extract pins from relationships
    for rel in json_repr['relationships']:
        if rel['type'] == 'pins':
            tactics['pins'].append({
                'pinned_piece': rel.get('pinned_piece', ''),
                'pinning_piece': rel.get('pinning_piece', ''),
                'pinned_to': rel.get('to_king', '')
            })
    
    # Find forks (pieces attacking multiple high-value targets)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == board.turn:
            attacked = board.attacks(square)
            high_value_targets = [
                sq for sq in attacked
                if board.piece_at(sq) and
                   board.piece_at(sq).piece_type in [chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP]
            ]
            
            if len(high_value_targets) >= 2:
                tactics['forks'].append({
                    'forking_piece': chess.square_name(square),
                    'piece_type': PIECE_TYPES[piece.piece_type],
                    'targets': [chess.square_name(sq) for sq in high_value_targets]
                })
    
    # Find hanging pieces (attacked and not defended)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == board.turn:
            attackers = board.attackers(not board.turn, square)
            defenders = board.attackers(board.turn, square)
            
            if len(attackers) > 0 and len(defenders) == 0:
                tactics['hanging_pieces'].append({
                    'square': chess.square_name(square),
                    'piece': f"{'white' if piece.color == chess.WHITE else 'black'}_{PIECE_TYPES[piece.piece_type]}"
                })
    
    # Optional: Use engine for deeper analysis
    if engine_path:
        try:
            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                info = engine.analyse(board, chess.engine.Limit(depth=10))
                tactics['evaluation'] = info['score'].relative.score() if 'score' in info else None
        except:
            pass  # Engine not available, skip
    
    return tactics

