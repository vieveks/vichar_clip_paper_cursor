"""
Data processing module for enriching chess datasets with multiple representations.
"""

from .representations import (
    fen_to_grid,
    board_to_json,
    board_to_graph,
    board_to_natural_language,
    analyze_tactics
)

from .converters import (
    json_to_fen,
    json_to_board_fen,
    validate_json_position,
    round_trip_test,
    batch_round_trip_test
)

__all__ = [
    # FEN to various representations
    'fen_to_grid',
    'board_to_json',
    'board_to_graph',
    'board_to_natural_language',
    'analyze_tactics',
    # JSON to FEN converters
    'json_to_fen',
    'json_to_board_fen',
    'validate_json_position',
    'round_trip_test',
    'batch_round_trip_test'
]

