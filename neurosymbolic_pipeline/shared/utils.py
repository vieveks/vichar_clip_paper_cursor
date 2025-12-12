"""
Shared utilities for neurosymbolic pipeline.

These utilities import from existing code (read-only access).
Original files remain untouched.
"""

import sys
from pathlib import Path
import importlib.util

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
IMPROVED_REP_PATH = PROJECT_ROOT / "Improved_representations"

# Add to sys.path for proper module resolution
sys.path.insert(0, str(IMPROVED_REP_PATH))
sys.path.insert(0, str(IMPROVED_REP_PATH / "data_processing"))
sys.path.insert(0, str(IMPROVED_REP_PATH / "json_predictor"))

# Import needed functions using proper module loading
try:
    # Try importing as packages first
    from data_processing.converters import json_to_fen, validate_json_position
    from json_predictor.dataset import grid_to_json, IDX_TO_PIECE, PIECE_TO_IDX, idx_to_square_name, square_name_to_idx
except ImportError:
    # Fallback: load modules directly and handle relative imports
    converters_path = IMPROVED_REP_PATH / "data_processing" / "converters.py"
    representations_path = IMPROVED_REP_PATH / "data_processing" / "representations.py"
    dataset_path = IMPROVED_REP_PATH / "json_predictor" / "dataset.py"
    
    # Load representations first (needed by converters)
    if representations_path.exists():
        spec = importlib.util.spec_from_file_location("data_processing.representations", representations_path)
        representations = importlib.util.module_from_spec(spec)
        sys.modules['data_processing'] = type(sys)('data_processing')
        sys.modules['data_processing.representations'] = representations
        spec.loader.exec_module(representations)
    
    # Load converters
    if converters_path.exists():
        spec = importlib.util.spec_from_file_location("data_processing.converters", converters_path)
        converters = importlib.util.module_from_spec(spec)
        sys.modules['data_processing.converters'] = converters
        spec.loader.exec_module(converters)
        json_to_fen = converters.json_to_fen
        validate_json_position = converters.validate_json_position
    
    # Load dataset
    if dataset_path.exists():
        spec = importlib.util.spec_from_file_location("json_predictor.dataset", dataset_path)
        dataset = importlib.util.module_from_spec(spec)
        sys.modules['json_predictor'] = type(sys)('json_predictor')
        sys.modules['json_predictor.dataset'] = dataset
        spec.loader.exec_module(dataset)
        grid_to_json = dataset.grid_to_json
        IDX_TO_PIECE = dataset.IDX_TO_PIECE
        PIECE_TO_IDX = dataset.PIECE_TO_IDX
        idx_to_square_name = dataset.idx_to_square_name
        square_name_to_idx = dataset.square_name_to_idx

__all__ = [
    'json_to_fen',
    'validate_json_position', 
    'grid_to_json',
    'IDX_TO_PIECE',
    'PIECE_TO_IDX',
    'idx_to_square_name',
    'square_name_to_idx'
]

