"""
Page-wise FEN Extraction Pipeline

A complete pipeline for extracting chess board diagrams from PDF files or images
and generating FEN (Forsyth–Edwards Notation) using OpenAI's GPT Vision models.
"""

from .board_extractor import (
    extract_boards_from_pdf_pages,
    extract_boards_from_single_image,
    extract_boards_from_image,
    save_board_crop
)

from .fen_generator import (
    generate_fen_from_image_array,
    generate_fen_from_file,
    get_openai_client
)

from .page_fen_processor import (
    process_pdf_to_page_fens,
    process_image_to_fens,
    save_results_to_json
)

__all__ = [
    # Board extraction functions
    'extract_boards_from_pdf_pages',
    'extract_boards_from_single_image',
    'extract_boards_from_image',
    'save_board_crop',
    
    # FEN generation functions
    'generate_fen_from_image_array',
    'generate_fen_from_file',
    'get_openai_client',
    
    # Main processing functions
    'process_pdf_to_page_fens',
    'process_image_to_fens',
    'save_results_to_json',
]

__version__ = '1.0.0'
__author__ = 'Auto-generated from existing chess analysis tools'

