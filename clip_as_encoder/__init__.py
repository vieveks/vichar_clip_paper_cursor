"""
ChessCLIP as Vision Encoder for LLaVA Experiment

This package contains code for testing whether chess-finetuned CLIP
improves LLaVA's chess reasoning compared to generic CLIP.
"""

from .model import ChessLLaVA
from .dataset import ChessQADataset, create_qa_dataset_from_benchmark

__all__ = ['ChessLLaVA', 'ChessQADataset', 'create_qa_dataset_from_benchmark']

