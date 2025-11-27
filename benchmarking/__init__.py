"""
Chess VLM Benchmarking Package

This package provides tools for benchmarking Vision Language Models (VLMs)
on chess-related questions, comparing performance with and without FEN context.
"""

from .benchmark import ChessVLMBenchmark
from .questions import QUESTIONS, get_all_questions, get_scoring_questions
from .clip_fen_extractor import CLIPFENExtractor
from .ground_truth import GroundTruthExtractor
from .vlm_integration import VLMInterface, MockVLMInterface
from .scoring import ResponseScorer

__all__ = [
    'ChessVLMBenchmark',
    'QUESTIONS',
    'get_all_questions',
    'get_scoring_questions',
    'CLIPFENExtractor',
    'GroundTruthExtractor',
    'VLMInterface',
    'MockVLMInterface',
    'ResponseScorer'
]

