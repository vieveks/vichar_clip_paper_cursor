"""
JSON Predictor module for chess board position prediction.

This module provides:
- JSONPredictorModel: CLIP-based model to predict JSON representation from images
- JSONDataset: Dataset loader for training
- Training and evaluation scripts
"""

from .model import JSONPredictorModel
from .dataset import JSONDataset, create_json_dataloaders

__all__ = [
    'JSONPredictorModel',
    'JSONDataset',
    'create_json_dataloaders'
]

