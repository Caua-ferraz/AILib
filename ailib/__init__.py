"""
AILib - A library for AI model management, training, evaluation, and deployment.

Version: 0.3.0
"""

from .core import AIModel
from .data_processing import preprocess_data, split_data
from .evaluation import evaluate_model, cross_validate_model
from .llm import LLM
from .model_training import train_model, fine_tune_model
from .unified_model import UnifiedModel

__all__ = [
    'AIModel',
    'preprocess_data',
    'split_data',
    'train_model',
    'fine_tune_model',
    'evaluate_model',
    'cross_validate_model',
    'LLM',
    'UnifiedModel'
]

__version__ = '0.3.0'
