# D:\AiProject\ailib\__init__.py

from .core import AIModel
from .data_processing import preprocess_data, split_data
from .model_training import train_model, fine_tune
from .evaluation import evaluate_model, cross_validate
from .llm import LLM
from .unified_model import UnifiedModel

__all__ = ['AIModel', 'preprocess_data', 'split_data', 'train_model', 'fine_tune', 
           'evaluate_model', 'cross_validate', 'LLM', 'UnifiedModel']

__version__ = '0.3.0'