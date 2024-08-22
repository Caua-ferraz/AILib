# D:\AiProject\ailib\unified_model.py

import os
import tempfile
from typing import Union, List, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator
from .core import AIModel
from .llm import LLM

class UnifiedModel:
    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        if model_type in ['neural_network', 'decision_tree']:
            self.model = AIModel(model_type, hyperparameters=kwargs)
        elif model_type == 'custom':
            custom_model = kwargs.pop('custom_model', None)
            self.model = AIModel(model_type, hyperparameters=kwargs, custom_model=custom_model)
        elif model_type == 'llm':
            self.model = LLM(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X: Union[np.ndarray, List[str]], y: Union[np.ndarray, List[str]] = None):
        if isinstance(self.model, AIModel):
            self.model.train(X, y)
        elif isinstance(self.model, LLM):
            raise ValueError("For LLM models, use the 'train_llm' method instead.")
        else:
            raise ValueError("Unsupported model for training")

    def train_llm(self, train_texts: List[str], train_labels: List[str] = None, **kwargs):
        if isinstance(self.model, LLM):
            # Create a temporary file for training data
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix='.txt') as temp_file:
                for text in train_texts:
                    temp_file.write(text + "\n")
                temp_file_path = temp_file.name

            try:
                # Pass the temporary file path to fine_tune
                self.model.fine_tune(temp_file_path, train_labels, **kwargs)
            finally:
                # Remove the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        else:
            raise ValueError("This method is only for LLM models")

    def predict(self, X: Union[np.ndarray, str, List[str]], **kwargs) -> Union[np.ndarray, List[str]]:
        if isinstance(self.model, AIModel):
            return self.model.predict(X)
        elif isinstance(self.model, LLM):
            if isinstance(X, str):
                return self.model.generate_text(X, **kwargs)
            elif isinstance(X, list):
                return [self.model.generate_text(prompt, **kwargs)[0] for prompt in X]

    def evaluate(self, X: Union[np.ndarray, List[str]], y: Union[np.ndarray, List[str]] = None) -> Dict[str, float]:
        if isinstance(self.model, AIModel):
            from .evaluation import evaluate_model
            return evaluate_model(self.model, X, y)
        else:
            print("Evaluation for LLM models is not implemented yet.")
            return {}

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, model_type: str, path: str) -> 'UnifiedModel':
        if model_type in ['neural_network', 'decision_tree', 'custom']:
            loaded_model = AIModel.load(path)
            return cls(model_type='custom', custom_model=loaded_model.model)
        elif model_type == 'llm':
            loaded_model = LLM.load_model(path)
            return cls(model_type='llm', model_name=loaded_model.model_name)
        else:
            raise ValueError(f"Unsupported model type for loading: {model_type}")

    def tokenize(self, text: str) -> List[str]:
        if isinstance(self.model, LLM):
            return self.model.tokenize(text)
        else:
            raise ValueError("Tokenization is only available for LLM models")

    def get_token_ids(self, text: str) -> List[int]:
        if isinstance(self.model, LLM):
            return self.model.get_token_ids(text)
        else:
            raise ValueError("Getting token IDs is only available for LLM models")