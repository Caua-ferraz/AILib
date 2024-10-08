# D:\AiProject\ailib\unified_model.py

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
            self.model.fine_tune(train_texts, train_labels, **kwargs)
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
        """
        Load a model from a file.
        
        Args:
            model_type (str): Type of the model to load ('neural_network', 'decision_tree', 'custom', or 'llm').
            path (str): Path to the saved model.
        
        Returns:
            UnifiedModel: Loaded model instance.
        
        Raises:
            ValueError: If the model type is not supported.
        """
        if model_type in ['neural_network', 'decision_tree', 'custom']:
            loaded_model = AIModel.load(path)
            return cls(model_type='custom', custom_model=loaded_model.model)
        elif model_type == 'llm':
            loaded_model = LLM.load_model(path)
            return cls(model_type='llm', model_name=loaded_model.model_name)
        else:
            raise ValueError(f"Unsupported model type for loading: {model_type}")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text (str): Input text to tokenize.
        
        Returns:
            List[str]: List of tokens.
        
        Raises:
            ValueError: If tokenization is not available for the current model type.
        """
        if isinstance(self.model, LLM):
            return self.model.tokenize(text)
        else:
            raise ValueError("Tokenization is only available for LLM models")

    def get_token_ids(self, text: str) -> List[int]:
        """
        Get token IDs for the input text.
        
        Args:
            text (str): Input text to convert to token IDs.
        
        Returns:
            List[int]: List of token IDs.
        
        Raises:
            ValueError: If getting token IDs is not available for the current model type.
        """
        if isinstance(self.model, LLM):
            return self.model.get_token_ids(text)
        else:
            raise ValueError("Getting token IDs is only available for LLM models")