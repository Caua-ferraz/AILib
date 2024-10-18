# D:\AiProject\ailib\core.py

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from .error_handling import AILibError, UnsupportedModelTypeError, ModelNotTrainedError


class AIModel:
    def __init__(
        self,
        model_type: str,
        hyperparameters: dict = None,
        custom_model: BaseEstimator = None
    ):
        self.model_type = model_type.lower()
        self.hyperparameters = hyperparameters or {}
        if self.model_type == 'neural_network' and 'max_iter' not in self.hyperparameters:
            self.hyperparameters['max_iter'] = 1000
        if custom_model is not None:
            self.model = custom_model
        else:
            self.model = self._create_model()

    def _create_model(self) -> BaseEstimator:
        if self.model_type == 'neural_network':
            return MLPClassifier(**self.hyperparameters)
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(**self.hyperparameters)
        else:
            raise UnsupportedModelTypeError(self.model_type)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict"):
            raise ModelNotTrainedError()
        return self.model.predict(X)

    def save(self, path: str):
        try:
            joblib.dump({
                'model_type': self.model_type,
                'hyperparameters': self.hyperparameters,
                'model': self.model
            }, path)
        except Exception as e:
            raise AILibError(f"Saving model failed: {e}")

    @classmethod
    def load(cls, path: str) -> 'AIModel':
        try:
            data = joblib.load(path)
            return cls(
                model_type=data['model_type'],
                hyperparameters=data['hyperparameters'],
                custom_model=data['model']
            )
        except Exception as e:
            raise AILibError(f"Loading model failed: {e}")
