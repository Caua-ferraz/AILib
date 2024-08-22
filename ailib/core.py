# D:\AiProject\ailib\core.py

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

class AIModel:
    def __init__(self, model_type: str, hyperparameters: dict = None, custom_model: BaseEstimator = None):
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        if self.model_type == 'neural_network' and 'max_iter' not in self.hyperparameters:
            self.hyperparameters['max_iter'] = 1000
        if custom_model is not None:
            self.model = custom_model
        else:
            self.model = self._create_model()

    def _create_model(self):
        if self.model_type == 'neural_network':
            return MLPClassifier(**self.hyperparameters)
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(**self.hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str):
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> 'AIModel':
        import joblib
        loaded_model = joblib.load(path)
        instance = cls(model_type='custom', custom_model=loaded_model)
        return instance