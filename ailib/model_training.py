# D:\AiProject\ailib\model_training.py

import numpy as np
from typing import Any, Dict
from sklearn.base import BaseEstimator
from .core import AIModel

def train_model(model: AIModel, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    model.train(X, y)
    return model.model

def fine_tune(model: AIModel, X: np.ndarray, y: np.ndarray, param_grid: Dict[str, Any], cv: int = 5):
    from sklearn.model_selection import GridSearchCV

    if model.model is None:
        raise ValueError("Model must be trained before fine-tuning")

    grid_search = GridSearchCV(model.model, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)

    model.model = grid_search.best_estimator_
    model.hyperparameters.update(grid_search.best_params_)

    return grid_search.best_estimator_