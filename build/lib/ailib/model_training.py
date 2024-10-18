# D:\AiProject\ailib\model_training.py

import numpy as np
from typing import Any, Dict
from sklearn.base import BaseEstimator
from .core import AIModel
from .error_handling import AILibError


def train_model(model: AIModel, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    try:
        model.train(X, y)
        return model.model
    except Exception as e:
        raise AILibError(f"Training the model failed: {e}")


def fine_tune_model(
    model: AIModel,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, Any],
    cv: int = 5
) -> BaseEstimator:
    from sklearn.model_selection import GridSearchCV

    if model.model is None:
        raise AILibError("Model must be trained before fine-tuning.")

    try:
        grid_search = GridSearchCV(model.model, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X, y)

        model.model = grid_search.best_estimator_
        model.hyperparameters.update(grid_search.best_params_)

        return grid_search.best_estimator_
    except Exception as e:
        raise AILibError(f"Fine-tuning the model failed: {e}")
