# D:\AiProject\ailib\evaluation.py

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer
)
from .core import AIModel
from .error_handling import AILibError


def evaluate_model(model: AIModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if model.model is None:
        raise AILibError("Model has not been trained yet.")

    y_pred = model.predict(X)

    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted', zero_division=1),
        'recall': recall_score(y, y_pred, average='weighted', zero_division=1),
        'f1_score': f1_score(y, y_pred, average='weighted', zero_division=1)
    }


def get_confusion_matrix(model: AIModel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if model.model is None:
        raise AILibError("Model has not been trained yet.")

    y_pred = model.predict(X)
    return confusion_matrix(y, y_pred)


def cross_validate_model(
    model: AIModel,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, np.ndarray]:
    from sklearn.model_selection import cross_validate

    if model.model is None:
        raise AILibError("Model has not been trained yet.")

    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', zero_division=1),
        'recall': make_scorer(recall_score, average='weighted', zero_division=1),
        'f1_score': make_scorer(f1_score, average='weighted', zero_division=1)
    }

    scores = cross_validate(model.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        'accuracy': scores['test_accuracy'],
        'precision': scores['test_precision'],
        'recall': scores['test_recall'],
        'f1_score': scores['test_f1_score']
    }
