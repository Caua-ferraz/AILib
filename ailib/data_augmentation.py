import numpy as np
from typing import Any, Tuple
from sklearn.utils import resample
from .error_handling import AILibError

def oversample_data(X: np.ndarray, y: np.ndarray, strategy: str = 'minority') -> Tuple[np.ndarray, np.ndarray]:
    try:
        X_resampled, y_resampled = resample(
            X[y == 1],
            y[y == 1],
            replace=True,
            n_samples=X[y == 0].shape[0],
            random_state=42
        )
        X_augmented = np.vstack((X, X_resampled))
        y_augmented = np.hstack((y, y_resampled))
        return X_augmented, y_augmented
    except Exception as e:
        raise AILibError(f"Oversampling Data failed: {e}")

def undersample_data(X: np.ndarray, y: np.ndarray, strategy: str = 'majority') -> Tuple[np.ndarray, np.ndarray]:
    try:
        X_resampled, y_resampled = resample(
            X[y == 0],
            y[y == 0],
            replace=False,
            n_samples=X[y == 1].shape[0],
            random_state=42
        )
        X_augmented = np.vstack((X[y == 1], X_resampled))
        y_augmented = np.hstack((y[y == 1], y_resampled))
        return X_augmented, y_augmented
    except Exception as e:
        raise AILibError(f"Undersampling Data failed: {e}")
