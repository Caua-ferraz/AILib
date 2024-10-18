# D:\AiProject\ailib\utils.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.utils import resample
from .core import AIModel
from .error_handling import AILibError


def visualize_data(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'pca',
    n_components: int = 2
) -> None:
    """
    Visualize the data using PCA or t-SNE.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        method (str, optional): Visualization method ('pca' or 'tsne').
        n_components (int, optional): Number of dimensions for reduction.
    """
    try:
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise AILibError(f"Unsupported visualization method: {method}")

        X_reduced = reducer.fit_transform(X)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=y,
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter)
        plt.title(f'Data Visualization using {method.upper()}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()
    except Exception as e:
        raise AILibError(f"Data visualization failed: {e}")


def learning_curve_plot(
    model: AIModel,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot the learning curve of a model.

    Args:
        model (AIModel): The AI model to evaluate.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        train_sizes (List[float], optional): Relative or absolute numbers of training examples.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Train sizes, train scores mean, test scores mean.
    """
    from sklearn.model_selection import learning_curve

    try:
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)

        train_sizes, train_scores, test_scores = learning_curve(
            model.model,
            X,
            y,
            train_sizes=train_sizes,
            cv=5,
            n_jobs=-1,
            shuffle=True,
            random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Score")
        plt.grid(True)

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r"
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g"
        )
        plt.plot(
            train_sizes,
            train_scores_mean,
            'o-',
            color="r",
            label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            'o-',
            color="g",
            label="Cross-validation score"
        )
        plt.legend(loc="best")
        plt.show()

        return train_sizes, train_scores_mean, test_scores_mean
    except Exception as e:
        raise AILibError(f"Learning curve plotting failed: {e}")


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    score_func = chi2
) -> Tuple[np.ndarray, Any]:
    """
    Select top k features based on the scoring function.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        k (int, optional): Number of top features to select.
        score_func: Scoring function to use for feature selection.

    Returns:
        Tuple[np.ndarray, Any]: Transformed feature matrix and the selector object.
    """
    try:
        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new, selector
    except Exception as e:
        raise AILibError(f"Feature selection failed: {e}")


def oversample_data(X: np.ndarray, y: np.ndarray, strategy: str = 'minority') -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class to balance the dataset.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        strategy (str, optional): Strategy to identify the minority class ('minority' by default).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled feature matrix and labels.
    """
    try:
        if strategy != 'minority':
            raise AILibError(f"Unsupported oversampling strategy: {strategy}")

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise AILibError("Oversampling currently supports binary classification only.")

        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        n_majority = counts.max()
        n_minority = counts.min()

        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]

        X_oversampled, y_oversampled = resample(
            X_minority,
            y_minority,
            replace=True,
            n_samples=n_majority - n_minority,
            random_state=42
        )

        X_resampled = np.vstack((X, X_oversampled))
        y_resampled = np.hstack((y, y_oversampled))

        return X_resampled, y_resampled
    except Exception as e:
        raise AILibError(f"Oversampling data failed: {e}")


def undersample_data(X: np.ndarray, y: np.ndarray, strategy: str = 'majority') -> Tuple[np.ndarray, np.ndarray]:
    """
    Undersample the majority class to balance the dataset.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        strategy (str, optional): Strategy to identify the majority class ('majority' by default).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled feature matrix and labels.
    """
    try:
        if strategy != 'majority':
            raise AILibError(f"Unsupported undersampling strategy: {strategy}")

        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise AILibError("Undersampling currently supports binary classification only.")

        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        n_minority = counts.min()

        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]

        X_undersampled, y_undersampled = resample(
            X_majority,
            y_majority,
            replace=False,
            n_samples=n_minority,
            random_state=42
        )

        X_resampled = np.vstack((X[y == minority_class], X_undersampled))
        y_resampled = np.hstack((y[y == minority_class], y_undersampled))

        return X_resampled, y_resampled
    except Exception as e:
        raise AILibError(f"Undersampling data failed: {e}")
