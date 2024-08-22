# D:\AiProject\ailib\utils.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .core import AIModel  # Add this import

def visualize_data(X: np.ndarray, y: np.ndarray, method: str = 'pca', n_components: int = 2) -> None:
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError(f"Unsupported visualization method: {method}")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Data Visualization using {method.upper()}')
    plt.show()

def learning_curve(model: AIModel, X: np.ndarray, y: np.ndarray, 
                   train_sizes: List[float] = np.linspace(0.1, 1.0, 5)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        model.model, X, y, train_sizes=train_sizes, cv=5, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

    return train_sizes, train_scores_mean, test_scores_mean