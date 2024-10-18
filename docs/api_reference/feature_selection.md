# API Reference: Feature Selection Tools

Selecting the most relevant features is essential for building efficient and effective machine learning models. **AILib** provides feature selection utilities to help users preprocess their data by selecting the best features based on statistical tests.

## Feature Selection Methods

AILib offers two primary feature selection methods:

1. **SelectKBest**: Selects the top `k` features based on a scoring function.
2. **SelectFromModel**: Selects features based on the importance weights from a trained model.

## SelectKBest

### Function: `select_k_best(X: Any, y: Any, k: int, score_func: Callable) -> Tuple[Any, Any]`

Select the top `k` features based on a scoring function.

- **Parameters:**
  - `X` (Any): Feature matrix.
  - `y` (Any): Labels.
  - `k` (int): Number of top features to select.
  - `score_func` (Callable): Scoring function (e.g., `chi2`, `f_classif`).

- **Returns:**
  - `Tuple[Any, Any]`: Transformed feature matrix and the selector object.

- **Example:**

  ```python
  from ailib import select_k_best
  from sklearn.feature_selection import chi2

  # Select top 5 features based on chi-squared test
  X_new, selector = select_k_best(X, y, k=5, score_func=chi2)
  ```

## SelectFromModel

### Function: `select_from_model(model: BaseEstimator, X: Any, y: Any, threshold: float = 0.05) -> Tuple[Any, Any]`

Select features based on feature importance from a trained model.

- **Parameters:**
  - `model` (BaseEstimator): Trained model with feature_importances_ or coef_ attribute.
  - `X` (Any): Feature matrix.
  - `y` (Any): Labels.
  - `threshold` (float, optional): Threshold for feature importance to select features.

- **Returns:**
  - `Tuple[Any, Any]`: Transformed feature matrix and the selector object.

- **Example:**

  ```python
  from ailib import select_from_model
  from sklearn.ensemble import RandomForestClassifier

  # Train a model to get feature importances
  clf = RandomForestClassifier()
  clf.fit(X, y)

  # Select features with importance above 0.05
  X_new, selector = select_from_model(clf, X, y, threshold=0.05)
  ```

## Best Practices

- **Understanding Feature Importance**: Choose the appropriate feature selection method based on your model and data characteristics.
- **Avoid Overfitting**: Feature selection helps in reducing model complexity and prevents overfitting by eliminating irrelevant features.
- **Data Leakage**: Perform feature selection within cross-validation folds to prevent data leakage and ensure unbiased evaluation.
- **Evaluation**: Always evaluate the impact of feature selection on model performance to ensure that essential features are not discarded.

## Integrating Feature Selection with Pipelines

Combine feature selection with preprocessing and modeling steps within an AILib pipeline for seamless workflows.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from ailib import AILibPipeline, AIModel

# Initialize model
model = AIModel('neural_network', hidden_layer_sizes=(20, 10))

# Define preprocessors including feature selection
preprocessors = [
    StandardScaler(),
    SelectKBest(score_func=chi2, k=5)
]

# Create the pipeline
pipeline = AILibPipeline(model=model, preprocessors=preprocessors)

# Train the pipeline
pipeline.train(X_train, y_train)

# Evaluate the pipeline
evaluation_results = pipeline.evaluate(X_test, y_test)
print("Pipeline Evaluation:", evaluation_results)
```

## Next Steps

- Explore [Data Augmentation Tools](data_augmentation.md) to enhance your dataset.
- Learn how to integrate [Explainability Tools](explainability.md) with feature-selected models.
- Review [Advanced Usage](advanced_usage.md) for comprehensive workflows.
