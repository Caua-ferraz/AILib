# API Reference: AILibPipeline

The `AILibPipeline` class in **AILib** facilitates the creation of end-to-end workflows by chaining data preprocessing and modeling steps. This ensures streamlined and modular AI workflows, enhancing maintainability and scalability.

## Class Definition

```python
class AILibPipeline:
    def __init__(self, model: AIModel, preprocessors: List[Any] = None):
        ...
    
    def train(self, X: Any, y: Any):
        ...
    
    def predict(self, X: Any) -> Any:
        ...
    
    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        ...
    
    def save(self, path: str):
        ...
    
    @classmethod
    def load(cls, path: str) -> 'AILibPipeline':
        ...
```

## Methods

### `__init__(self, model: AIModel, preprocessors: List[Any] = None)`

Initialize the pipeline with optional preprocessors and the model.

- **Parameters:**
  - `model` (AIModel): The machine learning model to include in the pipeline.
  - `preprocessors` (List[Any], optional): A list of preprocessing objects (e.g., `StandardScaler`, `PCA`).

- **Example:**

  ```python
  from sklearn.preprocessing import StandardScaler
  from ailib import AIModel, AILibPipeline

  # Initialize model
  model = AIModel('neural_network', hidden_layer_sizes=(20, 10))

  # Define preprocessors
  preprocessors = [
      StandardScaler(),
      PCA(n_components=5)
  ]

  # Create the pipeline
  pipeline = AILibPipeline(model=model, preprocessors=preprocessors)
  ```

### `train(self, X: Any, y: Any)`

Train the pipeline with the provided data.

- **Parameters:**
  - `X` (Any): Feature matrix.
  - `y` (Any): Labels.

- **Example:**

  ```python
  # Train the pipeline
  pipeline.train(X_train, y_train)
  ```

### `predict(self, X: Any) -> Any`

Make predictions using the trained pipeline.

- **Parameters:**
  - `X` (Any): Input features for prediction.

- **Returns:**
  - Predictions (Any): Model predictions.

- **Example:**

  ```python
  # Make predictions
  pipeline_predictions = pipeline.predict(X_test)
  ```

### `evaluate(self, X: Any, y: Any) -> Dict[str, float]`

Evaluate the pipeline's performance using various metrics.

- **Parameters:**
  - `X` (Any): Test feature matrix.
  - `y` (Any): True labels.

- **Returns:**
  - `Dict[str, float]`: Dictionary containing evaluation metrics like accuracy, precision, recall, and F1 score.

- **Example:**

  ```python
  # Evaluate the pipeline
  pipeline_evaluation = pipeline.evaluate(X_test, y_test)
  print("Pipeline Evaluation:", pipeline_evaluation)
  ```

### `save(self, path: str)`

Save the trained pipeline to the specified path.

- **Parameters:**
  - `path` (str): File path to save the pipeline.

- **Example:**

  ```python
  pipeline.save("saved_pipeline.pkl")
  ```

### `load(cls, path: str) -> 'AILibPipeline'`

Load a saved pipeline from the specified path.

- **Parameters:**
  - `path` (str): File path from where to load the pipeline.

- **Returns:**
  - `AILibPipeline`: An instance of `AILibPipeline` with the loaded pipeline.

- **Example:**

  ```python
  loaded_pipeline = AILibPipeline.load("saved_pipeline.pkl")
  ```

## Best Practices

- **Modular Preprocessing**: Incorporate multiple preprocessors to handle various data preparation tasks systematically.
- **Reuse Components**: Reuse trained pipelines across different datasets to ensure consistency.
- **Save Pipelines**: Always save trained pipelines to avoid retraining and to ensure reproducibility.

## Next Steps

- Learn how to integrate pipelines with [Hyperparameter Optimization](advanced_usage.md#hyperparameter-optimization) for automated model tuning.
- Explore [Model Explainability](explainability.md) within pipelines to interpret model decisions.
