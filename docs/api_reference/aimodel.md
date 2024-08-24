# AIModel

The `AIModel` class is the base class for traditional machine learning models in AILib.

## Class Definition

```python
class AIModel:
    def __init__(self, model_type: str, hyperparameters: dict = None, custom_model: BaseEstimator = None)
```

### Parameters:
- `model_type` (str): Type of the model. Options: 'neural_network', 'decision_tree', 'custom'.
- `hyperparameters` (dict, optional): Hyperparameters for the model.
- `custom_model` (BaseEstimator, optional): A custom sklearn-compatible model.

## Methods

### train

```python
def train(self, X: np.ndarray, y: np.ndarray)
```

Train the model on the given data.

### predict

```python
def predict(self, X: np.ndarray) -> np.ndarray
```

Make predictions using the trained model.

### save

```python
def save(self, path: str)
```

Save the model to the specified path.

### load (class method)

```python
@classmethod
def load(cls, path: str) -> 'AIModel'
```

Load a model from the specified path.

## Examples

```python
# Neural Network
model = AIModel('neural_network', hyperparameters={'hidden_layer_sizes': (10, 5)})
model.train(X_train, y_train)
predictions = model.predict(X_test)

# Custom Model
from sklearn.ensemble import RandomForestClassifier
custom_rf = RandomForestClassifier(n_estimators=100)
model = AIModel('custom', custom_model=custom_rf)
model.train(X_train, y_train)
```