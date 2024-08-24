# Working with Custom Models

This tutorial demonstrates how to integrate custom models into AILib, allowing you to leverage AILib's unified interface with your own model implementations.

## Creating a Custom Model

To use a custom model with AILib, your model should be compatible with the scikit-learn API. Here's an example of a simple custom model:

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class SimpleCustomModel(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def fit(self, X, y):
        # Simple logic: just remember the mean of each feature for each class
        self.class_means = {
            cls: np.mean(X[y == cls], axis=0)
            for cls in np.unique(y)
        }
        return self
    
    def predict(self, X):
        predictions = []
        for sample in X:
            distances = {
                cls: np.linalg.norm(sample - mean)
                for cls, mean in self.class_means.items()
            }
            predictions.append(min(distances, key=distances.get))
        return np.array(predictions)

```

## Integrating the Custom Model with AILib

Now that we have our custom model, let's use it with AILib:

```python
from ailib import UnifiedModel

# Create an instance of your custom model
custom_model_instance = SimpleCustomModel(threshold=0.6)

# Create a UnifiedModel with your custom model
unified_custom_model = UnifiedModel('custom', custom_model=custom_model_instance)

# Use the model as you would any other AILib model
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data

unified_custom_model.train(X_train, y_train)
predictions = unified_custom_model.predict(X_test)
evaluation_results = unified_custom_model.evaluate(X_test, y_test)
print("Custom Model Evaluation:", evaluation_results)
```

## Advanced: Custom Model with Hyperparameter Tuning

You can also use AILib's unified interface with custom models that support hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'threshold': [0.3, 0.5, 0.7]
}

base_model = SimpleCustomModel()
grid_search = GridSearchCV(base_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_custom_model = UnifiedModel('custom', custom_model=grid_search.best_estimator_)
```

## Saving and Loading Custom Models

AILib supports saving and loading custom models:

```python
# Save the custom model
unified_custom_model.save("./custom_model_save_path")

# Load the custom model
loaded_custom_model = UnifiedModel.load('custom', "./custom_model_save_path")
```

By following this tutorial, you can integrate any custom model into AILib, as long as it follows the scikit-learn estimator interface. This allows you to use AILib's unified interface and utilities with your own specialized models.