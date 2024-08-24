# Working with Traditional ML Models

This tutorial dives deeper into using traditional machine learning models with AILib. We'll cover different model types, hyperparameter tuning, and more advanced techniques.

## Supported Model Types

AILib supports various traditional ML models:

1. Neural Networks
2. Decision Trees
3. Custom models (sklearn-compatible)

## Neural Networks

### Creating a Neural Network

```python
from ailib import UnifiedModel

nn_model = UnifiedModel('neural_network', hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000)
```

### Training and Evaluation

```python
nn_model.train(X_train, y_train)
evaluation_results = nn_model.evaluate(X_test, y_test)
print("Neural Network Evaluation:", evaluation_results)
```

## Decision Trees

### Creating a Decision Tree

```python
dt_model = UnifiedModel('decision_tree', max_depth=10, min_samples_split=5)
```

### Training and Evaluation

```python
dt_model.train(X_train, y_train)
evaluation_results = dt_model.evaluate(X_test, y_test)
print("Decision Tree Evaluation:", evaluation_results)
```

## Custom Models

You can use any sklearn-compatible model with AILib:

```python
from sklearn.ensemble import RandomForestClassifier

custom_model = RandomForestClassifier(n_estimators=100)
unified_custom_model = UnifiedModel('custom', custom_model=custom_model)

unified_custom_model.train(X_train, y_train)
evaluation_results = unified_custom_model.evaluate(X_test, y_test)
print("Custom Model Evaluation:", evaluation_results)
```

## Hyperparameter Tuning

AILib doesn't provide built-in hyperparameter tuning, but you can use sklearn's GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'max_iter': [500, 1000]
}

base_model = UnifiedModel('neural_network')
grid_search = GridSearchCV(base_model.model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = UnifiedModel('custom', custom_model=grid_search.best_estimator_)
```

## Feature Importance

For models that support feature importance (like Decision Trees):

```python
dt_model = UnifiedModel('decision_tree', max_depth=10)
dt_model.train(X_train, y_train)

feature_importance = dt_model.model.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f"Feature {i}: {importance}")
```

This tutorial covers more advanced usage of traditional ML models in AILib. For working with large language models, check out our LLM tutorial.