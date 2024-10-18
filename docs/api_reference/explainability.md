# API Reference: Explainability Tools

Understanding model predictions is vital for trust and transparency. **AILib** integrates model explainability tools such as **SHAP** and **LIME** to help users interpret and visualize how models make decisions. This section details the available methods and their usage.

## SHAP Integration

### `explain_shap(model: AIModel, X: np.ndarray) -> Any`

Generate SHAP values for the given model and dataset.

- **Parameters:**
  - `model` (AIModel): The trained machine learning model.
  - `X` (np.ndarray): Feature matrix for which to compute SHAP values.

- **Returns:**
  - SHAP values object.

- **Example:**

  ```python
  from ailib import explain_shap

  # Generate SHAP values
  shap_values = explain_shap(model, X_test)

  # Visualize SHAP summary
  import shap
  shap.summary_plot(shap_values, X_test)
  ```

## LIME Integration

### `explain_lime(model: AIModel, instance: Any, X: Any, y: Any) -> Any`

Generate a LIME explanation for a single prediction.

- **Parameters:**
  - `model` (AIModel): The trained machine learning model.
  - `instance` (Any): The single data instance to explain.
  - `X` (Any): Feature matrix used during training.
  - `y` (Any): Labels corresponding to the feature matrix.

- **Returns:**
  - LIME explanation object.

- **Example:**

  ```python
  from ailib import explain_lime

  # Select an instance to explain
  instance = X_test[0]

  # Generate LIME explanation
  explanation = explain_lime(model, instance, X_test, y_test)

  # Visualize the explanation
  explanation.show_in_notebook()
  ```

## Visualization Methods

Both SHAP and LIME offer various visualization tools to interpret model predictions effectively.

### SHAP Visualization

```python
import shap

# Summary plot
shap.summary_plot(shap_values, X_test)

# Dependence plot for a specific feature
shap.dependence_plot("feature_name", shap_values, X_test)
```

### LIME Visualization

```python
# Show explanation in Jupyter Notebook
explanation.show_in_notebook()

# Save explanation to an HTML file
explanation.save_to_file('lime_explanation.html')
```

## Best Practices

- **Global vs. Local Interpretability**:
  - Use **SHAP** for understanding global model behavior and feature importance.
  - Use **LIME** for explaining individual predictions in detail.

- **Consistent Interpretation**: Apply the same explainability tool across similar models to maintain consistency in interpretation.

- **Performance Considerations**: Generating explanations can be computationally intensive. Optimize by limiting the number of explanations or using efficient data sampling.

## Next Steps

- Integrate explainability tools within [Pipeline Integration](pipeline.md) for automated model interpretation.
- Explore advanced visualization techniques to create comprehensive interpretability reports.
