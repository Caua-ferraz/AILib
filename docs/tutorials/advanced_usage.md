# Advanced Usage Tutorial

This tutorial covers advanced operations in **AILib**, demonstrating how to leverage additional features such as hyperparameter optimization, model explainability, and pipeline integration. It also showcases configuration management, logging, and error handling to enhance your AI workflows.

## Traditional Machine Learning

### Preparing the Data

```python
import numpy as np
from ailib import UnifiedModel, preprocess_data, split_data

# Generate sample data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Preprocess and split the data
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed, test_size=0.2)
```

### Creating and Training a Model

```python
# Create a neural network model with specific hyperparameters
model = UnifiedModel('neural_network', hidden_layer_sizes=(20, 10), activation='relu')

# Train the model
model.train(X_train, y_train)
```

### Making Predictions and Evaluating the Model

```python
# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
evaluation_results = model.evaluate(X_test, y_test)
print("Evaluation Results:", evaluation_results)
```

### Hyperparameter Optimization

AILib supports advanced hyperparameter optimization techniques using Grid Search, Random Search, and Optuna.

```python
from ailib import fine_tune_model

# Define the parameter grid for Grid Search
param_grid = {
    'hidden_layer_sizes': [(20, 10), (30, 15)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd']
}

# Fine-tune the model
fine_tuned_model = fine_tune_model(model, X_train, y_train, param_grid, cv=5)
print("Best Hyperparameters:", fine_tuned_model.hyperparameters)
```

## Large Language Models

### Initializing an LLM

```python
llm_model = UnifiedModel('llm', model_name='gpt2')
```

### Generating Text

```python
prompt = "The future of artificial intelligence is"
generated_text = llm_model.predict(prompt, max_length=100)
print("Generated Text:", generated_text)
```

### Fine-tuning an LLM

```python
train_texts = [
    "AI is transforming healthcare by enabling predictive analytics.",
    "Machine learning and deep learning are subsets of artificial intelligence.",
    "Natural language processing allows machines to understand human language.",
    # Add more training examples...
]

llm_model.train_llm(train_texts, num_epochs=5, batch_size=8, learning_rate=3e-5)
```

### Using the Fine-tuned Model

```python
fine_tuned_text = llm_model.predict("In the realm of AI, the next breakthrough")
print("Fine-tuned Model Output:", fine_tuned_text)
```

## Model Explainability

Understanding model decisions is crucial. AILib integrates SHAP and LIME for model interpretability.

### SHAP Integration

```python
from ailib import explain_shap

# Explain model predictions using SHAP
shap_values = explain_shap(model, X_test)

# Visualize SHAP values
import shap
shap.summary_plot(shap_values, X_test)
```

### LIME Integration

```python
from ailib import explain_lime

# Explain a single prediction using LIME
instance = X_test[0]
explanation = explain_lime(model, instance, X_test, y_test)
explanation.show_in_notebook()
```

## Pipeline Integration

Create end-to-end workflows by chaining preprocessing and modeling steps.

```python
from ailib import AILibPipeline
from sklearn.preprocessing import StandardScaler

# Define preprocessors
preprocessors = [
    StandardScaler()
]

# Initialize the pipeline with preprocessors and the model
pipeline = AILibPipeline(model=model, preprocessors=preprocessors)

# Train the pipeline
pipeline.train(X_train, y_train)

# Make predictions
pipeline_predictions = pipeline.predict(X_test)

# Evaluate the pipeline
pipeline_evaluation = pipeline.evaluate(X_test, y_test)
print("Pipeline Evaluation:", pipeline_evaluation)
```

## Configuration Management

Manage your settings efficiently using structured configurations.

```python
from ailib import Config

# Load configuration from a JSON file
config = Config.load_config('path_to_config.json')

# Access configuration settings
print(config.training['num_epochs'])

# Save updated configuration
config.save_config('path_to_updated_config.json')
```

## Logging and Error Handling

Monitor and debug your workflows effectively with AILib's integrated logging and custom error handling.

```python
from ailib import Config, setup_logging
from ailib import AIModel, AILibError

# Load configuration and set up logging
config = Config.load_config('config.json')
setup_logging(config)

try:
    # Initialize and train model
    model = AIModel('neural_network')
    model.train(X_train, y_train)
except AILibError as e:
    print(f"AILib Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
```

## Saving and Loading Models

Ensure model persistence with AILib's save and load functionalities.

```python
# Save the trained models
model.save("saved_traditional_ml_model.pkl")
llm_model.save("saved_fine_tuned_llm_model.pkl")

# Load the models
loaded_ml_model = UnifiedModel.load('neural_network', "saved_traditional_ml_model.pkl")
loaded_llm_model = UnifiedModel.load('llm', "saved_fine_tuned_llm_model.pkl")
```

## Next Steps

- Dive deeper into other [Tutorials](tutorials/basic_usage.md) for more detailed examples.
- Check the [API Reference](api_reference/unified_model.md) for a complete list of available methods and classes.
- Learn about [Advanced Topics](tutorials/advanced_usage.md) for more complex use cases.
- Contribute to the project or report issues on our [GitHub repository](https://github.com/Caua-ferraz/ailib).
