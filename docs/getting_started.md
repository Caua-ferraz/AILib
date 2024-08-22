# Getting Started with AILib

This guide will help you get up and running with AILib, covering installation, basic usage, and common workflows.

## Installation

1. Ensure you have Python 3.6 or later installed.
2. Install AILib using pip:

```
pip install ailib
```

## Basic Usage

### Traditional Machine Learning

```python
from ailib import UnifiedModel, preprocess_data, split_data
from sklearn.datasets import load_iris

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

# Create and train a model
model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
evaluation_results = model.evaluate(X_test, y_test)
print(evaluation_results)
```

### Large Language Models

```python
from ailib import UnifiedModel

# Initialize an LLM
llm_model = UnifiedModel('llm', model_name='gpt2')

# Generate text
prompt = "Artificial Intelligence is"
generated_text = llm_model.predict(prompt, max_length=100, temperature=0.7)
print(generated_text)

# Tokenize text
text = "Hello, how are you?"
tokens = llm_model.tokenize(text)
print(tokens)
```

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed information on classes and methods.
- Check out the [Examples](examples.md) for more advanced usage scenarios.
- If you encounter any issues, please refer to our troubleshooting guide or open an issue on our GitHub repository.