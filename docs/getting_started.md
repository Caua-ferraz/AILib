# Getting Started with AILib

This guide will help you get up and running with AILib, covering basic usage for both traditional machine learning models and large language models.

## Importing AILib

First, import the necessary components from AILib:

```python
from ailib import UnifiedModel, preprocess_data, split_data
```

## Working with Traditional ML Models

### Preparing Data

```python
import numpy as np

# Create sample data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Preprocess and split the data
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)
```

### Creating and Training a Model

```python
# Create a neural network model
model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))

# Train the model
model.train(X_train, y_train)
```

### Making Predictions

```python
predictions = model.predict(X_test)
```

### Evaluating the Model

```python
evaluation_results = model.evaluate(X_test, y_test)
print(evaluation_results)
```

## Working with Large Language Models

### Initializing an LLM

```python
llm_model = UnifiedModel('llm', model_name='gpt2')
```

### Generating Text

```python
prompt = "Artificial Intelligence is"
generated_text = llm_model.predict(prompt)
print(generated_text)
```

### Fine-tuning an LLM

```python
train_texts = [
    "AI is revolutionizing various industries.",
    "Machine learning models require large amounts of data.",
    # ... more training examples ...
]

llm_model.train_llm(train_texts, num_epochs=3, batch_size=4)
```

## Next Steps

- Explore the [Tutorials](tutorials/basic_usage.md) for more detailed examples.
- Check the [API Reference](api_reference/unified_model.md) for a complete list of available methods and classes.
- Learn about [Advanced Topics](advanced_topics/fine_tuning_llms.md) for more complex use cases.