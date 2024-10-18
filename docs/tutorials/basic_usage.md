# Basic Usage Tutorial

This tutorial covers the fundamental operations in **AILib**, demonstrating how to use both traditional machine learning models and large language models. It also introduces configuration management, logging, and error handling to enhance your AI workflows.

## Traditional Machine Learning

### Preparing the Data

```python
import numpy as np
from ailib import UnifiedModel, preprocess_data, split_data

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
print("Evaluation Results:", evaluation_results)
```

## Large Language Models

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

### Using the Fine-tuned Model

```python
fine_tuned_text = llm_model.predict("In the next decade, AI will")
print("Fine-tuned Model Output:", fine_tuned_text)
```

## Configuration Management

Manage your settings efficiently using structured configurations.

```python
from ailib import Config

# Load configuration from a JSON file
config = Config.load_config('config.json')

# Access configuration settings
print(config.training['num_epochs'])

# Save updated configuration
config.save_config('updated_config.json')
```

## Logging and Error Handling

AILib integrates a robust logging system and custom error handling to streamline debugging and monitoring.

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

## Next Steps

- Explore the [Advanced Usage Tutorial](advanced_usage.md) for more sophisticated operations.
- Check out the [API Reference](api_reference/unified_model.md) for a complete list of available methods and classes.
- Learn about [Model Explainability](api_reference/explainability.md) to interpret model decisions.
