# Examples

This document provides various examples of how to use AILib for different tasks.

## Traditional Machine Learning

### Neural Network for Classification

```python
from ailib import UnifiedModel, preprocess_data, split_data
from sklearn.datasets import load_iris

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

# Create and train a neural network
model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5), max_iter=1000)
model.train(X_train, y_train)

# Evaluate the model
evaluation_results = model.evaluate(X_test, y_test)
print("Model Evaluation Results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")
```

## Large Language Models

### Text Generation with Different Parameters

```python
from ailib import UnifiedModel

# Initialize an LLM
llm_model = UnifiedModel('llm', model_name='gpt2')

prompt = "Artificial Intelligence is"

# Generate text with default parameters
generated_text = llm_model.predict(prompt, max_length=100)
print("Default generation:")
print(generated_text[0])

# Generate more creative text
creative_text = llm_model.predict(prompt, max_length=100, temperature=1.5, top_k=50)
print("\nMore creative generation:")
print(creative_text[0])

# Generate more focused text
focused_text = llm_model.predict(prompt, max_length=100, temperature=0.3, top_p=0.9)
print("\nMore focused generation:")
print(focused_text[0])
```

### Using Different LLM Models

```python
from ailib import UnifiedModel

prompt = "The future of technology is"

# Using GPT-2
gpt2_model = UnifiedModel('llm', model_name='gpt2')
gpt2_text = gpt2_model.predict(prompt, max_length=100)
print("GPT-2 generation:")
print(gpt2_text[0])

# Using GPT-2 Medium
gpt2_medium_model = UnifiedModel('llm', model_name='gpt2-medium')
gpt2_medium_text = gpt2_medium_model.predict(prompt, max_length=100)
print("\nGPT-2 Medium generation:")
print(gpt2_medium_text[0])
```

For more examples and advanced usage, please refer to the `examples` directory in the AILib repository.