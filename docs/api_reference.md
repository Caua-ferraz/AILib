# API Reference

This document provides detailed information about the classes and functions in AILib.

## UnifiedModel

The main class for working with both traditional ML models and LLMs.

### Methods

#### `__init__(model_type: str, **kwargs)`

Initialize a UnifiedModel.

- `model_type`: Type of model ('neural_network', 'decision_tree', or 'llm')
- `**kwargs`: Additional arguments for the specific model type

#### `train(X, y)`

Train the model on the given data.

#### `predict(X, **kwargs)`

Make predictions or generate text, depending on the model type.

#### `evaluate(X, y)`

Evaluate the model's performance (for traditional ML models).

#### `tokenize(text)`

Tokenize the given text (for LLM models).

## Utility Functions

### `preprocess_data(X, y)`

Preprocess the input data and labels.

### `split_data(X, y, test_size=0.2, random_state=42)`

Split the data into training and test sets.

## LLM

The class for working with Large Language Models.

### Methods

#### `__init__(model_name: str = "gpt2")`

Initialize an LLM with the specified model.

#### `generate_text(prompt: str, max_length: int = 50, num_return_sequences: int = 1, temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0)`

Generate text based on the given prompt and parameters.

#### `tokenize(text: str)`

Tokenize the given text.

For more detailed information on each class and method, including parameters and return types, please refer to the inline documentation in the source code.