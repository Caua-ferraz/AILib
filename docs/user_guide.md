# User Guide for AILib

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Working with Traditional ML Models](#working-with-traditional-ml-models)
4. [Working with Language Models](#working-with-language-models)
5. [The UnifiedModel Interface](#the-unifiedmodel-interface)
6. [Data Processing](#data-processing)
7. [Model Evaluation](#model-evaluation)
8. [Troubleshooting](#troubleshooting)

## Installation

Install AILib using pip:

```
pip install ailib
```

## Quick Start

Here's a simple example to get you started:

```python
from ailib import UnifiedModel, preprocess_data, split_data

# For traditional ML
X, y = load_your_data()
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
model.train(X_train, y_train)
predictions = model.predict(X_test)

# For LLM
llm_model = UnifiedModel('llm', model_name='gpt2')
generated_text = llm_model.predict("Artificial Intelligence is")
print(generated_text)
```

## Working with Traditional ML Models

AILib supports various traditional ML models:

```python
# Neural Network
nn_model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))

# Decision Tree
dt_model = UnifiedModel('decision_tree', max_depth=5)

# Custom sklearn-compatible model
from sklearn.ensemble import RandomForestClassifier
rf_model = UnifiedModel('custom', custom_model=RandomForestClassifier())
```

## Working with Language Models

AILib provides an interface to work with pre-trained language models:

```python
llm_model = UnifiedModel('llm', model_name='gpt2')

# Generate text
generated_text = llm_model.predict("The future of AI is")

# Fine-tune the model
llm_model.train_llm(train_texts, num_epochs=3, batch_size=4)
```

## The UnifiedModel Interface

The UnifiedModel class provides a consistent interface for all model types:

- `train()`: Train the model
- `predict()`: Make predictions
- `evaluate()`: Evaluate the model's performance
- `save()`: Save the model
- `load()`: Load a saved model

## Data Processing

AILib includes utilities for data preprocessing:

```python
from ailib import preprocess_data, split_data

X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)
```

## Model Evaluation

Evaluate your models using the `evaluate()` method:

```python
evaluation_results = model.evaluate(X_test, y_test)
print(evaluation_results)
```
## Advanced LLM Training

AILib now supports more advanced training options for Language Models, including the use of custom datasets and optimized training for specific GPU models.

### Training with Custom Datasets

You can now use custom datasets from Hugging Face or your own data for training:

```python
from ailib import UnifiedModel
from datasets import load_dataset

# Load a custom dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = dataset["text"]

# Initialize and train the model
model = UnifiedModel('llm', model_name='gpt2')
model.train_llm(
    train_texts=train_texts,
    num_epochs=60,
    batch_size=4,
    learning_rate=2e-5
)
```

### Optimized Training for Specific GPUs

The `train_llm` method now includes optimizations for specific GPU models. For example, for an RTX 3060:

```python
model.train_llm(
    train_texts=train_texts,
    num_epochs=60,
    batch_size=4,
    learning_rate=2e-5,
    gradient_accumulation_steps=4,
    fp16=True
)
```

These settings are optimized for the RTX 3060's 12GB of VRAM. Adjust `batch_size` and `gradient_accumulation_steps` based on your specific GPU.

## Experiment Tracking with Weights & Biases

AILib now supports experiment tracking with Weights & Biases (wandb). To use this feature:

1. Install wandb: `pip install wandb`
2. Initialize wandb in your script:

```python
import wandb
wandb.init(project="your-project-name")
```

3. Enable wandb logging in `train_llm`:

```python
model.train_llm(
    train_texts=train_texts,
    num_epochs=60,
    batch_size=4,
    learning_rate=2e-5,
    use_wandb=True
)
```

To disable wandb logging (e.g., for testing), set `use_wandb=False` or set the environment variable `WANDB_DISABLED=true`.

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**: If you encounter GPU out-of-memory errors, try reducing the `batch_size` or increasing `gradient_accumulation_steps`.

2. **Convergence Warnings**: For neural network models, you may see ConvergenceWarnings. Try increasing `max_iter` or adjusting the model architecture.

3. **Wandb Initialization Errors**: Ensure you have initialized wandb correctly in your script. If you don't want to use wandb, set `use_wandb=False` in `train_llm`.

For any other issues, please refer to our GitHub issues page or contact support.


## Troubleshooting

If you encounter ConvergenceWarnings with neural networks, try:
- Increasing the `max_iter` parameter
- Using a simpler model architecture
- Preprocessing your data differently

For any other issues, please refer to our GitHub issues page or contact support.

