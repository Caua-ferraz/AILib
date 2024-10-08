# AILib

AILib is a versatile Python library that unifies traditional machine learning and large language models under a single, easy-to-use interface.

## Features

- Unified interface for both traditional ML models and LLMs
- Support for neural networks, decision trees, and custom models
- Integration with popular LLMs like GPT-2
- Advanced LLM training with custom datasets
- Optimized training for specific GPU models
- Experiment tracking with Weights & Biases
- Data preprocessing and model evaluation utilities
- Easy-to-use API for training, prediction, and model management

## Installation

```
pip install ailib
```

## Quick Start

```python
from ailib import UnifiedModel, preprocess_data, split_data

# Traditional ML
X, y = load_your_data()
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)

model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
model.train(X_train, y_train)
predictions = model.predict(X_test)

# Language Model
llm_model = UnifiedModel('llm', model_name='gpt2')
llm_model.train_llm(train_texts, num_epochs=60, batch_size=4, learning_rate=2e-5)
generated_text = llm_model.predict("Artificial Intelligence is")
print(generated_text)
```

## Advanced Usage

AILib now supports advanced LLM training with custom datasets and GPU optimizations:

```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = dataset["text"]

model = UnifiedModel('llm', model_name='gpt2')
model.train_llm(
    train_texts=train_texts,
    num_epochs=60,
    batch_size=4,
    learning_rate=2e-5,
    gradient_accumulation_steps=4,
    fp16=True,
    use_wandb=True
)
```

## Documentation

For more detailed information, check out our [User Guide](docs/user_guide.md) and [Developer Guide](docs/developer_guide.md).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
