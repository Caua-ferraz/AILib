# AILib

AILib is a versatile Python library that unifies traditional machine learning and large language models under a single, easy-to-use interface.

## Features

- Unified interface for both traditional ML models and LLMs
- Support for neural networks, decision trees, and custom models
- Integration with popular LLMs like GPT-2
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
generated_text = llm_model.predict("Artificial Intelligence is")
print(generated_text)
```

## Documentation

For more detailed information, check out our [User Guide](docs/user_guide.md) and [Developer Guide](docs/developer_guide.md).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the scikit-learn and Hugging Face teams for their excellent libraries.
- Special thanks to all our contributors and users.