# AILib Documentation

Welcome to AILib, a versatile Python library that unifies traditional machine learning and large language models under a single, easy-to-use interface.

## Key Features

- Unified interface for both traditional ML models and LLMs
- Support for neural networks, decision trees, and custom models
- Integration with popular LLMs like GPT-2
- Advanced LLM training with custom datasets
- Optimized training for specific GPU models
- Data preprocessing and model evaluation utilities

## Quick Start

```python
from ailib import UnifiedModel, preprocess_data, split_data

# Traditional ML
model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
model.train(X_train, y_train)
predictions = model.predict(X_test)

# Language Model
llm_model = UnifiedModel('llm', model_name='gpt2')
generated_text = llm_model.predict("Artificial Intelligence is")
```

## Documentation Contents

- [Installation](installation.md)
- [Getting Started](getting_started.md)
- [Tutorials](tutorials/basic_usage.md)
- [API Reference](api_reference/unified_model.md)
- [Advanced Topics](advanced_topics/fine_tuning_llms.md)
- [Contributing](contributing.md)
- [Changelog](changelog.md)

## License

AILib is released under the MIT License. See the [LICENSE](https://github.com/Caua-ferraz/ailib/blob/main/LICENSE) file for more details.