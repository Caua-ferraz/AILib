# AILib Documentation

Welcome to the AILib documentation! AILib is a versatile Python library that unifies traditional machine learning and large language models under a single, easy-to-use interface.

## Features

- Unified interface for both traditional ML models and LLMs
- Support for neural networks and decision trees
- Integration with popular LLMs like GPT-2
- Customizable text generation parameters
- Data preprocessing and evaluation utilities

## Table of Contents

1. [Getting Started](getting_started.md)
2. [API Reference](api_reference.md)
3. [Examples](examples.md)

## Installation

To install AILib, run the following command:

```
pip install ailib
```

For the latest development version, you can install directly from the repository:

```
pip install git+https://github.com/yourusername/ailib.git
```

## Quick Start

Here's a quick example of how to use AILib:

```python
from ailib import UnifiedModel, preprocess_data, split_data

# For traditional ML
ml_model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
ml_model.train(X_train, y_train)
predictions = ml_model.predict(X_test)

# For LLM
llm_model = UnifiedModel('llm', model_name='gpt2')
generated_text = llm_model.predict("Artificial Intelligence is")

print(generated_text)
```

For more detailed information, check out our [Getting Started](getting_started.md) guide.