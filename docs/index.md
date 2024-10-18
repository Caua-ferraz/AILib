# AILib Documentation

Welcome to **AILib**, a versatile Python library that unifies traditional machine learning and large language models under a single, easy-to-use interface. AILib provides comprehensive tools for model management, training, evaluation, and deployment, ensuring users have full control over their AI workflows while simplifying complex processes.

## Key Features

- **Unified Interface** for both traditional ML models and LLMs
- **Support for Various Models**: Neural Networks, Decision Trees, Custom Models, and Large Language Models like GPT-2
- **Advanced Training Options**: Fine-tuning, Hyperparameter Optimization with Grid Search, Random Search, and Optuna
- **Data Processing Utilities**: Preprocessing, Splitting, Feature Selection, and Data Augmentation
- **Model Evaluation Tools**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Cross-Validation
- **Explainability Tools**: SHAP and LIME Integration for Model Interpretability
- **Pipeline Support** for End-to-End Workflows
- **Configuration Management** using Structured Configurations
- **Robust Error Handling** with Custom Exceptions
- **Logging Integration** for Monitoring and Debugging
- **Model Serving Capabilities** with FastAPI
- **Automated Documentation** and **Unit Testing** for Reliability
- **Command-Line Interface (CLI)** for Easy Interaction

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
- [Advanced Usage](tutorials/advanced_usage.md)
- [API Reference](api_reference/unified_model.md)
- [Configuration](docs/configuration.md)
- [Error Handling](docs/error_handling.md)
- [Logging](docs/logging.md)
- [Contributing](contributing.md)
- [Changelog](changelog.md)

## License

AILib is released under the [MIT License](https://github.com/Caua-ferraz/ailib/blob/main/LICENSE). See the [LICENSE](https://github.com/Caua-ferraz/ailib/blob/main/LICENSE) file for more details.
