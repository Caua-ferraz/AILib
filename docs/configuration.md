# Configuration Management

**AILib** utilizes structured configurations to manage settings efficiently. This ensures that your AI workflows are flexible, reproducible, and easy to maintain. This section covers how to load, modify, and save configurations using AILib's `Config` class.

## Overview

The `Config` class in AILib is built using Python's `dataclasses`, providing a type-safe and organized way to handle various configuration settings, including model parameters, training options, and logging preferences.

## Loading Configuration

You can load a configuration from a JSON file using the `load_config` class method.

```python
from ailib import Config

# Load configuration from a JSON file
config = Config.load_config('config.json')
```

### Example `config.json`

```json
{
    "project_name": "AILib Project",
    "version": "0.3.0",
    "default_model_type": "neural_network",
    "hyperparameters": {
        "neural_network": {
            "hidden_layer_sizes": [10, 5],
            "activation": "relu",
            "solver": "adam",
            "max_iter": 1000
        },
        "decision_tree": {
            "max_depth": null
        }
    },
    "training": {
        "num_epochs": 60,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "gradient_accumulation_steps": 4,
        "fp16": false
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## Accessing Configuration Settings

Once loaded, you can access various configuration settings as attributes of the `Config` instance.

```python
# Access training settings
num_epochs = config.training['num_epochs']
batch_size = config.training['batch_size']

# Access logging settings
log_level = config.logging['level']
log_format = config.logging['format']
```

## Modifying Configuration

You can modify configuration settings programmatically and save the updated configuration back to a JSON file.

```python
# Update training settings
config.training['num_epochs'] = 100
config.training['learning_rate'] = 3e-5

# Save the updated configuration
config.save_config('updated_config.json')
```

## Creating a Configuration from Scratch

If you prefer to create a configuration without loading from a file, you can instantiate the `Config` class directly.

```python
from ailib import Config

# Create a new configuration
config = Config(
    project_name="My AI Project",
    version="0.1.0",
    default_model_type="decision_tree",
    hyperparameters={
        "decision_tree": {
            "max_depth": 10
        }
    },
    training={
        "num_epochs": 50,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "gradient_accumulation_steps": 2,
        "fp16": true
    },
    logging={
        "level": "DEBUG",
        "format": "%(levelname)s:%(name)s:%(message)s"
    }
)

# Save the configuration
config.save_config('my_config.json')
```

## Error Handling

Loading or saving configurations may raise `AILibError` if issues occur (e.g., file not found, invalid JSON). Ensure to handle these exceptions appropriately.

```python
from ailib import Config, AILibError

try:
    config = Config.load_config('config.json')
except AILibError as e:
    print(f"Configuration Error: {e}")
```

## Best Practices

- **Version Control**: Keep your configuration files under version control to track changes over time.
- **Environment-Specific Configurations**: Use different configuration files for different environments (e.g., development, testing, production).
- **Validation**: Ensure that your configuration files contain all necessary fields and valid values to prevent runtime errors.

## Next Steps

- Learn how to integrate configurations with AILib's training and evaluation workflows.
- Explore [Logging](logging.md) to monitor your AI workflows effectively.
