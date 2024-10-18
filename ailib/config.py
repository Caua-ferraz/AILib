from dataclasses import dataclass, field
from typing import Dict, Any
import json
from .error_handling import AILibError


@dataclass
class Config:
    # General Settings
    project_name: str = "AILib Project"
    version: str = "0.3.0"

    # Model Settings
    default_model_type: str = "neural_network"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "neural_network": {"max_iter": 1000},
        "decision_tree": {"max_depth": None}
    })

    # Training Settings
    training: Dict[str, Any] = field(default_factory=lambda: {
        "num_epochs": 60,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "gradient_accumulation_steps": 4,
        "fp16": False
    })

    # Logging Settings
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    })

    @classmethod
    def load_config(cls, config_path: str) -> 'Config':
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except Exception as e:
            raise AILibError(f"Loading configuration failed: {e}")

    def save_config(self, config_path: str):
        try:
            with open(config_path, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        except Exception as e:
            raise AILibError(f"Saving configuration failed: {e}")
