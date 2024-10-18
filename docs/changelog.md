# Changelog

All notable changes to **AILib** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Configuration Management**: Introduced `config.py` with structured configurations using dataclasses. Added methods to load and save configurations.
- **Error Handling**: Implemented custom exceptions in `error_handling.py` for clearer error messages and better exception management.
- **Logging Integration**: Added `logging_setup.py` to configure logging based on user-defined settings, replacing `print` statements.
- **Hyperparameter Optimization**: Integrated advanced hyperparameter tuning methods including Random Search and Optuna in `hyperparameter_optimization.py`.
- **Model Explainability**: Added SHAP and LIME integration in `explainability.py` for model interpretability.
- **Pipeline Support**: Introduced `pipeline.py` allowing users to create end-to-end workflows combining preprocessing and modeling steps.
- **Model Serving Tools**: Added `model_serving.py` to deploy models as APIs using FastAPI.
- **Hashing Utility**: Included `hashmap.py` to generate unique hashes for configuration dictionaries, aiding in version control.
- **Unit Testing**: Set up a testing framework using `pytest` with example test cases in the `tests/` directory.
- **Documentation Enhancements**: Expanded the documentation to include new features, API references, tutorials, and contributing guidelines.
- **Command-Line Interface (CLI)**: Developed `cli.py` to allow users to interact with AILib via the command line for training and evaluation tasks.

### Changed

- **Refactored Codebase**: Optimized various modules (`core.py`, `evaluation.py`, `model_training.py`, etc.) for better performance, readability, and error handling.
- **Improved Documentation**: Updated existing documentation files to reflect new features and changes, ensuring comprehensive coverage of all functionalities.

### Fixed

- **Bug Fixes**: Addressed issues related to model saving/loading, device handling for LLMs, and inconsistent method naming across modules.

---

## [0.3.1] - YYYY-MM-DD

### Added

- Example implementations and additional utility functions in `utils.py` including feature selection and data augmentation tools.

### Fixed

- Corrected typos in documentation files and ensured consistency in function naming across the library.

---

## [0.3.0] - YYYY-MM-DD

### Added

- Initial release of AILib with support for traditional ML models and LLMs, comprehensive training, evaluation, and deployment tools.

## License

AILib is released under the [MIT License](https://github.com/Caua-ferraz/ailib/blob/main/LICENSE).
