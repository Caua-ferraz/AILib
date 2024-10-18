class AILibError(Exception):
    """Base exception for AILib."""
    pass

class ModelNotTrainedError(AILibError):
    """Exception raised when attempting to use a model that hasn't been trained."""
    def __init__(self, message="The model has not been trained yet."):
        self.message = message
        super().__init__(self.message)

class UnsupportedModelTypeError(AILibError):
    """Exception raised for unsupported model types."""
    def __init__(self, model_type: str):
        self.message = f"Unsupported model type: {model_type}"
        super().__init__(self.message)

class InvalidConfigurationError(AILibError):
    """Exception raised for invalid configurations."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
