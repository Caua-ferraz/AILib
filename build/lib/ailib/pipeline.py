from typing import Any, List
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .core import AIModel
from .error_handling import AILibError

class AILibPipeline:
    def __init__(self, model: AIModel, preprocessors: List[Any] = None):
        self.preprocessors = preprocessors or []
        steps = []
        for idx, processor in enumerate(self.preprocessors):
            steps.append((f"preprocessor_{idx}", processor))
        steps.append(("model", model.model))
        self.pipeline = Pipeline(steps)

    def train(self, X: Any, y: Any):
        try:
            self.pipeline.fit(X, y)
        except Exception as e:
            raise AILibError(f"Pipeline training failed: {e}")

    def predict(self, X: Any) -> Any:
        try:
            return self.pipeline.predict(X)
        except Exception as e:
            raise AILibError(f"Pipeline prediction failed: {e}")

    def save(self, path: str):
        try:
            joblib.dump(self.pipeline, path)
        except Exception as e:
            raise AILibError(f"Saving pipeline failed: {e}")

    @classmethod
    def load(cls, path: str) -> 'AILibPipeline':
        try:
            pipeline = joblib.load(path)
            return cls(model=AIModel('custom', custom_model=pipeline.named_steps['model']))
        except Exception as e:
            raise AILibError(f"Loading pipeline failed: {e}")
