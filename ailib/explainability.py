import shap
from typing import Any
from .core import AIModel
from .error_handling import AILibError

def explain_model_predictions(model: AIModel, X: Any, feature_names: list = None) -> Any:
    try:
        explainer = shap.Explainer(model.model)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names)
        return shap_values
    except Exception as e:
        raise AILibError(f"SHAP Explainability failed: {e}")
