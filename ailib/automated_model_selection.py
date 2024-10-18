from typing import List, Any, Dict
from sklearn.model_selection import cross_val_score
from .core import AIModel
from .unified_model import UnifiedModel
from .error_handling import AILibError

def automated_model_selection(
    models: List[UnifiedModel],
    X: Any,
    y: Any,
    scoring: str = 'accuracy',
    cv: int = 5
) -> UnifiedModel:
    try:
        best_score = -float('inf')
        best_model = None
        for model in models:
            scores = cross_val_score(model.model.model if isinstance(model.model, AIModel) else model.model.model, X, y, cv=cv, scoring=scoring)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        if best_model is None:
            raise AILibError("No model was selected during automated model selection.")
        best_model.train(X, y)
        return best_model
    except Exception as e:
        raise AILibError(f"Automated Model Selection failed: {e}")
