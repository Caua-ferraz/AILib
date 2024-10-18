from typing import Any, Dict
from sklearn.model_selection import RandomizedSearchCV
from .core import AIModel
from .error_handling import AILibError
import optuna
from sklearn.model_selection import cross_val_score

def random_search_optimization(
    model: AIModel,
    X: Any,
    y: Any,
    param_distributions: Dict[str, Any],
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42
) -> AIModel:
    try:
        search = RandomizedSearchCV(
            estimator=model.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            scoring='accuracy'
        )
        search.fit(X, y)
        model.model = search.best_estimator_
        model.hyperparameters.update(search.best_params_)
        return model
    except Exception as e:
        raise AILibError(f"Random Search Optimization failed: {e}")

def optuna_optimization(
    model: AIModel,
    X: Any,
    y: Any,
    param_space: Dict[str, Any],
    cv: int = 5,
    n_trials: int = 100,
    direction: str = 'maximize'
) -> AIModel:
    try:
        def objective(trial):
            params = {key: trial.suggest_categorical(key, values) if isinstance(values, list) 
                      else trial.suggest_float(key, *values) 
                      for key, values in param_space.items()}
            model.model.set_params(**params)
            score = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy').mean()
            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        model.model.set_params(**best_params)
        model.hyperparameters.update(best_params)
        return model
    except Exception as e:
        raise AILibError(f"Optuna Optimization failed: {e}")
