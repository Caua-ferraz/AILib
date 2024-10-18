import pytest
import numpy as np
from ailib.core import AIModel
from ailib.error_handling import ModelNotTrainedError, UnsupportedModelTypeError


def test_ai_model_initialization():
    model = AIModel(model_type='neural_network')
    assert model.model_type == 'neural_network'
    assert model.hyperparameters['max_iter'] == 1000


def test_ai_model_training_and_prediction():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    model = AIModel(model_type='decision_tree')
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 2
    assert all(predictions == y)


def test_predict_without_training():
    X = np.array([[0, 0], [1, 1]])
    model = AIModel(model_type='decision_tree')
    with pytest.raises(ModelNotTrainedError):
        model.predict(X)


def test_unsupported_model_type():
    with pytest.raises(UnsupportedModelTypeError):
        AIModel(model_type='unsupported_model')
