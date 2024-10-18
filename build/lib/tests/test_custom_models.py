# D:\AiProject\tests\test_custom_models.py

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ailib import UnifiedModel

class TestCustomModels(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])

    def test_custom_model_initialization(self):
        custom_model = RandomForestClassifier(n_estimators=100)
        model = UnifiedModel('custom', custom_model=custom_model)
        self.assertIsInstance(model.model.model, RandomForestClassifier)

    def test_custom_model_training(self):
        custom_model = RandomForestClassifier(n_estimators=100)
        model = UnifiedModel('custom', custom_model=custom_model)
        model.train(self.X, self.y)
        # Check if the model can make predictions after training
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_custom_model_prediction(self):
        custom_model = RandomForestClassifier(n_estimators=100)
        model = UnifiedModel('custom', custom_model=custom_model)
        model.train(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_custom_model_save_load(self):
        custom_model = RandomForestClassifier(n_estimators=100)
        model = UnifiedModel('custom', custom_model=custom_model)
        model.train(self.X, self.y)
        
        # Save the model
        model.save("test_custom_model.joblib")
        
        # Load the model
        loaded_model = UnifiedModel.load('custom', "test_custom_model.joblib")
        
        # Check if predictions are the same
        original_predictions = model.predict(self.X)
        loaded_predictions = loaded_model.predict(self.X)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

if __name__ == '__main__':
    unittest.main()