# D:\AiProject\tests\test_model_training.py

import unittest
import numpy as np
from ailib.core import AIModel
from ailib.model_training import train_model, fine_tune

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        self.model = AIModel('neural_network', {'hidden_layer_sizes': (10,), 'max_iter': 1000})

    def test_train_model(self):
        trained_model = train_model(self.model, self.X, self.y)
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(self.model.model)

    def test_fine_tune(self):
        train_model(self.model, self.X, self.y)
        param_grid = {'hidden_layer_sizes': [(5,), (10,), (15,)]}
        fine_tuned_model = fine_tune(self.model, self.X, self.y, param_grid, cv=2)  # Changed cv to 2
        self.assertIsNotNone(fine_tuned_model)
        self.assertIn(self.model.hyperparameters['hidden_layer_sizes'], [(5,), (10,), (15,)])

if __name__ == '__main__':
    unittest.main()