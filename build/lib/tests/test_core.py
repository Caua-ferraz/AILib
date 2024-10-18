# D:\AiProject\tests\test_core.py

import unittest
import numpy as np
from ailib.core import AIModel

class TestAIModel(unittest.TestCase):
    def setUp(self):
        self.model = AIModel('neural_network', {'hidden_layer_sizes': (100,)})

    def test_initialization(self):
        self.assertEqual(self.model.model_type, 'neural_network')
        expected_hyperparameters = {
            'hidden_layer_sizes': (100,),
            'max_iter': 1000  # Added this line to account for the default max_iter
        }
        self.assertEqual(self.model.hyperparameters, expected_hyperparameters)
        self.assertIsNotNone(self.model.model)

    def test_unsupported_model_type(self):
        with self.assertRaises(ValueError):
            AIModel('unsupported_type')

    def test_custom_max_iter(self):
        model = AIModel('neural_network', {'hidden_layer_sizes': (100,), 'max_iter': 500})
        self.assertEqual(model.hyperparameters['max_iter'], 500)

    def test_decision_tree_no_max_iter(self):
        model = AIModel('decision_tree', {'max_depth': 5})
        self.assertNotIn('max_iter', model.hyperparameters)

if __name__ == '__main__':
    unittest.main()