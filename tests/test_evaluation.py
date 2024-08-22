# D:\AiProject\tests\test_evaluation.py

import unittest
import numpy as np
from ailib.core import AIModel
from ailib.model_training import train_model
from ailib.evaluation import evaluate_model, get_confusion_matrix, cross_validate

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])
        self.model = AIModel('neural_network', {'hidden_layer_sizes': (10,)})
        train_model(self.model, self.X, self.y)

    def test_evaluate_model(self):
        metrics = evaluate_model(self.model, self.X, self.y)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)

    def test_get_confusion_matrix(self):
        cm = get_confusion_matrix(self.model, self.X, self.y)
        self.assertEqual(cm.shape, (2, 2))

    def test_cross_validate(self):
        cv_results = cross_validate(self.model, self.X, self.y, cv=2)
        self.assertIn('accuracy', cv_results)
        self.assertIn('precision', cv_results)
        self.assertIn('recall', cv_results)
        self.assertIn('f1_score', cv_results)

if __name__ == '__main__':
    unittest.main()