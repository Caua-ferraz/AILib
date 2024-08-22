# D:\AiProject\tests\test_unified_model.py

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ailib import UnifiedModel

class TestUnifiedModel(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])

    def test_neural_network(self):
        model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
        model.train(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_decision_tree(self):
        model = UnifiedModel('decision_tree', max_depth=5)
        model.train(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_custom_model(self):
        custom_model = RandomForestClassifier(n_estimators=100)
        model = UnifiedModel('custom', custom_model=custom_model)
        model.train(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def test_llm(self):
        model = UnifiedModel('llm', model_name='gpt2')
        generated_text = model.predict("AI is")
        self.assertIsInstance(generated_text, list)
        self.assertGreater(len(generated_text[0]), 0)

if __name__ == '__main__':
    unittest.main()