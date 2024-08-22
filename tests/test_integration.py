# D:\AiProject\tests\test_integration.py

import unittest
import numpy as np
from sklearn.datasets import make_classification
from ailib import UnifiedModel, preprocess_data, split_data

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        self.X_processed, self.y_processed = preprocess_data(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.X_processed, self.y_processed)

    def test_end_to_end_neural_network(self):
        model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
        model.train(self.X_train, self.y_train)
        evaluation_results = model.evaluate(self.X_test, self.y_test)
        self.assertIn('accuracy', evaluation_results)
        self.assertGreater(evaluation_results['accuracy'], 0.5)

    def test_end_to_end_custom_model(self):
        from sklearn.ensemble import RandomForestClassifier
        custom_model = RandomForestClassifier(n_estimators=100)
        model = UnifiedModel('custom', custom_model=custom_model)
        model.train(self.X_train, self.y_train)
        evaluation_results = model.evaluate(self.X_test, self.y_test)
        self.assertIn('accuracy', evaluation_results)
        self.assertGreater(evaluation_results['accuracy'], 0.5)

    def test_end_to_end_llm(self):
        model = UnifiedModel('llm', model_name='gpt2')
        train_texts = [
            "Artificial Intelligence is revolutionizing various industries.",
            "Machine learning models require large amounts of data for training.",
            "Deep learning has shown remarkable results in image and speech recognition.",
            "Natural Language Processing is a key component of modern AI systems.",
            "The future of AI looks promising with advancements in neural networks."
        ]
        model.train_llm(train_texts, num_epochs=1, batch_size=2)
        
        generated_text = model.predict("AI and machine learning")
        self.assertIsInstance(generated_text, list)
        self.assertGreater(len(generated_text[0]), 0)

if __name__ == '__main__':
    unittest.main()