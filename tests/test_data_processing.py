# D:\AiProject\tests\test_data_processing.py

import unittest
import numpy as np
from ailib.data_processing import preprocess_data, split_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([0, 1, 0, 1])

    def test_preprocess_data(self):
        X_scaled, y = preprocess_data(self.X, self.y)
        self.assertEqual(X_scaled.shape, self.X.shape)
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-7))
        self.assertTrue(np.allclose(X_scaled.std(axis=0), [1, 1], atol=1e-7))

    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, test_size=0.5)
        self.assertEqual(X_train.shape[0], 2)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(y_train.shape[0], 2)
        self.assertEqual(y_test.shape[0], 2)

if __name__ == '__main__':
    unittest.main()