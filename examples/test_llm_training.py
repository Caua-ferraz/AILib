# D:\AiProject\AILib-library\tests\test_llm_training.py

import unittest
import warnings
import os
from ailib import UnifiedModel

class TestLLMTraining(unittest.TestCase):
    def setUp(self):
        self.train_texts = [
            "Artificial Intelligence is revolutionizing various industries.",
            "Machine learning models require large amounts of data for training.",
            "Deep learning has shown remarkable results in image and speech recognition.",
            "Natural Language Processing is a key component of modern AI systems.",
            "The future of AI looks promising with advancements in neural networks."
        ]
        warnings.filterwarnings("ignore", message="Using pad_token, but it is not set yet.")
        warnings.filterwarnings("ignore", message="You have modified the pretrained model configuration to control generation.")

    def test_llm_initialization(self):
        model = UnifiedModel('llm', model_name='gpt2')
        self.assertIsNotNone(model.model)

    def test_llm_training(self):
        model = UnifiedModel('llm', model_name='gpt2')
        model.train_llm(self.train_texts, num_epochs=1, batch_size=2)
        
        generated_text = model.predict("AI and machine learning")
        self.assertIsInstance(generated_text, list)
        self.assertGreater(len(generated_text[0]), 0)

    def test_llm_save_load(self):
        model = UnifiedModel('llm', model_name='gpt2')
        model.train_llm(self.train_texts, num_epochs=1, batch_size=2)
        
        # Save the model
        model.save("test_llm_model")
        
        # Load the model
        loaded_model = UnifiedModel.load('llm', "test_llm_model")
        
        # Check if the loaded model can generate text
        generated_text = loaded_model.predict("AI and machine learning")
        self.assertIsInstance(generated_text, list)
        self.assertGreater(len(generated_text[0]), 0)

        # Clean up
        if os.path.exists("test_llm_model"):
            import shutil
            shutil.rmtree("test_llm_model")

if __name__ == '__main__':
    unittest.main()