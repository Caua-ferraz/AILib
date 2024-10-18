import unittest
from ailib import UnifiedModel

class TestGPT2Training(unittest.TestCase):
    def setUp(self):
        # Initialize the GPT-2 model
        self.llm_model = UnifiedModel('llm', model_name='gpt2')
        
        # Sample training texts
        self.train_texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning enables computers to learn from data.",
            "Natural language processing allows for human-like text generation.",
            "Deep learning models require large amounts of data.",
            "GPT-2 is a powerful language model developed by OpenAI."
        ]
    
    def test_train_gpt2_model(self):
        try:
            # Train the GPT-2 model
            self.llm_model.train_llm(
                train_texts=self.train_texts,
                num_epochs=1,          # Using 1 epoch for testing purposes
                batch_size=2,
                learning_rate=3e-5
            )
        except Exception as e:
            self.fail(f"Training GPT-2 model failed with exception: {e}")
    
    def tearDown(self):
        # Optionally, clean up resources or save the model after tests
        pass

if __name__ == '__main__':
    unittest.main()
