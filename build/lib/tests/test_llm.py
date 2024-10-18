# D:\AiProject\tests\test_llm.py

import unittest
from ailib.llm import LLM

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm = LLM()

    def test_initialization(self):
        self.assertEqual(self.llm.model_name, "gpt2")
        self.assertIsNotNone(self.llm.model)
        self.assertIsNotNone(self.llm.tokenizer)
        self.assertIsNotNone(self.llm.generator)

    def test_generate_text(self):
        prompt = "Once upon a time"
        generated_text = self.llm.generate_text(prompt)
        self.assertIsInstance(generated_text, list)
        self.assertEqual(len(generated_text), 1)
        self.assertTrue(generated_text[0].startswith(prompt))

    def test_tokenize(self):
        text = "Hello, world!"
        tokens = self.llm.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)

    def test_get_token_ids(self):
        text = "Hello, world!"
        token_ids = self.llm.get_token_ids(text)
        self.assertIsInstance(token_ids, list)
        self.assertTrue(len(token_ids) > 0)

if __name__ == '__main__':
    unittest.main()