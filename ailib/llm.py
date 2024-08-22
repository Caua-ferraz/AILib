# D:\AiProject\ailib\llm.py

import os
import json
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, pipeline


class LLM:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_text(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1, 
                      temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0) -> List[str]:
        outputs = self.generator(prompt, 
                                 max_length=max_length, 
                                 num_return_sequences=num_return_sequences,
                                 temperature=temperature,
                                 top_k=top_k,
                                 top_p=top_p)
        return [output['generated_text'] for output in outputs]

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def fine_tune(self, file_path: str, train_labels: Optional[List[str]] = None, 
                  num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The training file {file_path} does not exist.")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"The training file {file_path} is empty.")

        # Prepare dataset
        train_dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=file_path,
            block_size=128,
        )

        if len(train_dataset) == 0:
            raise ValueError(f"The training dataset is empty. File content: {content[:100]}...")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=10_000,
            save_total_limit=2,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        # Train the model
        trainer.train()

        # Update the model and tokenizer
        self.model = trainer.model
        self.tokenizer.save_pretrained("./results")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, 'model_name.json'), 'w') as f:
            json.dump({'model_name': self.model_name}, f)

    @classmethod
    def load_model(cls, path: str) -> 'LLM':
        with open(os.path.join(path, 'model_name.json'), 'r') as f:
            model_name = json.load(f)['model_name']
        
        instance = cls(model_name)
        instance.model = AutoModelForCausalLM.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Ensure padding token is set after loading
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
            instance.model.config.pad_token_id = instance.tokenizer.eos_token_id
        
        instance.generator = pipeline("text-generation", model=instance.model, tokenizer=instance.tokenizer)
        return instance