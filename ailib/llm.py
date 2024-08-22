# D:\AiProject\AILib-library\ailib\llm.py

import os
from typing import List, Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
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
        
        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
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

    def fine_tune(self, train_texts: List[str], train_labels: Optional[List[str]] = None, 
                  num_epochs: int = 60, batch_size: int = 4, learning_rate: float = 2e-5):
        # Create a HuggingFace Dataset
        dataset = Dataset.from_dict({"text": train_texts})
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False,
        )

        # Calculate optimal training steps
        total_samples = len(tokenized_dataset)
        steps_per_epoch = max(1, total_samples // batch_size)
        total_steps = steps_per_epoch * num_epochs

        # Optimize for RTX 3060 (12GB VRAM)
        gradient_accumulation_steps = 4  # Adjust this based on your specific needs
        fp16 = True  # Use mixed precision training

        # Ensure logging_steps is non-zero
        logging_steps = max(1, steps_per_epoch // 10)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            save_steps=steps_per_epoch,
            save_total_limit=2,
            logging_steps=logging_steps,
            max_steps=total_steps,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )

        # Train the model
        trainer.train()

        # Update the model and tokenizer
        self.model = trainer.model
        self.tokenizer.save_pretrained("./results")

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_model(cls, path: str) -> 'LLM':
        instance = cls()
        instance.model = AutoModelForCausalLM.from_pretrained(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Ensure padding token is set after loading
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
            instance.model.config.pad_token_id = instance.tokenizer.eos_token_id
        
        instance.generator = pipeline("text-generation", model=instance.model, tokenizer=instance.tokenizer)
        return instance