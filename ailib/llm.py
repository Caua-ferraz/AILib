import os
from typing import List, Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from .error_handling import AILibError


class LLM:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1  # Transformers pipeline expects device index
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(f'cuda:{self.device}' if self.device >= 0 else 'cpu')
        
        self._set_padding_token()
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def _set_padding_token(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_text(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0
    ) -> List[str]:
        try:
            outputs = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            return [output['generated_text'] for output in outputs]
        except Exception as e:
            raise AILibError(f"Error in text generation: {e}")

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def fine_tune(
        self,
        train_texts: List[Any],
        train_labels: Optional[List[str]] = None,
        num_epochs: int = 60,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        custom_training_args: Optional[Dict[str, Any]] = None
    ):
        try:
            if isinstance(train_texts[0], list) and all(isinstance(item, int) for item in train_texts[0]):
                dataset = Dataset.from_dict({"input_ids": train_texts})
            else:
                dataset = Dataset.from_dict({"text": train_texts})
                dataset = dataset.map(self._tokenize_function, batched=True, remove_columns=dataset.column_names)

            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

            training_args_dict = {
                "output_dir": "./results",
                "overwrite_output_dir": True,
                "num_train_epochs": num_epochs,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": 4,
                "learning_rate": learning_rate,
                "fp16": torch.cuda.is_available(),
                "save_steps": max(1, len(dataset) // batch_size),
                "save_total_limit": 2,
                "logging_steps": max(1, (len(dataset) // batch_size) // 10),
                "max_steps": (max(1, len(dataset) // batch_size)) * num_epochs
            }

            if custom_training_args:
                training_args_dict.update(custom_training_args)

            training_args = TrainingArguments(**training_args_dict)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )

            trainer.train()

            self.model = trainer.model
            self.tokenizer.save_pretrained(training_args.output_dir)
        except Exception as e:
            raise AILibError(f"Fine-tuning the LLM failed: {e}")

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def save(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise AILibError(f"Saving LLM failed: {e}")

    @classmethod
    def load_model(cls, path: str) -> 'LLM':
        try:
            instance = cls()
            instance.model = AutoModelForCausalLM.from_pretrained(path).to(instance.model.device)
            instance.tokenizer = AutoTokenizer.from_pretrained(path)
            instance._set_padding_token()
            instance.generator = pipeline(
                "text-generation",
                model=instance.model,
                tokenizer=instance.tokenizer,
                device=instance.device
            )
            return instance
        except Exception as e:
            raise AILibError(f"Loading LLM failed: {e}")

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)
