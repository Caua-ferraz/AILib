# D:\AiProject\AILib-library\examples\custom_huggingface_training.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from ailib import UnifiedModel

def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a custom dataset
    try:
        dataset = load_dataset("fineweb")
        train_dataset = dataset["train"]
    except Exception as e:
        print(f"Error loading IMDB dataset: {e}")
        print("Falling back to a smaller dataset for testing purposes.")
        dataset = load_dataset("rotten_tomatoes")
        train_dataset = dataset["train"]

    # Initialize tokenizer and model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the UnifiedModel
    model = UnifiedModel('llm', model_name=model_name)

    # Set up custom training arguments
    custom_training_args = {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "logging_steps": 100,
        "save_steps": 1000,
        "eval_steps": 1000,
        "evaluation_strategy": "steps",
        "fp16": True,
        "gradient_accumulation_steps": 8,
    }

    # Fine-tune the model
    model.train_llm(
        train_texts=train_dataset["text"],
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        custom_training_args=custom_training_args
    )

    # Save the fine-tuned model
    model.save("./fine_tuned_model")

    # Test the model
    test_text = "This movie was fantastic! I really enjoyed every moment of it."
    try:
        generated_text = model.predict(test_text)
        print(f"Generated text: {generated_text}")
    except Exception as e:
        print(f"Error generating text: {e}")

if __name__ == "__main__":
    main()