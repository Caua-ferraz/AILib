# D:\AiProject\examples\custom_huggingface_training.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from ailib import UnifiedModel

def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a custom dataset (replace with your own dataset)
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"]

    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

    # Initialize the UnifiedModel
    model = UnifiedModel('llm', model_name=model_name)

    # Set up training arguments
    # Adjust these values based on your RTX 3060's memory and performance
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=60,
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,  # Keep only the last 3 checkpoints
        fp16=True,  # Enable mixed precision training for RTX 3060
        gradient_accumulation_steps=4,  # Increase effective batch size
    )

    # Fine-tune the model
    model.train_llm(
        train_texts=tokenized_datasets["text"],
        train_labels=tokenized_datasets["label"],
        training_args=training_args,
        tokenizer=tokenizer,
    )

    # Save the fine-tuned model
    model.save("./fine_tuned_model")

    # Test the model
    test_text = "This movie was fantastic! I really enjoyed every moment of it."
    generated_text = model.predict(test_text)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()