# D:\AiProject\examples\train_llm_custom_dataset.py

from ailib import UnifiedModel
from datasets import load_dataset

def main():
    # Load a custom dataset (e.g., wikitext)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Extract text from the dataset
    train_texts = dataset["text"]
    
    # Initialize the UnifiedModel with an LLM
    model = UnifiedModel('llm', model_name='gpt2')
    
    # Train the model
    model.train_llm(
        train_texts=train_texts,
        num_epochs=60,
        batch_size=4,  # Adjust based on your GPU memory
        learning_rate=2e-5
    )
    
    # Generate some text with the fine-tuned model
    generated_text = model.predict("Artificial Intelligence is")
    print(generated_text)

    # Save the model
    model.save("./fine_tuned_model")

if __name__ == "__main__":
    main()