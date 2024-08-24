# Working with Large Language Models (LLMs)

This tutorial provides an in-depth look at working with Large Language Models in AILib. We'll cover initialization, text generation, fine-tuning, and advanced usage.

## Initializing an LLM

```python
from ailib import UnifiedModel

# Initialize with a pre-trained model (e.g., GPT-2)
llm_model = UnifiedModel('llm', model_name='gpt2')
```

## Text Generation

### Basic Text Generation

```python
prompt = "The future of artificial intelligence is"
generated_text = llm_model.predict(prompt, max_length=100)
print(generated_text)
```

### Controlling Generation Parameters

```python
generated_text = llm_model.predict(
    prompt,
    max_length=200,
    num_return_sequences=3,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
print(generated_text)
```

## Fine-tuning LLMs

### Preparing Training Data

```python
train_texts = [
    "AI is revolutionizing various industries.",
    "Machine learning models require large amounts of data for training.",
    "Natural language processing has made significant progress in recent years.",
    # Add more training examples...
]
```

### Fine-tuning the Model

```python
llm_model.train_llm(
    train_texts,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5
)
```

### Using Custom Training Arguments

```python
custom_training_args = {
    "output_dir": "./custom_model_output",
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 1000,
    "warmup_steps": 200,
    "fp16": True
}

llm_model.train_llm(
    train_texts,
    num_epochs=5,
    batch_size=8,
    learning_rate=3e-5,
    custom_training_args=custom_training_args
)
```

## Advanced LLM Usage

### Tokenization

```python
text = "AILib makes working with LLMs easy!"
tokens = llm_model.tokenize(text)
print("Tokens:", tokens)

token_ids = llm_model.get_token_ids(text)
print("Token IDs:", token_ids)
```

### Saving and Loading Fine-tuned Models

```python
# Save the fine-tuned model
llm_model.save("./fine_tuned_llm")

# Load the fine-tuned model
loaded_llm = UnifiedModel.load('llm', "./fine_tuned_llm")
```

### Using Different Pre-trained Models

```python
gpt2_medium = UnifiedModel('llm', model_name='gpt2-medium')
bart_model = UnifiedModel('llm', model_name='facebook/bart-large')
```

This tutorial covers the main aspects of working with LLMs in AILib. For more advanced topics, such as using custom datasets or optimizing for specific hardware, check out our advanced tutorials.