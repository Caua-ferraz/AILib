# LLM

The `LLM` class handles operations related to Large Language Models in AILib.

## Class Definition

```python
class LLM:
    def __init__(self, model_name: str = "gpt2")
```

### Parameters:
- `model_name` (str): Name of the pre-trained model to use. Default is "gpt2".

## Methods

### generate_text

```python
def generate_text(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1, 
                  temperature: float = 1.0, top_k: int = 50, top_p: float = 1.0) -> List[str]
```

Generate text based on the given prompt.

### tokenize

```python
def tokenize(self, text: str) -> List[str]
```

Tokenize the input text.

### get_token_ids

```python
def get_token_ids(self, text: str) -> List[int]
```

Get token IDs for the input text.

### fine_tune

```python
def fine_tune(self, train_texts: List[str], train_labels: Optional[List[str]] = None, 
              num_epochs: int = 60, batch_size: int = 4, learning_rate: float = 2e-5,
              custom_training_args: Optional[Dict[str, Any]] = None)
```

Fine-tune the language model on custom data.

### save

```python
def save(self, path: str)
```

Save the model to the specified path.

### load_model (class method)

```python
@classmethod
def load_model(cls, path: str) -> 'LLM'
```

Load a model from the specified path.

## Examples

```python
llm = LLM(model_name="gpt2-medium")
generated_text = llm.generate_text("The future of AI is", max_length=100)
print(generated_text)

# Fine-tuning
train_texts = ["AI is transforming industries", "Machine learning requires data"]
llm.fine_tune(train_texts, num_epochs=3, batch_size=2)
```