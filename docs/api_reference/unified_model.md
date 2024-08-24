# UnifiedModel

The `UnifiedModel` class provides a unified interface for working with both traditional machine learning models and large language models.

## Class Definition

```python
class UnifiedModel:
    def __init__(self, model_type: str, **kwargs)
```

### Parameters:
- `model_type` (str): Type of the model. Options: 'neural_network', 'decision_tree', 'custom', 'llm'.
- `**kwargs`: Additional arguments specific to the chosen model type.

## Methods

### train

```python
def train(self, X: Union[np.ndarray, List[str]], y: Union[np.ndarray, List[str]] = None)
```

Train the model on the given data.

### train_llm

```python
def train_llm(self, train_texts: List[str], train_labels: List[str] = None, **kwargs)
```

Fine-tune a large language model.

### predict

```python
def predict(self, X: Union[np.ndarray, str, List[str]], **kwargs) -> Union[np.ndarray, List[str]]
```

Make predictions using the trained model.

### evaluate

```python
def evaluate(self, X: Union[np.ndarray, List[str]], y: Union[np.ndarray, List[str]] = None) -> Dict[str, float]
```

Evaluate the model's performance.

### save

```python
def save(self, path: str)
```

Save the model to the specified path.

### load (class method)

```python
@classmethod
def load(cls, model_type: str, path: str) -> 'UnifiedModel'
```

Load a model from the specified path.

### tokenize

```python
def tokenize(self, text: str) -> List[str]
```

Tokenize the input text (for LLM models).

### get_token_ids

```python
def get_token_ids(self, text: str) -> List[int]
```

Get token IDs for the input text (for LLM models).

## Examples

```python
# Traditional ML
model = UnifiedModel('neural_network', hidden_layer_sizes=(10, 5))
model.train(X_train, y_train)
predictions = model.predict(X_test)

# LLM
llm_model = UnifiedModel('llm', model_name='gpt2')
generated_text = llm_model.predict("AI is")
```

For more detailed examples and usage, refer to the tutorials section.