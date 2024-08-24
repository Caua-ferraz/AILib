# Basic Usage Tutorial

This tutorial covers the fundamental operations in AILib, demonstrating how to use both traditional machine learning models and large language models.

## Traditional Machine Learning

### Preparing the Data

```python
import numpy as np
from ailib import UnifiedModel, preprocess_data, split_data

# Generate sample data
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# Preprocess and split the data
X_processed, y_processed = preprocess_data(X, y)
X_train, X_test, y_train, y_test = split_data(X_processed, y_processed, test_size=0.2)
```

### Creating and Training a Model

```python
# Create a neural network model
model = UnifiedModel('neural_network', hidden_layer_sizes=(20, 10))

# Train the model
model.train(X_train, y_train)
```

### Making Predictions and Evaluating the Model

```python
# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
evaluation_results = model.evaluate(X_test, y_test)
print("Evaluation Results:", evaluation_results)
```

## Large Language Models

### Initializing an LLM

```python
llm_model = UnifiedModel('llm', model_name='gpt2')
```

### Generating Text

```python
prompt = "The future of artificial intelligence is"
generated_text = llm_model.predict(prompt, max_length=100)
print("Generated Text:", generated_text)
```

### Fine-tuning an LLM

```python
train_texts = [
    "AI is transforming various industries.",
    "Machine learning models require large datasets for training.",
    "Natural language processing has made significant progress in recent years.",
    # Add more training examples...
]

llm_model.train_llm(train_texts, num_epochs=3, batch_size=4, learning_rate=2e-5)
```

### Using the Fine-tuned Model

```python
fine_tuned_text = llm_model.predict("In the next decade, AI will")
print("Fine-tuned Model Output:", fine_tuned_text)
```

## Saving and Loading Models

```python
# Save the models
model.save("traditional_ml_model")
llm_model.save("fine_tuned_llm_model")

# Load the models
loaded_ml_model = UnifiedModel.load('neural_network', "traditional_ml_model")
loaded_llm_model = UnifiedModel.load('llm', "fine_tuned_llm_model")
```

This tutorial covers the basics of using AILib. For more advanced usage, check out our other tutorials and the API reference.