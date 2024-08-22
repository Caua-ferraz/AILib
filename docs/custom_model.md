# Working with Custom Models

AILib now supports working with custom models and training models from scratch. This guide will show you how to use these new features.

## Custom Traditional ML Models

You can use any scikit-learn compatible model with AILib. Here's an example:

```python
from ailib import UnifiedModel
from sklearn.ensemble import RandomForestClassifier

# Create a custom model
custom_model = RandomForestClassifier(n_estimators=100)

# Use the custom model with AILib
model = UnifiedModel('custom', custom_model=custom_model)

# Train and use the model as usual
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

## Training LLMs from Scratch

You can now train LLMs from scratch using your own dataset. Here's how:

```python
from ailib import UnifiedModel

# Initialize an LLM
llm_model = UnifiedModel('llm', model_name='gpt2')

# Prepare your training data
train_texts = [
    "This is an example sentence for training.",
    "Here's another sentence for the model to learn from.",
    # ... more training sentences ...
]

# Train the model
llm_model.train_llm(train_texts, num_epochs=3, batch_size=4, learning_rate=2e-5)

# Generate text with the trained model
generated_text = llm_model.predict("The trained model can now")
print(generated_text)
```

## Saving and Loading Custom Models

You can save and load custom models just like pre-defined models:

```python
# Save the model
model.save("path/to/save/model")

# Load the model
loaded_model = UnifiedModel.load('custom', "path/to/save/model")
```

Remember to handle larger models and datasets appropriately, considering memory constraints and processing time.