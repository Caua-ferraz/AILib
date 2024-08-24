# Model Optimization Techniques

This guide covers advanced techniques for optimizing models in AILib, applicable to both traditional ML models and LLMs.

## Hyperparameter Tuning

### For Traditional ML Models

```python
from sklearn.model_selection import GridSearchCV
from ailib import UnifiedModel

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

base_model = UnifiedModel('neural_network')
grid_search = GridSearchCV(base_model.model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = UnifiedModel('custom', custom_model=grid_search.best_estimator_)
```

### For LLMs

```python
from ailib import UnifiedModel

hyperparameters = [
    {"learning_rate": 1e-5, "num_train_epochs": 3},
    {"learning_rate": 2e-5, "num_train_epochs": 4},
    {"learning_rate": 5e-5, "num_train_epochs": 2}
]

best_model = None
best_performance = float('inf')

for params in hyperparameters:
    model = UnifiedModel('llm', model_name='gpt2')
    model.train_llm(train_texts, **params)
    performance = model.evaluate(eval_texts, eval_labels)
    
    if performance['loss'] < best_performance:
        best_model = model
        best_performance = performance['loss']
```

## Model Pruning

For traditional ML models, especially neural networks:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_model_optimization as tfmot

def prune_model(model, target_sparsity=0.5):
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=target_sparsity,
            begin_step=0, end_step=1000
        )
    }
    
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model_for_pruning

pruned_model = prune_model(original_model)
```

## Quantization

For LLMs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Quantize the model to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Use the quantized model
input_ids = tokenizer.encode("AI is", return_tensors="pt")
output = quantized_model.generate(input_ids)
```

## Model Distillation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, outputs, labels, teacher_outputs):
        ce_loss = self.ce_loss(outputs, labels)
        kl_loss = self.kl_loss(
            nn.functional.log_softmax(outputs / self.temperature, dim=1),
            nn.functional.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        return self.alpha * ce_loss + (1 - self.alpha) * kl_loss

def distill_model(teacher_model, student_model, train_loader, num_epochs=10):
    optimizer = optim.Adam(student_model.parameters())
    distillation_loss = DistillationLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            inputs, labels = batch
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)
            
            loss = distillation_loss(student_outputs, labels, teacher_outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return student_model

distilled_model = distill_model(teacher_model, student_model, train_loader)
```

These advanced optimization techniques can help improve the performance and efficiency of your models in AILib.