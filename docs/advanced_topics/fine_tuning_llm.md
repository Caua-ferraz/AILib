# Advanced Fine-tuning of Large Language Models

This guide covers advanced techniques for fine-tuning Large Language Models (LLMs) using AILib.

## Preparing Custom Datasets

For effective fine-tuning, prepare your dataset carefully:

```python
from datasets import Dataset

def prepare_dataset(texts, labels=None):
    dataset_dict = {"text": texts}
    if labels:
        dataset_dict["labels"] = labels
    return Dataset.from_dict(dataset_dict)

train_dataset = prepare_dataset(train_texts, train_labels)
```

## Custom Training Arguments

Customize the training process with advanced arguments:

```python
custom_training_args = {
    "output_dir": "./results",
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 1000,
    "fp16": True,
}

model.train_llm(train_dataset, custom_training_args=custom_training_args)
```

## Gradient Accumulation and Mixed Precision

For training on limited GPU memory:

```python
custom_training_args = {
    "gradient_accumulation_steps": 4,
    "fp16": True,
    "max_grad_norm": 1.0,
}
```

## Learning Rate Scheduling

Implement learning rate decay:

```python
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=1000
)

custom_training_args = {
    "learning_rate": 2e-5,
    "lr_scheduler_type": "linear",
    "warmup_steps": 100,
}
```

## Evaluation During Training

Set up evaluation during the training process:

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

custom_training_args = {
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "metric_for_best_model": "accuracy",
    "load_best_model_at_end": True,
}

model.train_llm(train_dataset, eval_dataset=eval_dataset, 
                compute_metrics=compute_metrics,
                custom_training_args=custom_training_args)
```

These advanced techniques allow for more control and potentially better results when fine-tuning LLMs with AILib.