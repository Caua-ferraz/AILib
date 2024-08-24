# Working with Custom Datasets

This guide covers advanced techniques for working with custom datasets in AILib, applicable to both traditional ML models and LLMs.

## Creating Custom Datasets

### For Traditional ML Models

```python
import numpy as np
from sklearn.model_selection import train_test_split
from ailib import preprocess_data

class CustomDataset:
    def __init__(self, X, y):
        self.X, self.y = preprocess_data(X, y)

    def split(self, test_size=0.2, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

# Usage
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
dataset = CustomDataset(X, y)
X_train, X_test, y_train, y_test = dataset.split()
```

### For LLMs

```python
from datasets import Dataset

class CustomLLMDataset:
    def __init__(self, texts, labels=None):
        self.dataset = Dataset.from_dict({"text": texts, "label": labels} if labels else {"text": texts})

    def preprocess(self, tokenizer, max_length=512):
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        
        return self.dataset.map(tokenize_function, batched=True)

# Usage
texts = ["AI is fascinating", "Machine learning is powerful", ...]
labels = [1, 0, ...]  # Optional
dataset = CustomLLMDataset(texts, labels)
```

## Data Augmentation

### For Traditional ML

```python
from sklearn.preprocessing import PolynomialFeatures

def augment_features(X, degree=2):
    poly = PolynomialFeatures(degree)
    return poly.fit_transform(X)

X_augmented = augment_features(X)
```

### For LLMs

```python
import nlpaug.augmenter.word as naw

def augment_text(texts, num_aug=1):
    aug = naw.SynonymAug(aug_src='wordnet')
    return [aug.augment(text, num_aug=num_aug) for text in texts]

augmented_texts = augment_text(original_texts)
```

## Handling Imbalanced Datasets

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def balance_dataset(X, y):
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    return pipeline.fit_resample(X, y)

X_balanced, y_balanced = balance_dataset(X, y)
```

## Implementing Custom DataLoaders

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomTorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Usage
dataset = CustomTorchDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

These advanced techniques for working with custom datasets can help you preprocess, augment, and manage your data more effectively when using AILib.