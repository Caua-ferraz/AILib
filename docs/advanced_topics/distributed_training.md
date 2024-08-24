# Distributed Training

This guide covers advanced techniques for distributed training in AILib, focusing on LLMs and deep learning models.

## Data Parallel Training

### Using PyTorch DistributedDataParallel

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ailib import UnifiedModel

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    model = UnifiedModel('llm', model_name='gpt2')
    ddp_model = DDP(model.model, device_ids=[rank])
    
    # Training loop
    # ...
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## Model Parallel Training

For very large models that don't fit on a single GPU:

```python
import torch
from torch.distributed.pipeline.sync import Pipe

class PipelineParallelLLM(torch.nn.Module):
    def __init__(self, vocab_size, num_layers, num_gpus):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 512)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
            for _ in range(num_layers)
        ])
        self.output = torch.nn.Linear(512, vocab_size)
        
        # Split layers across GPUs
        self.partition_len = ((num_layers + 2) + num_gpus - 1) // num_gpus
        self.example_device = torch.device("cuda:0")

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x

def create_pipeline_parallel_model(vocab_size, num_layers, num_gpus):
    model = PipelineParallelLLM(vocab_size, num_layers, num_gpus)
    wrapper = torch.nn.Sequential(
        model.embed,
        *model.layers,
        model.output
    )
    return Pipe(wrapper, chunks=8)

# Usage
model = create_pipeline_parallel_model(vocab_size=50000, num_layers=24, num_gpus=4)
```

## Distributed Data Loading

```python
from torch.utils.data import DataLoader, DistributedSampler

def create_distributed_dataloader(dataset, rank, world_size, batch_size=32):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# In your training function
train_loader = create_distributed_dataloader(train_dataset, rank, world_size)
```

## Gradient Accumulation

For effective large batch training:

```python
def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.zero_grad()
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
```

## Distributed Evaluation

```python
def distributed_evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in eval_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    # Gather results from all processes
    world_size = dist.get_world_size()
    gathered_losses = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_losses, total_loss)
    
    if dist.get_rank() == 0:
        avg_loss = sum(gathered_losses) / len(gathered_losses)
        print(f"Average Evaluation Loss: {avg_loss}")

# Usage in training loop
distributed_evaluate(model, eval_dataloader, device)
```

These advanced distributed training techniques can significantly speed up the training process for large models and datasets in AILib.