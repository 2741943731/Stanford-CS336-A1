import torch
import numpy as np

def dataLoader(x: np.array, 
               batch_size: int,
               context_length: int,
               device: torch.device):
    x_len = len(x)
    start = np.random.randint(0, x_len - context_length, size=batch_size)
    batch = np.asarray(np.stack([x[i: i + context_length] for i in start]), dtype=np.int64)
    target = np.asarray(np.stack([x[i + 1: i + context_length + 1] for i in start]), dtype=np.int64)
    return torch.tensor(batch).to(device), torch.tensor(target).to(device)

