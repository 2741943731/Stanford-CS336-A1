import torch
import typing
import os
from torch import nn

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    torch.save({'model.state_dict': model.state_dict(),
                'optimizer.state_dict': optimizer.state_dict(),
                'iter': iteration}, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model.state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
    iter = checkpoint['iter']
    return iter