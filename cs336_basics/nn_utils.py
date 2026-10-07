import torch
import json
import math
import pickle
import numpy as np
from torch import nn
from einops import rearrange, einsum
from torch.nn import functional as F

class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device))
        std = math.sqrt(2 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        torch.nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = x.pow(2).mean(-1, keepdim=True)
        rms = torch.sqrt(rms + self.eps)
        x = x / rms
        return einsum(x, self.g, "... d_model, d_model -> ... d_model").to(self.dtype)

