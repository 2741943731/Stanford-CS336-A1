import torch
import json
import math
import pickle
import numpy as np
from torch import nn
from einops import rearrange, einsum
from torch.nn import functional as F

class Linear(nn.Module):
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

class Embedding(nn.Module):
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

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def SiLU(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pos = torch.arange(self.max_seq_len)
        freq = 1 / (self.theta ** (torch.arange(0, d_k, 2).float() / self.d_k))

        angle = pos[:, None] * freq[None, :]
        self.register_buffer("cos", angle.cos())
        self.register_buffer("sin", angle.sin())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x (... seq, d)
        token_positions (... seq)
        """
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        return torch.stack([x_even_rotated, x_odd_rotated], dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

