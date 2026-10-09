import torch
from torch import nn
from einops import rearrange, einsum
from torch.nn import functional as F
from cs336_basics.nn_utils import RoPE

def ScaledDotProductAttention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v)

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int=None, theta: int=None, device: torch.device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.RoPE = RoPE(self.theta, self.head_dim, self.max_seq_len, device) if theta is not None else None

    def forward(self, x, wq, wk, wv, wo, token_positions=None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = x @ wq.T
        k = x @ wk.T
        v = x @ wv.T
        # b s d_model @ d_model d_model = b s d_model

        q = rearrange(q, "batch_size seq_len (num_heads d_head) -> batch_size num_heads seq_len d_head", num_heads=self.num_heads)
        k = rearrange(k, "batch_size seq_len (num_heads d_head) -> batch_size num_heads seq_len d_head", num_heads=self.num_heads)
        v = rearrange(v, "batch_size seq_len (num_heads d_head) -> batch_size num_heads seq_len d_head", num_heads=self.num_heads)

        if self.RoPE is not None:
            q = self.RoPE(q, token_positions)
            k = self.RoPE(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()
        attention = ScaledDotProductAttention(q, k, v, mask)
        attention = rearrange(attention, "b n s d -> b s (n d)")
        attention = attention @ wo.T
        return attention


