import torch
import math
from torch import nn
from einops import rearrange, einsum
from torch.nn import functional as F
from cs336_basics.nn_utils import RoPE, Linear, RMSNorm, SwiGLU, Embedding, softmax

def ScaledDotProductAttention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v)

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int=None, theta: float=None, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wk = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wv = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))
        self.Wo = nn.Parameter(torch.empty(self.d_model, self.d_model, device=device, dtype=dtype))

        std = math.sqrt(1 / (self.d_model))
        torch.nn.init.trunc_normal_(self.Wq, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.Wk, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.Wv, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.Wo, std=std, a=-3 * std, b=3 * std)

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.RoPE = RoPE(self.theta, self.head_dim, self.max_seq_len, device) if theta is not None else None

    def forward(self, x, token_positions=None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = x @ self.Wq.T
        k = x @ self.Wk.T
        v = x @ self.Wv.T
        # b s d_model @ d_model d_model = b s d_model

        q = rearrange(q, "batch_size seq_len (num_heads d_head) -> batch_size num_heads seq_len d_head", num_heads=self.num_heads)
        k = rearrange(k, "batch_size seq_len (num_heads d_head) -> batch_size num_heads seq_len d_head", num_heads=self.num_heads)
        v = rearrange(v, "batch_size seq_len (num_heads d_head) -> batch_size num_heads seq_len d_head", num_heads=self.num_heads)

        if token_positions is None:
            token_positions = torch.arange(seq_len).to(x.device)

        if self.RoPE is not None:
            q = self.RoPE(q, token_positions)
            k = self.RoPE(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()
        attention = ScaledDotProductAttention(q, k, v, mask)
        attention = rearrange(attention, "b n s d -> b s (n d)")
        attention = attention @ self.Wo.T
        return attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.causalMultiHeadSelfAttentionWithRoPE = CausalMultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.norm1(x)
        out1 = self.causalMultiHeadSelfAttentionWithRoPE(out1)
        out1 = out1 + x

        out2 = self.norm2(out1)
        out2 = self.ffn(out2)
        out2 = out2 + out1
        return out2

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.tokenEmbedding = Embedding(vocab_size, d_model)
        self.transformerBlocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.outputEmbedding = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenEmbedding(x)
        for layer in self.transformerBlocks:
            x = layer(x)
        x = self.norm(x)
        x = self.outputEmbedding(x)
        # logits = softmax(x, dim=-1)
        return x