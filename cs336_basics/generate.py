import torch
import pickle
import argparse
import math
import time
import numpy as np

from cs336_basics.BPEtrainer import run_train_bpe
from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.nn_utils import Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax
from cs336_basics.model import ScaledDotProductAttention, CausalMultiHeadAttention, TransformerBlock, TransformerLM
from cs336_basics.crossEntropy import CrossEntropy
from cs336_basics.optimizer import AdamW
from cs336_basics.learningRateSchedule import learningRateSchedule
from cs336_basics.gradientClip import gradientClip
from cs336_basics.dataLoader import dataLoader
from cs336_basics.checkPoint import load_checkpoint, save_checkpoint

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

    keep = cumulative_probs <= top_p
    if not torch.any(keep):
        keep[0] = True
    sorted_logits = keep * sorted_logits
    out = torch.zeros_like(logits)
    out[sorted_indices] = sorted_logits
    return out / out.sum(dim=-1, keepdim=True)

def generate(model,
             tokenizer,
             prompt: str,
             temperature: float=1.0,
             max_length: int=200,
             top_p: float=1.0,
             end_token: str="<|endoftext|>",
             device: torch.device="cpu") -> torch.Tensor:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    context_length = model.context_length
    eos_id = tokenizer.encode(end_token)[0]

    with torch.no_grad():
        for _ in range(max_length):
            win = input_ids[-context_length:]
            x = torch.tensor(win, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(x)
            next_token_logits = logits[0, -1, :] / temperature
            softmaxed_logits = torch.softmax(next_token_logits, dim=-1)
            filtered_logits = top_p_filtering(softmaxed_logits, top_p)
            next_id = int(torch.multinomial(filtered_logits, num_samples=1))

            if next_id == eos_id:
                break;
            input_ids.append(next_id)

    return tokenizer.decode(input_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default="vocab.pkl")
    parser.add_argument("--merges_file", type=str, default="merges.pkl")

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344) # 8 / 3 * d_model
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges_file, "rb") as f:
        merges = pickle.load(f)
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    weight = torch.load("checkpoints/model_checkpoint.pt", map_location=torch.device('cpu'))['model.state_dict']
    model.load_state_dict(weight)

    prompt = "One upon a time,"
    generated = generate(model, tokenizer, prompt)
    print(prompt + generated)