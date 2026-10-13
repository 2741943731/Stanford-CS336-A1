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

def buildOptimizer(model, lr, betas, eps, weight_decay):
    """
    RMSnorm的权重不能衰减,所以要区分开
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, iters=50):
    model.eval()
    span = context_length
    width = span + 1
    max_start = len(data) - batch_size * width
    offsets = np.arange(batch_size * width).reshape(batch_size, width)
    losses = []
    for i in range(iters):
        start = (i * span * batch_size) % max_start
        idx = start + offsets
        x = torch.from_numpy(np.asarray(data[idx[:, :-1]], dtype=np.int64)).to(device)
        y = torch.from_numpy(np.asarray(data[idx[:, 1:]], dtype=np.int64)).to(device)
        logits = model(x)
        losses.append(CrossEntropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--valid_file", type=str, default="data/data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab_file", type=str, default="vocab.pkl")
    parser.add_argument("--merges_file", type=str, default="merges.pkl")
    parser.add_argument("--encoded_train_file", type=str, default="data/data/encoded_TinyStoriesV2_GPT4_train.npy")
    parser.add_argument("--encoded_valid_file", type=str, default="data/data/encoded_TinyStoriesV2_GPT4_valid.npy")
    # 文件路径

    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344) # 8 / 3 * d_model
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vocab_size", type=int, default=10000)
    # 模型参数

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    # 优化器参数

    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000)
    # 学习率调度参数

    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model_checkpoint.pt")
    parser.add_argument("--resume", default=False, action="store_true")
    # 训练参数

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()

    device = torch.device(args.device)

    train_data = np.memmap(args.encoded_train_file, dtype=np.uint16, mode="r")
    valid_data = np.memmap(args.encoded_valid_file, dtype=np.uint16, mode="r")
    print(f"train {len(train_data):,} tokens | valid {len(valid_data):,} tokens")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    # optimizer = AdamW(model.parameters(),
    #                 lr=args.lr, 
    #                 betas=[args.beta1, args.beta2], 
    #                 eps=args.eps, 
    #                 weight_decay=args.weight_decay)

    optimizer = buildOptimizer(model, 
                               lr=args.lr, 
                               betas=[args.beta1, args.beta2], 
                               eps=args.eps, 
                               weight_decay=args.weight_decay) 

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"Resumed from checkpoint at step {start_step}")

    model.train()
    t0 = time.time()
    ema = None

    for step in range(start_step, args.max_iters):
        lr = learningRateSchedule(step + 1, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr

        x, y = dataLoader(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = CrossEntropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        gradientClip(model.parameters(), maxL2Norm=args.max_grad_norm)
        optimizer.step()

        ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"Step {step}: train loss {ema:.4f}, time elapsed {elapsed:.2f}s, lr {lr:.6e}")

        if step % args.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, device)
            elapsed = time.time() - t0
            print(f"Step {step}: train loss {ema:.4f}, valid loss {val_loss:.4f}, time elapsed {elapsed:.2f}s")

        if step % args.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Checkpoint saved at step {step}")    

    save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)

if __name__ == "__main__":
    main()    






        





        









