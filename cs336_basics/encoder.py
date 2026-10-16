import torch
import pickle
import argparse
import numpy as np

from cs336_basics.Tokenizer import Tokenizer

def encodeCorpus(tokenizer: Tokenizer, 
                 input_file: str, 
                 output_file: str, 
                 batch_size: int = 1_000_000):
    """
    Encodes a text corpus using the provided tokenizer and saves the encoded data to a binary file.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        input_file (str): Path to the input text file.
        output_file (str): Path to the output binary file where encoded data will be saved.
        batch_size (int): Number of lines to process in each batch. Default is 1,000,000.
    """
    buffer = []
    total = 0

    with open(input_file, "r", encoding="utf-8", errors="ignore") as fin, \
            open(output_file, "wb") as fout:
        for tid in tokenizer.encode_iterable(fin):
            buffer.append(tid)
            if len(buffer) >= batch_size:
                np.asarray(buffer, dtype=np.uint16).tofile(fout)
                total += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(fout)
            total += len(buffer)
    return total

if __name__ == "__main__":
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("merges.pkl", "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    n = encodeCorpus(tokenizer, "data/data/TinyStoriesV2-GPT4-train.txt", "data/data/encoded_TinyStoriesV2_GPT4_train.npy")
    print(f"encoded_TinyStoriesV2_GPT4_train.npy: {n:,} tokens, {n * 2 / 1e9:.2f} GB")
    t = encodeCorpus(tokenizer, "data/data/TinyStoriesV2-GPT4-valid.txt", "data/data/encoded_TinyStoriesV2_GPT4_valid.npy")
    print(f"encoded_TinyStoriesV2_GPT4_valid.npy: {t:,} tokens, {t * 2 / 1e9:.2f} GB")
