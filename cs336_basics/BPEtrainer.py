import os
import torch
import json
import pickle
import regex as re
from collections import defaultdict
from pprint import pprint
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    next_vocab_index = 256
    token_frequency_table = defaultdict(int)
    existing_byte_values = set(vocab.values())

    for st in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes = st.encode("utf-8")
        if st_bytes not in existing_byte_values:
            vocab[next_vocab_index] = st_bytes
            existing_byte_values.add(st_bytes)
            next_vocab_index += 1

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = re.split("|".join(map(re.escape, special_tokens)), text)

    for chunk in chunks:
        for word in re.findall(PAT, chunk):
            # pprint(word)
            word_bytes = word.encode("utf-8")
            byte_list = [bytes([i]) for i in word_bytes]
            # pprint(byte_list)
            token_frequency_table[tuple(byte_list)] += 1

    pair_counts = defaultdict(int)

    for token, freq in token_frequency_table.items():
        for i in range(0, len(token) - 1):
            pair_counts[token[i], token[i + 1]] += freq

    while len(vocab) < vocab_size:
        # print("-" * 20 + "round" + "-" * 20 )
        if not pair_counts:
            break
        max_value = max(pair_counts.values())
        candidates = [key for key, value in pair_counts.items() if value == max_value]
        largest_candidate = max(candidates)

        merges.append(largest_candidate)
        new_token = largest_candidate[0] + largest_candidate[1]
        vocab[next_vocab_index] = new_token
        next_vocab_index += 1

        # pprint(new_token)

        token_changed = []
        for token, freq in token_frequency_table.items():
            flag = any(token[i:i+2] == largest_candidate for i in range(0, len(token) - 1))
            if flag:
                token_changed.append((token, freq))
        for token, freq in token_changed:
            for i in range(0, len(token) - 1):
                pair_counts[token[i], token[i + 1]] -= freq
                if pair_counts[token[i], token[i + 1]] <= 0:
                    del pair_counts[token[i], token[i + 1]]

            t = 0
            new_token_frequency_seq = []
            while t < len(token):
                if t < len(token) - 1 and token[t] + token[t + 1] == new_token:
                    new_token_frequency_seq.append(new_token)
                    t += 2
                else:
                    new_token_frequency_seq.append(token[t])
                    t += 1

            for i in range(0, len(new_token_frequency_seq) - 1):
                pair_counts[new_token_frequency_seq[i], new_token_frequency_seq[i + 1]] += freq

            del token_frequency_table[token]
            token_frequency_table[tuple(new_token_frequency_seq)] += freq

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = run_train_bpe("tests/fixtures/tinystories_sample.txt", 20000, ["<|endoftext|>"])
    # pprint(vocab)
    # pprint(merges)

