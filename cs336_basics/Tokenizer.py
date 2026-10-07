import os
import torch
import json
import pickle
import regex as re
from collections import defaultdict
from pprint import pprint
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [] if not special_tokens else special_tokens
        self.merges_priority_map = {pair : i for i, pair in enumerate(merges)}
        self.bytes2id = {v: k for k, v in vocab.items()}
        """
        vocab: dict[int, bytes]
        merges: list[tuple(bytes, bytes)]
        special: list[str]
        """

    def singleWord2BPEbytes(self, word: bytes) -> list[bytes]:
        res = [bytes([byte]) for byte in word]
        while len(res) > 1:
            pairs = set()
            for i in range(0, len(res) - 1):
                pair = (res[i], res[i + 1])
                if pair in self.merges_priority_map.keys():
                    pairs.add(pair)

            if not pairs:
                break

            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            next_res = []
            t = 0
            while t < len(res):
                if t < len(res) - 1 and (res[t], res[t + 1]) == best_pair:
                    next_res.append(res[t] + res[t + 1])
                    t += 2
                else:
                    next_res.append(res[t])
                    t += 1
            res = next_res
        return res

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))

        if self.special_tokens:
            chunks = re.split(f'({pattern})', text)
        else:
            chunks = [text]

        idList = []
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                idList.append(self.bytes2id[chunk.encode("utf-8")])
            else:
                words = re.findall(PAT, chunk)
                for word in words:
                    if not word:
                        continue

                    merged_word = self.singleWord2BPEbytes(word.encode("utf-8"))

                    for byte in merged_word:
                        idList.append(self.bytes2id[byte])
        return idList

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list) -> str:
        textBytes = b''.join(self.vocab[id] for id in ids)
        return textBytes.decode("utf-8", errors="replace")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        unicode2GPTbytes = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        vocab = {}
        merges = []
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_read = json.load(f)
            # vocab = {int(id): bytes([unicode2GPTbytes[token]]) for token, id in vocab.items()}
            for token, id in vocab_read.items():
                tokenBytes = b''
                for part in token:
                    tokenBytes += bytes([unicode2GPTbytes[part]])
                vocab[id] = tokenBytes


        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_read = [tuple(line.rstrip().split(" ")) for line in f]
            # merges = [(bytes([unicode2GPTbytes[token1]]), bytes([unicode2GPTbytes[token2]])) for token1, token2 in merges]
            for token1, token2 in merges_read:
                token1Bytes = b''
                token2Bytes = b''
                for part in token1:
                    token1Bytes += bytes([unicode2GPTbytes[part]])
                for part in token2:
                    token2Bytes += bytes([unicode2GPTbytes[part]])
                merges.append((token1Bytes, token2Bytes))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    



