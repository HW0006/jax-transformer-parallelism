"""Character-level tokenizer and batching for the tiny-Shakespeare dataset."""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int32)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load_dataset(path: str, val_fraction: float = 0.1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    n_val = int(len(data) * val_fraction)
    train_data, val_data = data[:-n_val], data[-n_val:]
    return tokenizer, train_data, val_data


def get_batch(rng: np.random.Generator, data: np.ndarray, batch_size: int, block_size: int):
    """Returns (x, y) of shape (batch_size, block_size): y is x shifted by one
    (next-token prediction)."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
