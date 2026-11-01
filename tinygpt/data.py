"""Character-level text data: tokenizer, batching and a bigram baseline."""

import json
import math
from pathlib import Path

import torch


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.to_id = {char: i for i, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.to_id[char] for char in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.chars), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def load_splits(data_dir, max_chars=None):
    """Return (tokenizer, train_ids, val_ids) from train.txt and val.txt."""
    data_dir = Path(data_dir)
    train_text = (data_dir / "train.txt").read_text(encoding="utf-8")
    val_text = (data_dir / "val.txt").read_text(encoding="utf-8")
    if max_chars:
        train_text = train_text[:max_chars]
    # Both splits contribute characters. They're cleaned to ASCII, so this only
    # avoids a crash on a rare character and doesn't reveal anything useful.
    tokenizer = CharTokenizer.from_text(train_text + val_text)
    encode = lambda text: torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokenizer, encode(train_text), encode(val_text)


def get_batch(ids, block_size, batch_size, generator=None):
    """Sample random windows. The targets are the inputs shifted along by one character."""
    starts = torch.randint(len(ids) - block_size, (batch_size,), generator=generator)
    x = torch.stack([ids[start : start + block_size] for start in starts])
    y = torch.stack([ids[start + 1 : start + block_size + 1] for start in starts])
    return x, y


def bigram_val_loss(train_ids, val_ids, vocab_size):
    """Loss of a model that only looks at the previous character, as a baseline."""
    pairs = train_ids[:-1] * vocab_size + train_ids[1:]
    counts = torch.bincount(pairs, minlength=vocab_size**2).reshape(vocab_size, vocab_size).float()
    log_probs = (counts + 1).log() - (counts + 1).sum(dim=1, keepdim=True).log()
    return -log_probs[val_ids[:-1], val_ids[1:]].mean().item()


def uniform_loss(vocab_size):
    """Loss of a model that treats every character as equally likely."""
    return math.log(vocab_size)
