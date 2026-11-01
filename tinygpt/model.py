"""A GPT: token and position embeddings, then a stack of transformer blocks."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # the longest context the model can see
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.query_key_value = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # The mask stops each position from attending to anything after it. This
        # is what makes the model a decoder.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        batch, time, channels = x.shape
        q, k, v = self.query_key_value(x).chunk(3, dim=-1)
        head_size = channels // self.n_head

        def split_heads(tensor):
            return tensor.view(batch, time, self.n_head, head_size).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        # Dot products grow with dimension, so scale them to keep softmax well behaved.
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)
        scores = scores.masked_fill(~self.mask[:time, :time], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, time, channels)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Attention lets tokens communicate, and the feed-forward layer lets each one think."""

    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # The residual connections give gradients a direct path through deep stacks.
        x = x + self.attention(self.norm_1(x))
        return x + self.feed_forward(self.norm_2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        length = ids.shape[1]
        if length > self.config.block_size:
            raise ValueError(f"input length {length} exceeds block_size {self.config.block_size}")
        positions = torch.arange(length, device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.config.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            ids = torch.cat([ids, next_id], dim=1)
        self.train(was_training)
        return ids

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
