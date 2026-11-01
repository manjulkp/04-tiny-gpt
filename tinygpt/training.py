"""The training loop, shared with the projects that build on this one."""

import copy
import math
import time
from dataclasses import dataclass

import torch

from tinygpt.data import get_batch


@dataclass(frozen=True)
class TrainConfig:
    max_iters: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-3
    warmup_iters: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_iters: int = 50
    seed: int = 0


def learning_rate_at(step, config):
    """Warm up linearly, then follow a cosine curve down towards zero."""
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters
    progress = (step - config.warmup_iters) / max(1, config.max_iters - config.warmup_iters)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, ids, block_size, config, generator):
    was_training = model.training
    model.eval()
    losses = [
        model(*get_batch(ids, block_size, config.batch_size, generator))[1].item()
        for _ in range(config.eval_iters)
    ]
    model.train(was_training)
    return sum(losses) / len(losses)


def train(model, train_ids, val_ids, config, log=print, block_size=None):
    """Train the model and leave it holding its best weights by validation loss."""
    block_size = block_size or model.config.block_size
    torch.manual_seed(config.seed)
    batch_generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history, best_val, best_state = [], float("inf"), None
    started = time.perf_counter()
    model.train()
    for step in range(config.max_iters + 1):
        if step % config.eval_every == 0 or step == config.max_iters:
            # A fresh generator with the same seed means that every evaluation,
            # in every experiment, measures the same batches.
            eval_generator = torch.Generator().manual_seed(1234)
            train_loss = estimate_loss(model, train_ids, block_size, config, eval_generator)
            val_loss = estimate_loss(model, val_ids, block_size, config, eval_generator)
            elapsed = time.perf_counter() - started
            history.append(
                {"iter": step, "train_loss": train_loss, "val_loss": val_loss, "seconds": elapsed}
            )
            if log:
                log(f"iter {step:6d}  train {train_loss:.4f}  val {val_loss:.4f}  {elapsed:6.0f}s")
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            if step == config.max_iters:
                break

        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, config)
        x, y = get_batch(train_ids, block_size, config.batch_size, batch_generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    model.load_state_dict(best_state)
    model.eval()
    return history
