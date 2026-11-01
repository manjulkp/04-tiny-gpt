"""Train the Tiny GPT and write its loss curve, checkpoint and text samples."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from tinygpt.data import bigram_val_loss, load_splits, uniform_loss
from tinygpt.model import GPT, GPTConfig
from tinygpt.plots import plot_loss
from tinygpt.training import TrainConfig, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embd", type=int, default=128)
    parser.add_argument("--block", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not (Path(args.data) / "train.txt").exists():
        parser.error(f"{args.data}/train.txt not found. Run prepare_data.py first.")

    tokenizer, train_ids, val_ids = load_splits(args.data)
    torch.manual_seed(args.seed)
    config = GPTConfig(tokenizer.vocab_size, args.block, args.layers, args.heads, args.embd)
    model = GPT(config)
    print(
        f"{len(train_ids):,} training characters, {tokenizer.vocab_size} distinct characters, "
        f"{model.num_parameters():,} parameters",
        flush=True,
    )

    train_config = TrainConfig(max_iters=args.iters, seed=args.seed, eval_every=max(1, args.iters // 10))
    history = train(model, train_ids, val_ids, train_config, log=lambda text: print(text, flush=True))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state": model.state_dict()}, out / "model.pt")
    tokenizer.save(out / "tokenizer.json")

    baselines = {
        "Uniform guess": uniform_loss(tokenizer.vocab_size),
        "Bigram model": bigram_val_loss(train_ids, val_ids, tokenizer.vocab_size),
    }
    best_val = min(row["val_loss"] for row in history)
    plot_loss(history, baselines, "Tiny GPT training on movie dialogue", out / "loss.png")

    # Start from the opening of a held-out movie, which works for any dataset.
    prompt = val_ids[: args.block // 2].unsqueeze(0)
    generator = torch.Generator().manual_seed(args.seed)
    sample = tokenizer.decode(model.generate(prompt, 800, temperature=0.8, generator=generator)[0].tolist())
    (out / "sample.txt").write_text(sample, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {
                "parameters": model.num_parameters(),
                "best_val_loss": best_val,
                "baselines": baselines,
                "seconds": history[-1]["seconds"],
                "history": history,
            },
            indent=1,
        )
    )
    print(f"\nBest validation loss {best_val:.4f}; baselines {baselines}\n\n{sample}")


if __name__ == "__main__":
    main()
