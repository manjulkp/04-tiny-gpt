"""Generate text from a trained Tiny GPT."""

import argparse
from pathlib import Path

import torch

from tinygpt.data import CharTokenizer
from tinygpt.model import GPT, GPTConfig


def load_model(out_dir):
    out_dir = Path(out_dir)
    checkpoint = torch.load(out_dir / "model.pt", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state"])
    return model.eval(), CharTokenizer.load(out_dir / "tokenizer.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="JOHN: ")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not (Path(args.out) / "model.pt").exists():
        parser.error(f"{args.out}/model.pt not found. Run train.py first.")
    model, tokenizer = load_model(args.out)
    unknown = sorted(set(args.prompt) - set(tokenizer.chars))
    if unknown:
        parser.error(f"the prompt uses characters the model doesn't know: {unknown}")

    prompt = torch.tensor([tokenizer.encode(args.prompt)])
    generator = torch.Generator().manual_seed(args.seed)
    output = model.generate(prompt, args.tokens, args.temperature, args.top_k, generator)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
