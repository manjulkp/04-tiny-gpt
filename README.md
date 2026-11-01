# Project 4: Tiny GPT

A character-level GPT written from scratch in PyTorch and trained on movie dialogue
from the Cornell Movie-Dialogs Corpus. Projects 5, 10 and 11 reuse its code and data.

## Results

The model has 4 layers, 4 heads, a 128 embedding size and 827,392 parameters. It
trained for 4,000 iterations in about 21 minutes on a CPU. Validation uses 10% of the
movies, which the model never saw.

| Model | Validation loss (lower is better) |
| --- | --- |
| Uniform guess | 4.533 |
| Bigram model (previous character only) | 2.523 |
| Tiny GPT | 1.540 |

Training loss ended at 1.509, close to validation, so the model isn't overfitting much.

Here's part of a generated sample. It has learned the script format, character names
and common phrases, but it doesn't make sense yet and it invents words:

```text
KING ARNULF: Open up! I know you're in there!
AUPER: He didn't get here.
HANNY: You'll see me?
LUH: Grother the man was right.
ELLIE: But this world.
WOMAN: Don't you start to see in good?
```

## Run it

```powershell
uv sync
uv run python prepare_data.py     # download the corpus, about 10 MB
uv run python train.py            # train the model, about 20 minutes on a CPU
uv run python sample.py --prompt "JOHN: " --temperature 0.8
uv run pytest
```

Results are in `outputs\`: `loss.png`, `sample.txt`, `results.json` and the model.

### Training options

| Option | Default | Meaning |
| --- | --- | --- |
| `--iters` | `4000` | Training iterations |
| `--layers` | `4` | Transformer blocks |
| `--heads` | `4` | Attention heads |
| `--embd` | `128` | Embedding size |
| `--block` | `96` | Longest context, in characters |
| `--seed` | `0` | Random seed |

## Files

| File | What it does |
| --- | --- |
| `tinygpt/model.py` | Attention, transformer blocks and the GPT |
| `tinygpt/data.py` | Character tokenizer, batching and the bigram baseline |
| `tinygpt/training.py` | Training loop and learning rate schedule |
| `tinygpt/plots.py` | Shared chart styling |
| `prepare_data.py` | Downloads the corpus and holds out whole movies for validation |
| `train.py` | Trains the model and writes the results |
| `sample.py` | Generates text from a trained model |
| `tests/` | Includes a check that future characters can't affect earlier predictions |
