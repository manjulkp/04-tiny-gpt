import math

import pytest
import torch

from prepare_data import build_scripts, split_movies
from sample import load_model
from sample import main as sample_main
from tinygpt.data import CharTokenizer, bigram_val_loss, get_batch, load_splits, uniform_loss
from tinygpt.model import GPT, GPTConfig
from tinygpt.training import TrainConfig, learning_rate_at, train
from train import main as train_main

SMALL = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def test_tokenizer_round_trip():
    tokenizer = CharTokenizer.from_text("hello world")
    assert tokenizer.decode(tokenizer.encode("low")) == "low"
    assert tokenizer.vocab_size == len(set("hello world"))


def test_get_batch_targets_are_inputs_shifted_by_one():
    ids = torch.arange(100)
    x, y = get_batch(ids, block_size=8, batch_size=4)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)


def test_batches_can_reach_the_last_characters():
    ids = torch.arange(12)
    seen = {get_batch(ids, 8, 1)[1].max().item() for _ in range(300)}
    assert 11 in seen


def test_output_shapes():
    logits, loss = GPT(SMALL)(torch.zeros(3, 10, dtype=torch.long), torch.zeros(3, 10, dtype=torch.long))
    assert logits.shape == (3, 10, 20)
    assert loss.ndim == 0


def test_rejects_input_longer_than_block_size():
    with pytest.raises(ValueError, match="exceeds block_size"):
        GPT(SMALL)(torch.zeros(1, 17, dtype=torch.long))


def test_rejects_bad_head_count():
    with pytest.raises(ValueError, match="divisible"):
        GPT(GPTConfig(vocab_size=5, n_embd=10, n_head=3))


def test_future_tokens_cannot_affect_past_predictions():
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    ids = torch.randint(20, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 20
    with torch.no_grad():
        first, _ = model(ids)
        second, _ = model(changed)
    assert torch.equal(first[0, :8], second[0, :8])
    assert not torch.equal(first[0, 8:], second[0, 8:])


def test_initial_loss_is_close_to_uniform_guess():
    torch.manual_seed(0)
    ids = torch.randint(20, (32, 16))
    _, loss = GPT(SMALL)(ids, ids)
    assert loss.item() == pytest.approx(math.log(20), abs=0.2)


def test_can_memorize_a_small_batch():
    torch.manual_seed(0)
    model = GPT(SMALL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = torch.randint(20, (4, 16)), torch.randint(20, (4, 16))
    for _ in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


def test_generate_appends_tokens_and_uses_a_sliding_window():
    model = GPT(SMALL)
    output = model.generate(torch.zeros(2, 3, dtype=torch.long), 30)
    assert output.shape == (2, 33)
    assert output.max() < 20


def test_top_k_one_is_greedy():
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    start = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        expected = model(start)[0][0, -1].argmax().item()
    assert model.generate(start, 1, top_k=1)[0, 1].item() == expected


def test_generate_restores_training_mode():
    model = GPT(SMALL).train()
    model.generate(torch.zeros(1, 1, dtype=torch.long), 2)
    assert model.training


def test_learning_rate_schedule():
    config = TrainConfig(max_iters=1000, learning_rate=1.0, warmup_iters=100)
    assert learning_rate_at(0, config) == pytest.approx(0.01)
    assert learning_rate_at(99, config) == pytest.approx(1.0)
    assert learning_rate_at(100, config) == pytest.approx(1.0)
    assert learning_rate_at(1000, config) == pytest.approx(0.0)


def test_bigram_loss_beats_uniform_on_patterned_text():
    ids = torch.tensor([0, 1, 2] * 100)
    assert bigram_val_loss(ids, ids, 3) < 0.05 < uniform_loss(3)


def test_build_scripts_orders_turns_and_skips_bad_rows():
    sep = " +++$+++ "
    lines = "\n".join(
        [
            sep.join(["L2", "u1", "m0", "ANNA", "Hello caf\xe9."]),
            sep.join(["L1", "u2", "m0", "BEN", "Hi."]),
            "a malformed row",
        ]
    )
    conversations = sep.join(["u1", "u2", "m0", "['L1', 'L2', 'L_missing']"])
    assert build_scripts(lines, conversations) == {"m0": "BEN: Hi.\nANNA: Hello caf.\n\n"}


def test_split_keeps_each_movie_in_one_set():
    scripts = {f"m{i}": f"<{i}>" for i in range(20)}
    train_text, val_text = split_movies(scripts)
    assert val_text.count("<") == 2
    assert train_text.count("<") == 18
    assert not set(train_text.split(">")[:-1]) & set(val_text.split(">")[:-1])


def test_training_lowers_loss_and_keeps_best_weights():
    text = "ANNA: hello there\nBEN: hello back\n\n" * 200
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text))
    torch.manual_seed(0)
    model = GPT(GPTConfig(tokenizer.vocab_size, 16, 1, 2, 16, 0.0))
    config = TrainConfig(max_iters=120, eval_every=30, eval_iters=5, batch_size=8, warmup_iters=5, learning_rate=3e-3)
    history = train(model, ids, ids, config, log=None)
    assert history[-1]["val_loss"] < history[0]["val_loss"] / 2
    assert not model.training


def test_same_seed_reproduces_training():
    ids = torch.randint(20, (500,))
    config = TrainConfig(max_iters=10, eval_every=10, eval_iters=2, batch_size=4)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        runs.append(train(GPT(SMALL), ids, ids, config, log=None))
    assert [r["val_loss"] for r in runs[0]] == [r["val_loss"] for r in runs[1]]


def test_train_and_sample_scripts_end_to_end(tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.txt").write_text("ANNA: hello there\nBEN: hi\n\n" * 50)
    (data / "val.txt").write_text("BEN: hello\nANNA: hi there\n\n" * 10)
    out = tmp_path / "out"
    train_main(
        ["--data", str(data), "--out", str(out), "--iters", "20", "--layers", "1", "--heads", "2", "--embd", "16", "--block", "16"]
    )
    for name in ("model.pt", "tokenizer.json", "loss.png", "sample.txt", "results.json"):
        assert (out / name).exists()

    model, tokenizer = load_model(out)
    assert tokenizer.vocab_size == model.config.vocab_size
    capsys.readouterr()
    sample_main(["--out", str(out), "--prompt", "ANNA", "--tokens", "5"])
    assert capsys.readouterr().out.startswith("ANNA")

    with pytest.raises(SystemExit):
        sample_main(["--out", str(out), "--prompt", "@@@"])
    assert load_splits(data)[0].vocab_size == tokenizer.vocab_size
