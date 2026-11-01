"""Download the Cornell Movie-Dialogs Corpus and turn it into script-style text.

Whole movies are held out for validation, so the model is tested on films it
never saw, rather than on other lines from the same scene.
"""

import ast
import io
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
SEPARATOR = " +++$+++ "
DATA_DIR = Path("data")


def clean(text):
    """Keep plain ASCII so that the character vocabulary stays small."""
    text = text.replace("\t", " ").strip()
    return text.encode("ascii", errors="ignore").decode("ascii")


def build_scripts(lines_text, conversations_text):
    """Return a dictionary mapping each movie ID to its dialogue as script text."""
    lines = {}
    for row in lines_text.splitlines():
        fields = row.split(SEPARATOR)
        if len(fields) == 5:
            line_id, _, _, character, words = fields
            lines[line_id] = f"{clean(character)}: {clean(words)}"

    movies = defaultdict(list)
    for row in conversations_text.splitlines():
        fields = row.split(SEPARATOR)
        if len(fields) == 4:
            _, _, movie, line_ids = fields
            turns = [lines[i] for i in ast.literal_eval(line_ids) if i in lines]
            if turns:
                movies[movie].append("\n".join(turns))
    return {movie: "\n\n".join(scenes) + "\n\n" for movie, scenes in movies.items()}


def split_movies(scripts, val_fraction=0.1, seed=0):
    movies = sorted(scripts)
    random.Random(seed).shuffle(movies)
    n_val = max(1, round(len(movies) * val_fraction))
    val, train = movies[:n_val], movies[n_val:]
    return "".join(scripts[m] for m in train), "".join(scripts[m] for m in val)


if __name__ == "__main__":
    request = urllib.request.Request(URL, headers={"User-Agent": "tiny-gpt-learning-project"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    # Read the files straight from the zip in memory, so nothing gets unpacked to disk.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        prefix = "cornell movie-dialogs corpus/"
        lines_text = archive.read(prefix + "movie_lines.txt").decode("iso-8859-1")
        conversations_text = archive.read(prefix + "movie_conversations.txt").decode("iso-8859-1")

    scripts = build_scripts(lines_text, conversations_text)
    train_text, val_text = split_movies(scripts)
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "train.txt").write_text(train_text, encoding="utf-8", newline="\n")
    (DATA_DIR / "val.txt").write_text(val_text, encoding="utf-8", newline="\n")
    print(f"{len(scripts)} movies: {len(train_text):,} training and {len(val_text):,} validation characters")
    print(train_text[:400])
