"""
train_tokenizer.py — train the assignment tokenizer from the faithful
Markdown corpus, following the ERA V5 Session 2 reference solution.

Training choices (same as the reference):
    Model:        HuggingFace BPE
    Vocab size:   10,000
    min_frequency = 1
    Normalizer:   NFKC only
    Pretokenizer: Metaspace, using ▁ as the space marker
    Decoder:      Metaspace

Metaspace is used instead of ByteLevel because ByteLevel spends too many
tokens on UTF-8 bytes for Indic scripts. Metaspace preserves punctuation,
brackets, URL characters, apostrophes, number separators and spaces, so
decode(encode(text)) keeps the same visible characters.

Per-language training weights (how many times each corpus is repeated in the
training stream) are tuned so every language's fertility stays under the 1.2
threshold with the smallest spread:

    python train_tokenizer.py                  # uses WEIGHTS below
    python train_tokenizer.py en=3 hi=4 te=4 ta=3   # override

Dependencies:  pip install tokenizers
"""

import os
import sys

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
OUT_PATH = os.path.join(HERE, "tokenizer.json")

LANGS = ["en", "hi", "te", "ta"]
VOCAB_SIZE = 10000
# chosen by grid search over the faithful corpus: minimal fertility spread
# with every language under the 1.2 threshold (see metrics.json)
WEIGHTS = {"en": 2, "hi": 3, "te": 5, "ta": 2}


def corpus_iterator(weights):
    """Yield corpus lines, each language's file repeated `weight` times."""
    for lang in LANGS:
        path = os.path.join(CORPUS_DIR, f"{lang}.faithful.txt")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for _ in range(weights.get(lang, 1)):
            yield from lines


def full_alphabet(weights):
    """Every character that appears anywhere in the corpus, so no character
    can ever be dropped as unknown — this is what keeps the faithful
    roundtrip gate satisfiable."""
    chars = set()
    for lang in LANGS:
        with open(os.path.join(CORPUS_DIR, f"{lang}.faithful.txt"), encoding="utf-8") as f:
            chars.update(f.read())
    chars.discard("\n")
    return sorted(chars)


def train(weights=None, out_path=OUT_PATH, quiet=False):
    weights = weights or WEIGHTS
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁")
    tok.decoder = decoders.Metaspace(replacement="▁")

    alphabet = full_alphabet(weights)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=alphabet,
        limit_alphabet=max(len(alphabet) + 100, 2000),
    )
    tok.train_from_iterator(corpus_iterator(weights), trainer=trainer)
    tok.save(out_path)
    if not quiet:
        print(f"trained: vocab={tok.get_vocab_size()}  weights={weights}")
        print(f"wrote {out_path}")
    return tok


if __name__ == "__main__":
    overrides = dict(WEIGHTS)
    for arg in sys.argv[1:]:
        k, v = arg.split("=")
        overrides[k] = int(v)
    train(overrides)
