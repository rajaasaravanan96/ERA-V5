"""
tokenize_only.py — load the already-trained tokenizer.json and just encode
text. No Flask, no Wikipedia fetch, no BPE training.

Usage:
    python tokenize_only.py "your text here"
    python tokenize_only.py            # then type text at the prompt
"""

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from tokenizer_core import (
    bytes_to_unicode,
    build_merge_rank,
    encode_word_symbols,
    pretokenize,
    symbols_to_text,
    word_count,
    word_to_symbols,
)

TOKENIZER_PATH = "tokenizer.json"


def load_tokenizer(path=TOKENIZER_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    merges = [tuple(pair) for pair in data["merges"]]
    return build_merge_rank(merges), data["meta"]


def encode(text, merge_rank, byte_to_char, char_to_byte):
    tokens = []
    for tok in pretokenize(text):
        syms = encode_word_symbols(word_to_symbols(tok, byte_to_char), merge_rank)
        tokens.extend(symbols_to_text(s, char_to_byte) for s in syms)
    return tokens


def main():
    byte_to_char, char_to_byte = bytes_to_unicode()
    merge_rank, meta = load_tokenizer()

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = input("Input text: ").strip()

    tokens = encode(text, merge_rank, byte_to_char, char_to_byte)
    words = word_count(text)
    fertility = (len(tokens) / words) if words else None

    print("\nInput: ", text)
    print("Output tokens:", tokens)
    print(f"\nword_count={words}  token_count={len(tokens)}  fertility={fertility}")


if __name__ == "__main__":
    main()
