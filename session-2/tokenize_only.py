"""
tokenize_only.py — load the trained tokenizer.json and encode/decode text.
No Flask, no Wikipedia fetch, no training.

Usage:
    python tokenize_only.py "your text here"
    python tokenize_only.py            # then type text at the prompt
"""

import io
import sys

from tokenizers import Tokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TOKENIZER_PATH = "tokenizer.json"


def visible(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def main():
    tok = Tokenizer.from_file(TOKENIZER_PATH)

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = input("Input text: ").strip()

    enc = tok.encode(text)
    decoded = tok.decode(enc.ids)

    print("\nInput: ", text)
    print("Output tokens:", [t.replace("▁", " ") for t in enc.tokens])
    print("Token ids:    ", enc.ids)
    print("Decoded:      ", decoded)
    print("Roundtrip:    ", "OK — visible text preserved"
          if visible(decoded) == visible(text) else "FAILED — visible text changed")
    print(f"\ntoken_count={len(enc.ids)}")


if __name__ == "__main__":
    main()
