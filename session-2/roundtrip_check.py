"""
roundtrip_check.py — the grader's faithful-roundtrip gate, runnable locally.

decode(encode(text)) must keep the same visible non-whitespace characters as
text, for every sample. Loads tokenizer.json exactly like a grading harness
(Tokenizer.from_file) and checks the grader's failing sample, Markdown/URL
samples, and excerpts from every corpus file.

Run before resubmitting:
    python roundtrip_check.py
"""

import os
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "tokenizer.json")
CORPUS_DIR = os.path.join(HERE, "corpus")

SAMPLES = [
    # the exact sample the grader flagged
    "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
    # the reference solution's own roundtrip example
    "India's population is 1,428,627,663.",
    "# Heading\n\nSome **bold**, `code_with_underscores`, a [link](https://x.y/z?a=1&b=2).",
    "भारत एक विशाल देश है। [संदर्भ 12]",
    "இந்தியா ஒரு பெரிய நாடு (2011).",
    "తెలుగు: భారతదేశం | GDP $3.7T (2026)",
]


def visible(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def main():
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"vocab size: {tok.get_vocab_size()}")

    samples = list(SAMPLES)
    for name in sorted(os.listdir(CORPUS_DIR)) if os.path.isdir(CORPUS_DIR) else []:
        if name.endswith(".faithful.txt"):
            with open(os.path.join(CORPUS_DIR, name), encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            samples += lines[:200]

    ok = True
    fails = 0
    for text in samples:
        decoded = tok.decode(tok.encode(text).ids)
        if visible(decoded) != visible(text):
            ok = False
            fails += 1
            if fails <= 5:
                print(f"  FAIL: {text[:70]!r}\n        decoded: {decoded[:90]!r}")
    for text in SAMPLES:
        decoded = tok.decode(tok.encode(text).ids)
        status = "OK" if visible(decoded) == visible(text) else "FAIL"
        print(f"  [{status}] {text[:64]!r}")

    print(f"\nchecked {len(samples)} samples ({fails} failures)")
    print("RESULT:", "PASS — faithful roundtrip gate satisfied" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
