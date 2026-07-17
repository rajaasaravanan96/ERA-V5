"""
evaluate_tokenizer.py — evaluate tokenizer.json with the assignment-stated
scoring, following the ERA V5 Session 2 reference solution.

    faithful_unit = one contiguous Unicode letter/mark/number run
                    OR one visible non-space punctuation/symbol character
    fertility(language) = token_count(language) / faithful_unit_count(language)
    score = 1000 / (max_fertility - min_fertility)

    hindi_penalty        = exp(max(0, hindi_fertility / 1.2 - 1))
    hindi_adjusted_score = raw_score / hindi_penalty

It also runs the faithful roundtrip gate — decode(encode(text)) must keep
the same visible non-whitespace characters — on the grader's failing sample,
Markdown/URL samples, and excerpts from every corpus file.

Writes metrics.json. Exit code 0 only if the gate passes and every language
is under the 1.2 fertility threshold.

Dependencies:  pip install tokenizers regex
Run:           python evaluate_tokenizer.py
"""

import json
import math
import os
import sys

import regex
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
TOKENIZER_PATH = os.path.join(HERE, "tokenizer.json")
METRICS_PATH = os.path.join(HERE, "metrics.json")

LANGS = {"en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil"}
THRESHOLD = 1.2

# one letter/mark/number run OR one visible non-space char
UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

GATE_SAMPLES = [
    "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
    "India's population is 1,428,627,663.",
    "# Heading\n\nSome **bold**, `code_with_underscores`, a [link](https://x.y/z?a=1&b=2).",
    "भारत एक विशाल देश है। [संदर्भ 12]",
    "இந்தியா ஒரு பெரிய நாடு (2011).",
    "తెలుగు: భారతదేశం | GDP $3.7T",
]


def faithful_units(text: str) -> int:
    return len(UNIT_RE.findall(text))


def visible(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def count_tokens(tok: Tokenizer, text: str) -> int:
    total = 0
    for line in text.splitlines():
        if line.strip():
            total += len(tok.encode(line).ids)
    return total


def roundtrip_gate(tok: Tokenizer, samples) -> list:
    failures = []
    for text in samples:
        decoded = tok.decode(tok.encode(text).ids)
        if visible(decoded) != visible(text):
            failures.append({"sample": text[:80], "decoded": decoded[:120]})
    return failures


def evaluate(tok=None, tokenizer_path=TOKENIZER_PATH, write=True, quiet=False):
    tok = tok or Tokenizer.from_file(tokenizer_path)

    per_lang, corpus_excerpts = {}, []
    for code, label in LANGS.items():
        with open(os.path.join(CORPUS_DIR, f"{code}.faithful.txt"), encoding="utf-8") as f:
            text = f.read()
        units = faithful_units(text)
        tokens = count_tokens(tok, text)
        per_lang[code] = {"label": label, "tokens": tokens, "faithful_units": units,
                          "fertility": tokens / units}
        corpus_excerpts += [ln for ln in text.splitlines() if ln.strip()][:40]

    ferts = {c: v["fertility"] for c, v in per_lang.items()}
    spread = max(ferts.values()) - min(ferts.values())
    raw_score = 1000 / spread if spread > 0 else None
    hindi_penalty = math.exp(max(0.0, ferts["hi"] / THRESHOLD - 1))
    adjusted = raw_score / hindi_penalty if raw_score else None
    over = [c for c, f in ferts.items() if f > THRESHOLD]

    gate_failures = roundtrip_gate(tok, GATE_SAMPLES + corpus_excerpts)

    metrics = {
        "vocab_size": tok.get_vocab_size(),
        "per_language": per_lang,
        "spread": spread,
        "raw_score": raw_score,
        "hindi_penalty_factor": hindi_penalty,
        "hindi_adjusted_score": adjusted,
        "fertility_threshold": THRESHOLD,
        "languages_over_threshold": over,
        "faithful_roundtrip_gate": "PASS" if not gate_failures else "FAIL",
        "gate_failures": gate_failures,
    }
    if write:
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not quiet:
        print(f"{'Language':10} {'Tokens':>10} {'Faithful Units':>15} {'Fertility':>10}")
        for c, v in per_lang.items():
            print(f"{v['label']:10} {v['tokens']:>10,} {v['faithful_units']:>15,} {v['fertility']:>10.6f}")
        print(f"\nSpread = {spread:.6f}")
        print(f"Raw score = 1000 / {spread:.6f} = {raw_score:.2f}")
        print(f"Hindi penalty factor = {hindi_penalty:.6f}")
        print(f"Hindi-adjusted score = {adjusted:.2f}")
        print(f"1.2 threshold: {'all languages satisfy it' if not over else 'OVER: ' + str(over)}")
        print(f"Faithful roundtrip gate: {metrics['faithful_roundtrip_gate']}")
        for fail in gate_failures[:5]:
            print(f"  gate failure: {fail}")
    return metrics


if __name__ == "__main__":
    m = evaluate()
    sys.exit(0 if m["faithful_roundtrip_gate"] == "PASS" and not m["languages_over_threshold"] else 1)
