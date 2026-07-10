"""
tokenizer_core.py — byte-level BPE, pure stdlib.

Same algorithm used in the ERA V5 Session 2 assignment:
  - every language is turned into UTF-8 bytes, so English, Hindi, Telugu,
    Tamil (or anything else) are handled by one uniform algorithm with
    zero unknown-token risk.
  - the pretokenizer explicitly keeps letters + combining marks (matras)
    + format characters (ZWJ/ZWNJ) fused into one unit, so BPE never sees
    a word pre-split in the middle of a conjunct — the exact failure mode
    Session 2 covers for Indic scripts.
  - fertility = tokens_produced / whitespace_word_count, per language.
  - score = 1000 / (max_fertility - min_fertility), as defined in class.
"""

import unicodedata
from typing import Dict, List, Tuple


# ---------------------------------------------------------------- bytes <-> unicode
def bytes_to_unicode():
    """GPT-2's trick: map every byte value 0-255 to a printable unicode
    character, so byte sequences can be manipulated as ordinary strings."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    byte_to_char = dict(zip(bs, cs))
    char_to_byte = dict(zip(cs, bs))
    return byte_to_char, char_to_byte


def word_to_symbols(word: str, byte_to_char: dict) -> List[str]:
    return [byte_to_char[b] for b in word.encode("utf-8")]


def symbols_to_text(symbols: str, char_to_byte: dict) -> str:
    """Best-effort decode of a (possibly merged) token back to display text."""
    try:
        raw = bytes(char_to_byte[ch] for ch in symbols)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return symbols


# ---------------------------------------------------------------- pretokenizer
def _char_class(ch: str) -> str:
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)  # 'Lu','Ll','Lo','Mn','Mc','Cf','Nd', etc.
    if cat[0] in ("L", "M") or cat == "Cf":
        return "word"     # letters + combining marks + ZWJ/ZWNJ fused together
    if cat[0] == "N":
        return "number"
    return "other"


def pretokenize(text: str) -> List[str]:
    """Splits text into pretokens, GPT-2 style: a single leading space is
    folded into the following run, and — critically for Indic scripts —
    letters, combining marks (matras), and format joiners (ZWJ/ZWNJ) are
    never separated from the base letter they belong to."""
    tokens: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            if j - i == 1 and j < n:
                # fold this single space onto the next run as a leading space
                cls = _char_class(text[j])
                k = j
                while k < n and _char_class(text[k]) == cls:
                    k += 1
                tokens.append(" " + text[j:k])
                i = k
            else:
                tokens.append(text[i:j])
                i = j
        else:
            cls = _char_class(ch)
            j = i
            while j < n and _char_class(text[j]) == cls:
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def word_count(text: str) -> int:
    t = text.strip()
    return len(t.split()) if t else 0


# ---------------------------------------------------------------- BPE training
def _build_word_freqs(lang_texts: Dict[str, str], weights: Dict[str, int], byte_to_char):
    freq_map: Dict[tuple, int] = {}
    for lang, text in lang_texts.items():
        w = weights.get(lang, 1)
        for tok in pretokenize(text):
            symbols = tuple(word_to_symbols(tok, byte_to_char))
            freq_map[symbols] = freq_map.get(symbols, 0) + w
    return [{"symbols": list(k), "freq": v} for k, v in freq_map.items()]


def _remove_word_pairs(word, pair_counts, pair_words, idx):
    s = word["symbols"]
    for i in range(len(s) - 1):
        key = (s[i], s[i + 1])
        if key in pair_counts:
            pair_counts[key] -= word["freq"]
            if pair_counts[key] <= 0:
                del pair_counts[key]
        if key in pair_words:
            pair_words[key].discard(idx)
            if not pair_words[key]:
                del pair_words[key]


def _add_word_pairs(word, pair_counts, pair_words, idx):
    s = word["symbols"]
    for i in range(len(s) - 1):
        key = (s[i], s[i + 1])
        pair_counts[key] = pair_counts.get(key, 0) + word["freq"]
        pair_words.setdefault(key, set()).add(idx)


def _apply_merge(symbols, a, b, merged):
    out = []
    i, n = 0, len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(lang_texts: Dict[str, str], weights: Dict[str, int], vocab_size: int,
              byte_to_char, progress_cb=None) -> Tuple[List[str], List[Tuple[str, str]]]:
    words = _build_word_freqs(lang_texts, weights, byte_to_char)

    vocab: List[str] = []
    seen = set()
    for w in words:
        for s in w["symbols"]:
            if s not in seen:
                seen.add(s)
                vocab.append(s)

    merges: List[Tuple[str, str]] = []
    pair_counts: Dict[tuple, int] = {}
    pair_words: Dict[tuple, set] = {}
    for idx, w in enumerate(words):
        _add_word_pairs(w, pair_counts, pair_words, idx)

    target_merges = max(0, vocab_size - len(vocab))
    done = 0
    while done < target_merges:
        if not pair_counts:
            break
        best_key = max(pair_counts, key=pair_counts.get)
        if pair_counts[best_key] < 2:
            break
        a, b = best_key
        merged = a + b
        affected = list(pair_words.get(best_key, ()))
        for idx in affected:
            word = words[idx]
            _remove_word_pairs(word, pair_counts, pair_words, idx)
            word["symbols"] = _apply_merge(word["symbols"], a, b, merged)
            _add_word_pairs(word, pair_counts, pair_words, idx)
        merges.append((a, b))
        vocab.append(merged)
        done += 1
        if progress_cb and done % 250 == 0:
            progress_cb(done, target_merges, len(vocab))
    if progress_cb:
        progress_cb(done, target_merges, len(vocab))
    return vocab, merges


# ---------------------------------------------------------------- encode / fertility
def build_merge_rank(merges: List[Tuple[str, str]]) -> Dict[tuple, int]:
    return {pair: i for i, pair in enumerate(merges)}


def encode_word_symbols(symbols: List[str], merge_rank: dict) -> List[str]:
    syms = list(symbols)
    while len(syms) > 1:
        min_rank, min_pos = None, -1
        for i in range(len(syms) - 1):
            r = merge_rank.get((syms[i], syms[i + 1]))
            if r is not None and (min_rank is None or r < min_rank):
                min_rank, min_pos = r, i
        if min_pos == -1:
            break
        syms = syms[:min_pos] + [syms[min_pos] + syms[min_pos + 1]] + syms[min_pos + 2:]
    return syms


def count_tokens(text: str, merge_rank: dict, byte_to_char) -> int:
    total = 0
    for tok in pretokenize(text):
        total += len(encode_word_symbols(word_to_symbols(tok, byte_to_char), merge_rank))
    return total


def fertility_of(text: str, merge_rank: dict, byte_to_char) -> dict:
    words = word_count(text)
    tokens = count_tokens(text, merge_rank, byte_to_char)
    return {"words": words, "tokens": tokens, "fertility": (tokens / words) if words > 0 else float("nan")}
