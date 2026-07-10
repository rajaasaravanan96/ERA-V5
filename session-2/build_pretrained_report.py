"""
build_pretrained_report.py — run this LOCALLY (not on the free server).

1. Fetches the 4 Wikipedia "India" pages once and caches them under data/
   (data/wiki_en.txt, wiki_hi.txt, wiki_te.txt, wiki_ta.txt). Commit these
   files — the deployed app then reads from disk instead of calling
   Wikipedia's API on every request.
2. Encodes each cached page with the already-trained tokenizer.json and
   writes data/pretrained_report.json: words vs tokens vs fertility per
   language, computed once, statically, with zero server-side work.

Run it whenever tokenizer.json or the source pages change:
    python build_pretrained_report.py
"""

import json
import os

from tokenizer_core import bytes_to_unicode, build_merge_rank, fertility_of
from wiki_fetch import fetch_wiki_text

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(HERE, "tokenizer.json")
REPORT_PATH = os.path.join(HERE, "data", "pretrained_report.json")

LANGS = {
    "en": {"label": "English", "code": "en", "title": "India"},
    "hi": {"label": "Hindi",   "code": "hi", "title": "भारत"},
    "te": {"label": "Telugu",  "code": "te", "title": "భారతదేశం"},
    "ta": {"label": "Tamil",   "code": "ta", "title": "இந்தியா"},
}


def main():
    byte_to_char, _ = bytes_to_unicode()

    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        tok = json.load(f)
    merge_rank = build_merge_rank([tuple(p) for p in tok["merges"]])

    fert = {}
    for key, cfg in LANGS.items():
        print(f"fetching/caching {cfg['label']} ({cfg['code']})...")
        text = fetch_wiki_text(cfg["code"], cfg["title"])
        f = fertility_of(text, merge_rank, byte_to_char)
        fert[key] = {**f, "label": cfg["label"], "code": cfg["code"]}
        print(f"  {cfg['label']}: {f['words']} words -> {f['tokens']} tokens, fertility {f['fertility']:.4f}")

    vals = [v["fertility"] for v in fert.values()]
    spread = max(vals) - min(vals)
    score = (1000 / spread) if spread > 0 else None

    report = {"fert": fert, "spread": spread, "score": score, "vocab_size": len(tok["vocab"])}

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nspread={spread:.4f}  score={score:.1f}" if score else f"\nspread={spread}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
