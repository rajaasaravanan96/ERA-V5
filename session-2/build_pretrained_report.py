"""
build_pretrained_report.py — superseded.

The pretrained tokenizer's report is now metrics.json, produced by the
reference-solution pipeline:

    python build_wiki_faithful_markdown.py   # fetch + convert corpus
    python train_tokenizer.py                # train tokenizer.json
    python evaluate_tokenizer.py             # write metrics.json

The app (app.py) reads metrics.json directly; data/pretrained_report.json is
no longer used.
"""

if __name__ == "__main__":
    print(__doc__)
