"""
build_wiki_faithful_markdown.py — fetch Wikipedia REST HTML for the four
assignment pages and convert each to a *faithful Markdown* corpus snapshot,
following the ERA V5 Session 2 reference solution.

"Faithful" means links, URLs, tables, references, image links and categories
are preserved wherever the HTML-to-Markdown conversion emits them — the
evaluation corpus is NOT clipped article prose.

Outputs, per language:
    corpus/<lang>.faithful.md     generated Markdown corpus snapshot
    corpus/<lang>.faithful.txt    same corpus as plain text input
    corpus/<lang>.meta.json       corpus metadata

Dependencies:  pip install requests beautifulsoup4 lxml markdownify regex
Run:           python build_wiki_faithful_markdown.py
"""

import json
import os
import unicodedata
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
UA = "ERA-V5-BPE-Assignment/2.0 (student resubmission)"

PAGES = {
    "en": {"label": "English", "title": "India"},
    "hi": {"label": "Hindi", "title": "भारत"},
    "te": {"label": "Telugu", "title": "భారతదేశం"},
    "ta": {"label": "Tamil", "title": "இந்தியா"},
}


def fetch_rest_html(lang: str, title: str) -> str:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title, safe='')}"
    r = requests.get(url, timeout=60, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text


def html_to_faithful_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["style", "script", "link", "meta"]):
        tag.decompose()
    text = md(str(soup.body or soup), heading_style="ATX")
    # collapse runs of 3+ blank lines but keep everything visible
    lines = [ln.rstrip() for ln in text.split("\n")]
    out, blanks = [], 0
    for ln in lines:
        blanks = blanks + 1 if not ln else 0
        if blanks <= 2:
            out.append(ln)
    text = "\n".join(out).strip() + "\n"
    # normalize once at build time so training/eval/gate all see NFKC text
    return unicodedata.normalize("NFKC", text)


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    for lang, cfg in PAGES.items():
        print(f"fetching {cfg['label']} ({lang}) — {cfg['title']} ...")
        html = fetch_rest_html(lang, cfg["title"])
        markdown = html_to_faithful_markdown(html)

        md_path = os.path.join(CORPUS_DIR, f"{lang}.faithful.md")
        txt_path = os.path.join(CORPUS_DIR, f"{lang}.faithful.txt")
        meta_path = os.path.join(CORPUS_DIR, f"{lang}.meta.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "language": cfg["label"], "code": lang, "title": cfg["title"],
                "source": f"https://{lang}.wikipedia.org/api/rest_v1/page/html/",
                "chars": len(markdown), "lines": markdown.count("\n"),
                "normalization": "NFKC",
            }, f, ensure_ascii=False, indent=2)
        print(f"  wrote {md_path} ({len(markdown):,} chars)")


if __name__ == "__main__":
    main()
