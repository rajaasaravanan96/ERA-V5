"""wiki_fetch.py — pulls a plain-text extract of a Wikipedia article.

Results are cached to disk under data/wiki_<lang_code>.txt. On a free-tier
server this means /api/train never has to hit Wikipedia's API at all once
the cache files are committed to the repo — it just reads local text, which
is instant and can't time out or get rate-limited.
"""

import os
import unicodedata
import requests

UA = "ERA-V5-BPE-Assignment/1.0 (student project; contact: none)"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _cache_path(lang_code: str) -> str:
    return os.path.join(CACHE_DIR, f"wiki_{lang_code}.txt")


def fetch_wiki_text(lang_code: str, title: str, timeout: int = 20, use_cache: bool = True) -> str:
    cache_path = _cache_path(lang_code)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()

    text = _fetch_wiki_text_live(lang_code, title, timeout)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text


def _fetch_wiki_text_live(lang_code: str, title: str, timeout: int = 20) -> str:
    url = f"https://{lang_code}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": UA})
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError("unexpected response from Wikipedia")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"page not found for title '{title}' ({lang_code}.wikipedia.org)")
    text = page.get("extract", "") or ""
    if len(text) < 200:
        raise ValueError("extract too short — try a different article title")
    return unicodedata.normalize("NFC", text)
