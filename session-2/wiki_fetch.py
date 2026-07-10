"""wiki_fetch.py — pulls a plain-text extract of a Wikipedia article."""

import unicodedata
import requests

UA = "ERA-V5-BPE-Assignment/1.0 (student project; contact: none)"


def fetch_wiki_text(lang_code: str, title: str, timeout: int = 20) -> str:
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
