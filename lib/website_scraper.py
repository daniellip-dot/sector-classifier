"""Scrape a small number of pages from a company's website and return clean text."""

from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup


PATHS = ["/", "/about", "/about-us", "/services", "/what-we-do", "/products", "/our-story"]
PAGE_TIMEOUT = 5
MIN_GOOD_CHARS = 500
MAX_GOOD_PAGES = 3
MAX_TOTAL_CHARS = 3000
USER_AGENT = "Mozilla/5.0 (compatible; sector-classifier research bot)"


def _fetch(url: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            timeout=PAGE_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.text


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def scrape(domain: str) -> Optional[Dict[str, object]]:
    """Fetch up to MAX_GOOD_PAGES pages from a domain. Return:
        {"text": <combined, truncated>, "length": <raw length before truncation>}
    or None if nothing usable came back."""
    if not domain:
        return None
    base = "https://" + domain.lstrip("/")

    chunks = []
    good = 0
    for path in PATHS:
        url = base.rstrip("/") + path
        html = _fetch(url)
        if not html:
            continue
        text = _extract_text(html)
        if len(text) >= MIN_GOOD_CHARS:
            chunks.append(text)
            good += 1
            if good >= MAX_GOOD_PAGES:
                break

    if not chunks:
        return None

    combined = "\n\n".join(chunks)
    raw_len = len(combined)
    return {
        "text": combined[:MAX_TOTAL_CHARS],
        "length": raw_len,
    }
