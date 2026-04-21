"""Find a company's domain: free HEAD-guess first, then a Serper search cascade."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests


DIRECTORY_DOMAINS = {
    "linkedin", "facebook", "companieshouse", "find-and-update", "gov.uk",
    "endole", "duedil", "opencorporates", "dnb.com", "bloomberg", "zoominfo",
    "yell.com", "192.com", "trustpilot", "glassdoor", "indeed",
    "companieslist", "bizstats", "pappers", "northdata", "creditsafe",
    "companycheck", "bizdb", "solocheck", "yahoo.com", "wikipedia",
    "companies-house", "companiesintheuk", "hithorizons", "companieshub",
    "ukcom.biz", "google.com", "bing.com",
}

SLUG_STRIP_WORDS = [
    "HOLDINGS", "LIMITED", "GROUP", "LTD", "PLC", "LLP",
]

TOKEN_STOP_WORDS = {
    "ltd", "limited", "plc", "llp", "group", "the", "and",
    "services", "company", "holdings",
}

HEAD_TIMEOUT = 5
TLD_ORDER = ["co.uk", "com", "uk", "org.uk"]

USER_AGENT = "Mozilla/5.0 (compatible; sector-classifier research bot)"


def slugify_for_guess(company_name: str) -> str:
    """Lowercase, strip LTD/LIMITED/PLC/LLP/GROUP/HOLDINGS, drop non-alphanumeric."""
    name = company_name.upper()
    for word in SLUG_STRIP_WORDS:
        name = re.sub(r"\b" + word + r"\b", " ", name)
    name = name.lower()
    return re.sub(r"[^a-z0-9]", "", name)


def clean_name_for_search(company_name: str) -> str:
    name = company_name.upper()
    for word in SLUG_STRIP_WORDS:
        name = re.sub(r"\b" + word + r"\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def name_tokens(company_name: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", company_name.lower())
    return [w for w in words if len(w) >= 3 and w not in TOKEN_STOP_WORDS]


def postcode_sector(postcode: Optional[str]) -> Optional[str]:
    if not postcode:
        return None
    pc = postcode.strip().upper()
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d)", pc)
    if not m:
        return None
    return "{} {}".format(m.group(1), m.group(2))


def _head_ok(url: str) -> bool:
    try:
        r = requests.head(
            url, timeout=HEAD_TIMEOUT, allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        return 200 <= r.status_code < 400
    except requests.RequestException:
        return False


def guess_domain(company_name: str) -> Optional[Dict[str, str]]:
    """Try {slug}.co.uk/.com/.uk/.org.uk concurrently. Return first hit in
    preference order, or None."""
    slug = slugify_for_guess(company_name)
    if not slug or len(slug) < 3:
        return None

    candidates = ["{}.{}".format(slug, tld) for tld in TLD_ORDER]
    urls = ["https://" + c for c in candidates]
    results: Dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=len(urls)) as ex:
        futures = {ex.submit(_head_ok, u): u for u in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                results[url] = fut.result()
            except Exception:
                results[url] = False

    for url in urls:
        if results.get(url):
            domain = url.replace("https://", "")
            return {"domain": domain, "confidence": "HIGH", "method": "guess"}
    return None


def bare_domain(url_or_host: str) -> str:
    """Extract the left-most label of the hostname (before first dot), lowercased."""
    s = url_or_host.strip()
    if "://" not in s:
        s = "http://" + s
    host = urlparse(s).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(".")[0] if host else ""


def is_directory(url: str) -> bool:
    host = (urlparse(url if "://" in url else "http://" + url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for d in DIRECTORY_DOMAINS:
        if d in host:
            return True
    return False


def score_domain(url: str, tokens: List[str]) -> int:
    bare = bare_domain(url)
    if not bare:
        return 0
    return sum(1 for t in tokens if t in bare)


def _confidence_from_score(score: int) -> Optional[str]:
    if score >= 6:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return None


def serper_search(query: str, api_key: str, rate_limiter=None) -> List[Dict]:
    if rate_limiter is not None:
        rate_limiter.acquire()
    r = requests.post(
        "https://google.serper.dev/search",
        json={"q": query, "gl": "gb", "hl": "en", "num": 10},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=15,
    )
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("organic", []) or []


def find_via_serper(
    company_name: str,
    postcode: Optional[str],
    api_key: str,
    rate_limiter=None,
) -> Optional[Dict[str, str]]:
    clean = clean_name_for_search(company_name)
    pc = postcode_sector(postcode)
    tokens = name_tokens(company_name)
    if not tokens:
        return None

    queries: List[str] = []
    if pc:
        queries.append('"{}" {}'.format(clean, pc))
    queries.append('"{}"'.format(clean))
    if pc:
        queries.append("{} {}".format(clean, pc))

    tried_urls = set()
    for q in queries:
        results = serper_search(q, api_key, rate_limiter=rate_limiter)
        for r in results:
            link = r.get("link", "")
            if not link or link in tried_urls:
                continue
            tried_urls.add(link)
            if is_directory(link):
                continue
            score = score_domain(link, tokens)
            conf = _confidence_from_score(score)
            if conf is None:
                continue
            host = (urlparse(link).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]
            if not host:
                continue
            return {"domain": host, "confidence": conf, "method": "serper"}
    return None


def find_domain(
    company_name: str,
    postcode: Optional[str],
    serper_api_key: str,
    serper_rate_limiter=None,
) -> Optional[Dict[str, str]]:
    """Full cascade: domain guess first, Serper second. Returns None if nothing sticks."""
    guess = guess_domain(company_name)
    if guess:
        return guess
    if not serper_api_key:
        return None
    return find_via_serper(
        company_name, postcode, serper_api_key, rate_limiter=serper_rate_limiter,
    )
