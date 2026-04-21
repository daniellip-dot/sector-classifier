"""Anthropic API calls with retry, JSON fence stripping, and soft-fail on persistent errors."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


log = logging.getLogger(__name__)


class _LLMError(Exception):
    """Wrapper for retryable API errors."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, (anthropic.APIStatusError,)):
        code = getattr(exc, "status_code", None)
        if code in (429, 529, 500, 502, 503, 504):
            return True
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return True
    return False


_retry = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(
        (
            anthropic.RateLimitError,
            anthropic.APIStatusError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        )
    ),
)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_json(raw: str) -> Optional[Any]:
    try:
        return json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as e:
        log.warning("JSON parse failed: %s | raw (first 500): %s", e, raw[:500])
        return None


@_retry
def _complete(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
    claude_rate_limiter=None,
) -> str:
    if claude_rate_limiter is not None:
        claude_rate_limiter.acquire()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


# ---- Haiku: services extraction ----

SERVICES_PROMPT = """You are analysing a UK company's website to extract what they do.

WEBSITE TEXT:
{website_text}

Extract a precise description of what the company does based ONLY on this text.

Rules:
- Focus on services/activities, not marketing fluff
- Specific capabilities ("fire alarm installation" not "safety solutions")
- Ignore testimonials, case studies, generic about-us content
- 2-4 sentences maximum
- If text is too generic or uninformative: output exactly "INSUFFICIENT_DATA"

Return ONLY valid JSON:
{{
  "services": "<2-4 sentences>",
  "specialisms": ["<specialism 1>", "<specialism 2>", ...]
}}
"""


def extract_services(
    client: anthropic.Anthropic,
    model: str,
    website_text: str,
    claude_rate_limiter=None,
) -> Optional[Dict[str, Any]]:
    prompt = SERVICES_PROMPT.format(website_text=website_text)
    try:
        raw = _complete(client, model, prompt, max_tokens=600, claude_rate_limiter=claude_rate_limiter)
    except Exception as e:
        log.warning("extract_services persistent failure: %s", e)
        return None
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        # Allow the literal INSUFFICIENT_DATA sentinel even outside JSON
        if "INSUFFICIENT_DATA" in raw.upper():
            return {"services": "INSUFFICIENT_DATA", "specialisms": []}
        return None
    services = parsed.get("services")
    specialisms = parsed.get("specialisms", [])
    if not isinstance(services, str):
        return None
    if not isinstance(specialisms, list):
        specialisms = []
    return {"services": services, "specialisms": specialisms}


# ---- Haiku: per-company classification ----

CLASSIFY_PROMPT = """Classify this UK company into the fixed taxonomy below.

COMPANY: {company_name}
WHAT THEY DO: {services_description}
SPECIALISMS: {specialisms}

TAXONOMY (you must choose from this list):
{taxonomy_text}

Rules:
- Choose EXACTLY ONE sector and EXACTLY ONE sub-sector from the taxonomy
- Both must be VERBATIM copies from the taxonomy — no rewording or paraphrasing
- If nothing genuinely fits, return sector="UNCLASSIFIED", sub_sector=null (but try hard first — UNCLASSIFIED should be rare)
- Confidence:
  HIGH: clear match, multiple keywords align
  MEDIUM: plausible match, some ambiguity
  LOW: thin evidence, best guess

Return ONLY valid JSON:
{{
  "sector": "<exact sector name>",
  "sub_sector": "<exact sub-sector name or null>",
  "confidence": "HIGH|MEDIUM|LOW",
  "rationale": "<one sentence>",
  "keywords": ["<kw1>", "<kw2>", "<kw3>"]
}}
"""


def classify_company(
    client: anthropic.Anthropic,
    model: str,
    company_name: str,
    services_description: str,
    specialisms: List[str],
    taxonomy_text: str,
    claude_rate_limiter=None,
) -> Optional[Dict[str, Any]]:
    prompt = CLASSIFY_PROMPT.format(
        company_name=company_name,
        services_description=services_description,
        specialisms=", ".join(specialisms) if specialisms else "(none)",
        taxonomy_text=taxonomy_text,
    )
    try:
        raw = _complete(client, model, prompt, max_tokens=400, claude_rate_limiter=claude_rate_limiter)
    except Exception as e:
        log.warning("classify_company persistent failure: %s", e)
        return None
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict):
        return None
    return parsed


# ---- Sonnet: taxonomy clustering ----

INITIAL_CLUSTER_PROMPT = """You are designing a sector taxonomy for a UK private equity deal-sourcing platform focused on owner-run SMBs. This taxonomy will be used to filter tens of thousands of UK companies for acquisition targeting.

Below are descriptions of {n} UK companies extracted from their own websites. Your job: cluster them into a useful taxonomy.

COMPANY DESCRIPTIONS:
{descriptions}

HARD RULES:
- EXACTLY 20 top-level sectors
- UP TO 20 sub-sectors per sector (fewer is better — quality over count)
- Sector names must be specific and descriptive — a PE professional should understand what kind of company fits in each sector from the name alone
- Sub-sector names must be granular enough to filter on
- NEVER use these words in sector names: "Other", "Misc", "General", "Business Services", "Professional Services", "Support Services", "Other Support Services"
- Reflect what companies ACTUALLY DO from the descriptions, not any external classification system
- Each company must fit into exactly one sector — if some don't fit, create a new sector rather than dumping them in "Other"
- Sector names use UK spelling

Return ONLY valid JSON (no markdown, no prose):
{{
  "sectors": [
    {{
      "name": "Sector name",
      "description": "One sentence definition",
      "sub_sectors": [
        {{"name": "Sub-sector name", "description": "What fits here"}}
      ]
    }}
  ]
}}
"""


REFINE_CLUSTER_PROMPT = """You previously built this sector taxonomy:
{current_taxonomy_json}

Here are {n} additional UK company descriptions not yet accounted for:
{new_descriptions}

Refine the taxonomy to accommodate these companies without breaking existing structure.

Hard rules:
- Keep EXACTLY 20 top-level sectors
- Keep UP TO 20 sub-sectors per sector
- ADD new sub-sectors where the new companies reveal gaps
- MERGE or RENAME sub-sectors if new data suggests clearer names
- Only restructure a TOP-LEVEL sector if absolutely necessary (historical classifications depend on sector names matching)
- NEVER introduce "Other" / "Misc" / "General" sectors

Return updated taxonomy in same JSON format.
"""


REFRESH_PROMPT = """You are refining an existing sector taxonomy to accommodate new data that didn't fit.

CURRENT TAXONOMY (version {version_id}):
{current_taxonomy_json}

Since this taxonomy was built, {n_total} new companies have been classified.
{n_unclassified} could not be placed into any existing sector.

UNCLASSIFIED COMPANIES (the gap — these need a home):
{unclassified_block}

RECENTLY CLASSIFIED COMPANIES (for context — already working):
{classified_block}

Refine the taxonomy so the UNCLASSIFIED companies can be placed, without breaking existing classifications.

HARD RULES:
- Keep EXACTLY 20 top-level sectors
- DO NOT rename existing top-level sectors (historical classifications match on name) — only as absolute last resort
- ADD new sub-sectors to existing sectors where UNCLASSIFIED companies reveal gaps
- Only CREATE a new sector (by removing/merging an existing one) if a clear new theme has emerged that doesn't fit anywhere
- NEVER introduce "Other" / "Misc" / "General Services" sectors

Log every change in change_log.

Return ONLY valid JSON:
{{
  "sectors": [...same structure...],
  "change_log": [
    {{"type": "added_sub_sector", "sector": "...", "sub_sector": "...", "reason": "..."}},
    {{"type": "renamed_sub_sector", "sector": "...", "from": "...", "to": "...", "reason": "..."}},
    {{"type": "merged_sub_sectors", "sector": "...", "merged": ["..."], "into": "...", "reason": "..."}},
    {{"type": "new_sector", "name": "...", "reason": "...", "replaces": "..."}}
  ]
}}
"""


def _format_descriptions(rows: List[Dict[str, str]]) -> str:
    lines = []
    for r in rows:
        lines.append("[{}]: {}".format(r.get("company_name", "?"), r.get("services_description", "")))
    return "\n".join(lines)


def cluster_taxonomy_initial(
    client: anthropic.Anthropic,
    model: str,
    rows: List[Dict[str, str]],
    claude_rate_limiter=None,
) -> Optional[Dict[str, Any]]:
    prompt = INITIAL_CLUSTER_PROMPT.format(
        n=len(rows),
        descriptions=_format_descriptions(rows),
    )
    try:
        raw = _complete(client, model, prompt, max_tokens=8000, claude_rate_limiter=claude_rate_limiter)
    except Exception as e:
        log.warning("cluster_taxonomy_initial persistent failure: %s", e)
        return None
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict) or "sectors" not in parsed:
        return None
    return parsed


def cluster_taxonomy_refine(
    client: anthropic.Anthropic,
    model: str,
    current_taxonomy: Dict[str, Any],
    rows: List[Dict[str, str]],
    claude_rate_limiter=None,
) -> Optional[Dict[str, Any]]:
    prompt = REFINE_CLUSTER_PROMPT.format(
        current_taxonomy_json=json.dumps(current_taxonomy, indent=2, ensure_ascii=False),
        n=len(rows),
        new_descriptions=_format_descriptions(rows),
    )
    try:
        raw = _complete(client, model, prompt, max_tokens=8000, claude_rate_limiter=claude_rate_limiter)
    except Exception as e:
        log.warning("cluster_taxonomy_refine persistent failure: %s", e)
        return None
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict) or "sectors" not in parsed:
        return None
    return parsed


def cluster_taxonomy_refresh(
    client: anthropic.Anthropic,
    model: str,
    current_taxonomy: Dict[str, Any],
    version_id: int,
    unclassified: List[Dict[str, Any]],
    classified_sample: List[Dict[str, Any]],
    n_total_since_taxonomy: int,
    claude_rate_limiter=None,
) -> Optional[Dict[str, Any]]:
    unclassified_block = "\n".join(
        "[{}]: {}".format(r.get("company_name", "?"), r.get("services_description") or "")
        for r in unclassified
    )
    classified_block = "\n".join(
        "[{}] ({} / {}): {}".format(
            r.get("company_name", "?"),
            r.get("sector", "?"),
            r.get("sub_sector", "?"),
            r.get("services_description") or "",
        )
        for r in classified_sample
    )
    prompt = REFRESH_PROMPT.format(
        version_id=version_id,
        current_taxonomy_json=json.dumps(current_taxonomy, indent=2, ensure_ascii=False),
        n_total=n_total_since_taxonomy,
        n_unclassified=len(unclassified),
        unclassified_block=unclassified_block,
        classified_block=classified_block,
    )
    try:
        raw = _complete(client, model, prompt, max_tokens=8000, claude_rate_limiter=claude_rate_limiter)
    except Exception as e:
        log.warning("cluster_taxonomy_refresh persistent failure: %s", e)
        return None
    parsed = _parse_json(raw)
    if not isinstance(parsed, dict) or "sectors" not in parsed:
        return None
    return parsed
