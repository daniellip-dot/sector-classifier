"""Shared helpers for discover and classify: CSV loading, the per-company pipeline,
cost tracking."""

import csv
import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from lib import domain_finder, website_scraper, llm, taxonomy as taxonomy_mod


log = logging.getLogger(__name__)


# ---- cost tracker ----

# Rough per-call costs in GBP. Sonnet/Haiku are averaged token costs for our
# typical prompt sizes; refine up as we learn more.
COST_PER_SERPER = 0.00024
COST_PER_HAIKU = 0.001
COST_PER_SONNET = 0.015


class CostTracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.serper = 0
        self.haiku = 0
        self.sonnet = 0
        self.domains_found = 0
        self.companies_processed = 0
        self.errors = 0

    def inc_serper(self) -> None:
        with self.lock:
            self.serper += 1

    def inc_haiku(self) -> None:
        with self.lock:
            self.haiku += 1

    def inc_sonnet(self) -> None:
        with self.lock:
            self.sonnet += 1

    def inc_domain(self) -> None:
        with self.lock:
            self.domains_found += 1

    def inc_company(self) -> None:
        with self.lock:
            self.companies_processed += 1

    def inc_error(self) -> None:
        with self.lock:
            self.errors += 1

    def _total_gbp_unlocked(self) -> float:
        """Compute total cost WITHOUT acquiring self.lock (caller must already hold it)."""
        return (
            self.serper * COST_PER_SERPER
            + self.haiku * COST_PER_HAIKU
            + self.sonnet * COST_PER_SONNET
        )

    def total_gbp(self) -> float:
        with self.lock:
            return self._total_gbp_unlocked()

    def summary_lines(self) -> List[str]:
        with self.lock:
            total = self._total_gbp_unlocked()
            return [
                "Companies processed:   {}".format(self.companies_processed),
                "Domains found:         {}".format(self.domains_found),
                "Errors:                {}".format(self.errors),
                "Serper queries:        {} (£{:.2f})".format(self.serper, self.serper * COST_PER_SERPER),
                "Haiku calls:           {} (£{:.2f})".format(self.haiku, self.haiku * COST_PER_HAIKU),
                "Sonnet calls:          {} (£{:.2f})".format(self.sonnet, self.sonnet * COST_PER_SONNET),
                "Total spend:           £{:.2f}".format(total),
            ]


# ---- CSV loading ----

def _fallback_company_number(name: str) -> str:
    slug = domain_finder.slugify_for_guess(name)
    if slug:
        return "slug:" + slug[:40]
    return "hash:" + hashlib.md5(name.encode("utf-8")).hexdigest()[:16]


def load_input_csv(path: str) -> List[Dict[str, Optional[str]]]:
    """Read input CSV. Returns list of dicts with keys company_number, company_name, postcode.
    company_name is required; company_number is synthesised if missing."""
    rows: List[Dict[str, Optional[str]]] = []
    seen_numbers = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "company_name" not in reader.fieldnames:
            raise ValueError("input CSV must have a 'company_name' column")
        for raw in reader:
            name = (raw.get("company_name") or "").strip()
            if not name:
                continue
            number = (raw.get("company_number") or "").strip()
            postcode = (raw.get("postcode") or "").strip() or None
            if not number:
                number = _fallback_company_number(name)
            if number in seen_numbers:
                continue
            seen_numbers.add(number)
            rows.append({
                "company_number": number,
                "company_name": name,
                "postcode": postcode,
            })
    return rows


# ---- pipeline stages ----

def run_discover_for_company(
    row: Dict[str, Optional[str]],
    anthropic_client,
    haiku_model: str,
    serper_api_key: str,
    serper_limiter,
    claude_limiter,
    cost: CostTracker,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the domain → scrape → services pipeline for one company.
    Returns a dict suitable for Database.upsert_company. `existing` is the current
    DB row if any, so already-done stages are skipped."""

    out: Dict[str, Any] = {
        "company_number": row["company_number"],
        "company_name": row["company_name"],
        "postcode": row.get("postcode"),
        "domain": existing.get("domain") if existing else None,
        "domain_confidence": existing.get("domain_confidence") if existing else None,
        "domain_method": existing.get("domain_method") if existing else None,
        "website_raw_text_length": existing.get("website_raw_text_length") if existing else None,
        "services_description": existing.get("services_description") if existing else None,
        "specialisms": existing.get("specialisms") if existing else None,
        "sampled_for_discovery": 0,
        "error_log": None,
    }

    errors: List[str] = []

    # Stage 1: domain
    if not out["domain"]:
        try:
            result = domain_finder.find_domain(
                row["company_name"],
                row.get("postcode"),
                serper_api_key,
                serper_rate_limiter=serper_limiter,
            )
            if result and result.get("method") == "serper":
                cost.inc_serper()  # at least one call was made; under-counts multi-query cascades
            if result:
                out["domain"] = result["domain"]
                out["domain_confidence"] = result["confidence"]
                out["domain_method"] = result["method"]
                cost.inc_domain()
        except Exception as e:
            errors.append("domain_find: {}".format(e))

    # Stage 2: scrape
    scraped_text: Optional[str] = None
    if out["domain"] and not out["services_description"]:
        try:
            scraped = website_scraper.scrape(out["domain"])
            if scraped:
                scraped_text = scraped["text"]
                out["website_raw_text_length"] = scraped["length"]
        except Exception as e:
            errors.append("scrape: {}".format(e))

    # Stage 3: services extraction
    if scraped_text and not out["services_description"]:
        try:
            svc = llm.extract_services(
                anthropic_client, haiku_model, scraped_text,
                claude_rate_limiter=claude_limiter,
            )
            cost.inc_haiku()
            if svc:
                out["services_description"] = svc["services"]
                out["specialisms"] = json.dumps(svc.get("specialisms", []))
            else:
                errors.append("services_extraction: None returned")
        except Exception as e:
            errors.append("services_extraction: {}".format(e))

    if errors:
        out["error_log"] = " | ".join(errors)
        cost.inc_error()
    cost.inc_company()
    return out


def classify_company_row(
    row_with_services: Dict[str, Any],
    anthropic_client,
    haiku_model: str,
    taxonomy_obj: Dict[str, Any],
    taxonomy_text: str,
    taxonomy_version_id: int,
    claude_limiter,
    cost: CostTracker,
) -> Dict[str, Any]:
    """Given a row that already has services_description populated, run classification.
    Returns a patch dict to upsert onto the existing row."""
    name = row_with_services.get("company_name") or "?"
    services = row_with_services.get("services_description") or ""
    specialisms_raw = row_with_services.get("specialisms") or "[]"
    try:
        specialisms = json.loads(specialisms_raw) if isinstance(specialisms_raw, str) else (specialisms_raw or [])
    except Exception:
        specialisms = []

    patch: Dict[str, Any] = {
        "company_number": row_with_services["company_number"],
        "company_name": name,
        "postcode": row_with_services.get("postcode"),
        "taxonomy_version_id": taxonomy_version_id,
    }

    if not services or services == "INSUFFICIENT_DATA":
        patch["sector"] = "UNCLASSIFIED"
        patch["sub_sector"] = None
        patch["classification_confidence"] = "LOW"
        patch["rationale"] = "no services data"
        patch["keywords"] = None
        return patch

    result = llm.classify_company(
        anthropic_client,
        haiku_model,
        name,
        services,
        specialisms,
        taxonomy_text,
        claude_rate_limiter=claude_limiter,
    )
    cost.inc_haiku()

    if not result:
        patch["error_log"] = "classify: LLM returned nothing"
        cost.inc_error()
        return patch

    sector = result.get("sector")
    sub = result.get("sub_sector")
    conf = result.get("confidence")
    rationale = result.get("rationale", "")
    keywords = result.get("keywords", [])

    # Validate verbatim match
    valid_sectors = set(taxonomy_mod.sector_names(taxonomy_obj))
    if sector == "UNCLASSIFIED":
        sub = None
    elif sector not in valid_sectors:
        log.warning(
            "[%s] sector '%s' not in taxonomy — marking UNCLASSIFIED",
            name, sector,
        )
        sector = "UNCLASSIFIED"
        sub = None
        conf = "LOW"
    else:
        valid_subs = set(taxonomy_mod.sub_sectors_for(taxonomy_obj, sector))
        if sub not in valid_subs:
            log.warning(
                "[%s] sub '%s' not in sector '%s' — marking UNCLASSIFIED",
                name, sub, sector,
            )
            sector = "UNCLASSIFIED"
            sub = None
            conf = "LOW"

    patch["sector"] = sector
    patch["sub_sector"] = sub
    patch["classification_confidence"] = conf
    patch["rationale"] = rationale
    patch["keywords"] = json.dumps(keywords) if isinstance(keywords, list) else str(keywords)
    return patch
