# sector-classifier

Standalone tool for classifying up to 50k UK companies into a website-derived
sector taxonomy. Built for PE deal-sourcing on owner-run SMBs.

## What it does

Takes a CSV of UK companies (`company_name`, optional `company_number`, `postcode`)
and for each one:

1. Finds the company's website (free domain guess → Serper search cascade).
2. Scrapes the site and extracts what the company actually does (Claude Haiku).
3. Classifies it into a sector + sub-sector from a taxonomy that was itself
   built from a sample of the input's website content (Claude Sonnet for
   clustering, Haiku for per-company classification).

The taxonomy is never derived from SIC codes or any external system — it
reflects what these specific companies actually do.

## Modes

- **discover** — pick a stratified sample, extract services, cluster into
  20 sectors × up to 20 sub-sectors. Produces the initial `taxonomy.json`.
- **classify** — run the full input through the taxonomy. Resumable,
  concurrent, writes to SQLite after every company.
- **refresh-taxonomy** — refine the taxonomy when unclassified rate climbs
  or on a schedule. Never renames top-level sectors (would break filters).

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in SERPER_API_KEY and ANTHROPIC_API_KEY
```

## Usage

```bash
# Build initial taxonomy from a 2,500-company sample
python3 sector_classifier.py discover --input companies.csv --sample-size 2500

# Classify the full input
python3 sector_classifier.py classify --input companies.csv --output results.csv

# Manual taxonomy refresh
python3 sector_classifier.py refresh-taxonomy

# Other commands
python3 sector_classifier.py status
python3 sector_classifier.py taxonomy-history
python3 sector_classifier.py export --output results.csv
python3 sector_classifier.py retry-failed
```

## Input CSV

Required column: `company_name`.
Optional: `company_number` (used as primary key — a slugified name is
generated if missing), `postcode` (helps domain-finding and geographic
sampling).

## Costs (rough)

- Serper: ~£0.00024 per query
- Haiku: ~£0.001 per call (domain + classification)
- Sonnet: ~£0.015 per call (only during discover + refresh, ~20–50 total)

50k companies end-to-end is typically £50–80 in API spend.

## Files

- `sector_classifier.py` — CLI entry point.
- `modes/discover.py` — builds initial taxonomy.
- `modes/classify.py` — runs classification + monitors for refresh triggers.
- `modes/refresh.py` — refines taxonomy when drift detected.
- `lib/domain_finder.py` — domain guess + Serper cascade.
- `lib/website_scraper.py` — polite website scrape.
- `lib/llm.py` — Anthropic API calls with retries.
- `lib/concurrency.py` — thread pool + rate limiters + thread-safe DB.
- `lib/taxonomy.py` — load / validate / save the taxonomy.
- `schema.sql` — SQLite schema.
