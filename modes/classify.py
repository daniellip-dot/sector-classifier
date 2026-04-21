"""Mode B: classify the full input against the current taxonomy with resumable,
concurrent workers and automatic refresh triggers."""

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import anthropic
from tqdm import tqdm

from lib import taxonomy as taxonomy_mod
from lib.concurrency import Database, TokenBucket
from modes import refresh as refresh_mod
from modes._common import (
    CostTracker,
    classify_company_row,
    load_input_csv,
    run_discover_for_company,
)


log = logging.getLogger(__name__)

CHECKPOINT_EVERY = 1000


def _load_current_taxonomy(db: Database, tax_path: str):
    tax = taxonomy_mod.load_taxonomy(tax_path)
    version_id = db.get_latest_version_id()
    if version_id is None:
        raise RuntimeError(
            "No taxonomy_versions row in DB. Taxonomy file exists but is unversioned — "
            "run discover mode first."
        )
    return tax, version_id


def _worker(
    row: Dict[str, Any],
    db: Database,
    client: anthropic.Anthropic,
    config: Dict[str, Any],
    taxonomy_obj: Dict[str, Any],
    taxonomy_text: str,
    taxonomy_version_id: int,
    serper_limiter: TokenBucket,
    claude_limiter: TokenBucket,
    cost: CostTracker,
) -> None:
    # Get existing row (for resume)
    with db.lock:
        cur = db.conn.execute(
            "SELECT * FROM company_research WHERE company_number=?",
            (row["company_number"],),
        )
        existing_row = cur.fetchone()
        existing = dict(existing_row) if existing_row else None

    # Stage 1: ensure we have services_description
    if not existing or not existing.get("services_description") or existing.get("services_description") == "INSUFFICIENT_DATA":
        patch = run_discover_for_company(
            row, client, config["HAIKU_MODEL"], config["SERPER_API_KEY"],
            serper_limiter, claude_limiter, cost, existing=existing,
        )
        if existing:
            # preserve sampled flag
            patch["sampled_for_discovery"] = existing.get("sampled_for_discovery", 0) or 0
        db.upsert_company(patch)
        # Re-read post-upsert so classify stage sees latest
        with db.lock:
            cur = db.conn.execute(
                "SELECT * FROM company_research WHERE company_number=?",
                (row["company_number"],),
            )
            existing_row = cur.fetchone()
            existing = dict(existing_row) if existing_row else None
    else:
        cost.inc_company()

    if not existing:
        return

    # Stage 2: classify
    patch = classify_company_row(
        existing,
        client,
        config["HAIKU_MODEL"],
        taxonomy_obj,
        taxonomy_text,
        taxonomy_version_id,
        claude_limiter,
        cost,
    )
    # Carry over domain fields so upsert doesn't null them
    for k in ("domain", "domain_confidence", "domain_method",
              "website_raw_text_length", "services_description",
              "specialisms", "sampled_for_discovery"):
        patch.setdefault(k, existing.get(k))
    db.upsert_company(patch)


def _check_refresh_triggers(
    db: Database,
    companies_since_refresh: int,
    config: Dict[str, Any],
) -> Optional[str]:
    threshold = float(config.get("REFRESH_UNCLASSIFIED_THRESHOLD", 0.05))
    window = int(config.get("REFRESH_WINDOW", 1000))
    every = int(config.get("REFRESH_EVERY", 10000))

    if companies_since_refresh >= every:
        return "scheduled"

    unclassified, total = db.unclassified_rate_in_window(window)
    if total >= window and (unclassified / total) > threshold:
        return "unclassified_threshold"

    return None


def _export_csv(db: Database, out_path: str) -> int:
    with db.lock:
        cur = db.conn.execute(
            "SELECT company_number, company_name, postcode, domain, domain_confidence, "
            "domain_method, services_description, specialisms, sector, sub_sector, "
            "classification_confidence, rationale, keywords, taxonomy_version_id, "
            "processed_at, error_log "
            "FROM company_research ORDER BY company_name"
        )
        rows = cur.fetchall()
    if not rows:
        return 0
    cols = rows[0].keys()
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return len(rows)


def run(
    input_path: str,
    output_path: Optional[str],
    workers: int,
    limit: Optional[int],
    config: Dict[str, Any],
) -> int:
    t0 = time.time()

    db = Database(config["DB_PATH"], config["SCHEMA_PATH"])
    client = anthropic.Anthropic(api_key=config["ANTHROPIC_API_KEY"])
    serper_limiter = TokenBucket(rate_per_sec=30, capacity=30)
    claude_limiter = TokenBucket(rate_per_sec=50 / 60.0, capacity=50)
    cost = CostTracker()
    refreshes_this_run: List[str] = []
    exit_code = 0

    try:
        try:
            taxonomy_obj, taxonomy_version_id = _load_current_taxonomy(db, config["TAXONOMY_PATH"])
        except (FileNotFoundError, RuntimeError) as e:
            print("ERROR: {}".format(e))
            exit_code = 2
            return exit_code

        taxonomy_text = taxonomy_mod.format_for_prompt(taxonomy_obj)
        print("Loaded taxonomy version {}".format(taxonomy_version_id))

        print("Loading input CSV: {}".format(input_path))
        rows = load_input_csv(input_path)
        print("  {} unique companies".format(len(rows)))

        done = db.get_classified_or_attempted()
        todo = [r for r in rows if r["company_number"] not in done]
        if limit:
            todo = todo[:limit]
        print("Already done: {}. To process: {}.".format(len(done), len(todo)))
        if not todo:
            print("Nothing to do.")
            if output_path:
                n = _export_csv(db, output_path)
                print("Exported {} rows to {}".format(n, output_path))
            return exit_code

        companies_since_refresh = 0
        processed_this_run = 0

        # Process in checkpoints so we can drain workers and run refresh between chunks.
        pbar = tqdm(total=len(todo), desc="classify")
        i = 0
        while i < len(todo):
            chunk = todo[i : i + CHECKPOINT_EVERY]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        _worker, r, db, client, config,
                        taxonomy_obj, taxonomy_text, taxonomy_version_id,
                        serper_limiter, claude_limiter, cost,
                    )
                    for r in chunk
                ]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        log.warning("worker error: %s", e)
                        cost.inc_error()
                    processed_this_run += 1
                    companies_since_refresh += 1
                    pbar.update(1)
                    if processed_this_run % 100 == 0:
                        counts = db.summary_counts()
                        pbar.set_postfix(
                            domain="{:.0%}".format((counts["domain_found"] or 0) / max(counts["total"], 1)),
                            classified="{:.0%}".format((counts["classified"] or 0) / max(counts["total"], 1)),
                            unclassified="{:.0%}".format((counts["unclassified"] or 0) / max(counts["total"], 1)),
                        )

            # Checkpoint: refresh?
            trigger = _check_refresh_triggers(db, companies_since_refresh, config)
            if trigger:
                pbar.write("\nRefresh trigger: {}. Running taxonomy refresh...".format(trigger))
                result = refresh_mod.run(
                    db=db, client=client, config=config,
                    trigger=trigger, cost=cost, claude_limiter=claude_limiter,
                    force=False, dry_run=False,
                )
                if result.get("new_version_id"):
                    taxonomy_obj = result["taxonomy"]
                    taxonomy_text = taxonomy_mod.format_for_prompt(taxonomy_obj)
                    taxonomy_version_id = result["new_version_id"]
                    companies_since_refresh = 0
                    refreshes_this_run.append(trigger)
                    pbar.write("Resumed with taxonomy v{}.".format(taxonomy_version_id))
                else:
                    pbar.write("Refresh did not produce a new taxonomy: {}".format(result.get("reason")))
            i += CHECKPOINT_EVERY

        pbar.close()

        if output_path:
            n = _export_csv(db, output_path)
            print("Exported {} rows to {}".format(n, output_path))

        elapsed = time.time() - t0
        counts = db.summary_counts()
        print("\n=== CLASSIFY SUMMARY ===")
        for line in cost.summary_lines():
            print(line)
        print("Elapsed: {:.1f}s".format(elapsed))
        if cost.companies_processed:
            print("Avg per company: {:.2f}s".format(elapsed / cost.companies_processed))
        print("")
        print("DB totals: total={}, domain_found={}, services={}, classified={}, "
              "unclassified={}, errored={}".format(
                  counts["total"], counts["domain_found"], counts["services_extracted"],
                  counts["classified"], counts["unclassified"], counts["errored"],
              ))
        print("\nTop sectors:")
        for sector, n in db.sector_counts(limit=10):
            print("  {:40s} {}".format(sector, n))
        print("\nRefreshes this run: {}{}".format(
            len(refreshes_this_run),
            " (" + ", ".join(refreshes_this_run) + ")" if refreshes_this_run else "",
        ))

    finally:
        client.close()
        db.close()

    return exit_code
