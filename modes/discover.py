"""Mode A: build the initial taxonomy from a stratified sample of the input."""

import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import anthropic
from tqdm import tqdm

from lib import llm, taxonomy as taxonomy_mod
from lib.concurrency import Database, TokenBucket
from modes._common import (
    CostTracker,
    classify_company_row,
    load_input_csv,
    run_discover_for_company,
)


log = logging.getLogger(__name__)

CLUSTER_BATCH = 100


def _postcode_area(pc: Optional[str]) -> Optional[str]:
    if not pc:
        return None
    m = re.match(r"^([A-Z]{1,2})", pc.strip().upper())
    return m.group(1) if m else None


def stratified_sample(
    rows: List[Dict[str, Any]], sample_size: int, seed: int = 42
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if sample_size >= len(rows):
        return list(rows)

    # If any row has a postcode, try to stratify by postcode area proportionally.
    areas: Dict[str, List[Dict[str, Any]]] = {}
    has_any_postcode = False
    for r in rows:
        area = _postcode_area(r.get("postcode"))
        if area:
            has_any_postcode = True
        key = area or "_NONE_"
        areas.setdefault(key, []).append(r)

    if not has_any_postcode:
        return rng.sample(rows, sample_size)

    total = len(rows)
    selected: List[Dict[str, Any]] = []
    for area, bucket in areas.items():
        quota = int(round(sample_size * (len(bucket) / total)))
        quota = min(quota, len(bucket))
        if quota > 0:
            selected.extend(rng.sample(bucket, quota))

    # Fix rounding drift
    if len(selected) < sample_size:
        remaining = [r for r in rows if r not in selected]
        top_up = rng.sample(remaining, min(sample_size - len(selected), len(remaining)))
        selected.extend(top_up)
    elif len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)

    return selected


def _resolve_sample(
    db: Database, rows: List[Dict[str, Any]], sample_size: int
) -> List[Dict[str, Any]]:
    """Re-use existing sample from DB if present, otherwise draw new sample and persist."""
    already = db.get_sampled_services()
    already_nums = set()
    with db.lock:
        cur = db.conn.execute(
            "SELECT company_number FROM company_research WHERE sampled_for_discovery=1"
        )
        already_nums = {r["company_number"] for r in cur.fetchall()}
    if already_nums:
        print(
            "Resuming: {} companies already marked as sampled in DB. "
            "Skipping re-sampling.".format(len(already_nums))
        )
        rows_by_num = {r["company_number"]: r for r in rows}
        return [rows_by_num[n] for n in already_nums if n in rows_by_num]

    sample = stratified_sample(rows, sample_size)
    db.mark_sampled([r["company_number"] for r in sample])
    return sample


def _cluster(
    client: anthropic.Anthropic,
    sonnet_model: str,
    sampled_services: List[Dict[str, str]],
    claude_limiter: TokenBucket,
    cost: CostTracker,
) -> Optional[Dict[str, Any]]:
    if not sampled_services:
        print("No services descriptions to cluster on — aborting.")
        return None

    first = sampled_services[:CLUSTER_BATCH]
    print(
        "Clustering batch 1/{} ({} descriptions) with Sonnet...".format(
            math.ceil(len(sampled_services) / CLUSTER_BATCH), len(first)
        )
    )
    tax = llm.cluster_taxonomy_initial(client, sonnet_model, first, claude_rate_limiter=claude_limiter)
    cost.inc_sonnet()
    if not tax:
        print("Initial clustering failed.")
        return None

    remaining = sampled_services[CLUSTER_BATCH:]
    batch_no = 2
    for i in range(0, len(remaining), CLUSTER_BATCH):
        batch = remaining[i : i + CLUSTER_BATCH]
        print("Refining batch {} ({} descriptions)...".format(batch_no, len(batch)))
        refined = llm.cluster_taxonomy_refine(
            client, sonnet_model, tax, batch, claude_rate_limiter=claude_limiter
        )
        cost.inc_sonnet()
        if refined:
            tax = refined
        else:
            print("  (refinement returned nothing — keeping previous taxonomy)")
        batch_no += 1
    return tax


def run(
    input_path: str,
    sample_size: int,
    workers: int,
    config: Dict[str, Any],
) -> int:
    t0 = time.time()

    db = Database(config["DB_PATH"], config["SCHEMA_PATH"])
    client = llm.make_client(config["ANTHROPIC_API_KEY"])
    serper_limiter = TokenBucket(rate_per_sec=30, capacity=30)
    claude_limiter = TokenBucket(rate_per_sec=50 / 60.0, capacity=50)
    cost = CostTracker()
    exit_code = 0

    try:
        print("Loading input CSV: {}".format(input_path))
        rows = load_input_csv(input_path)
        print("  {} unique companies".format(len(rows)))

        sample = _resolve_sample(db, rows, sample_size)
        print("Sample size: {}".format(len(sample)))

        # Pass everything through — run_discover_for_company merges with any existing DB row
        # and skips stages already completed, so resume is automatic.
        to_do = sample
        print("Running domain → scrape → services on {} companies (workers={})...".format(
            len(to_do), workers))

        def worker(row: Dict[str, Any]) -> Dict[str, Any]:
            with db.lock:
                cur = db.conn.execute(
                    "SELECT * FROM company_research WHERE company_number=?",
                    (row["company_number"],),
                )
                existing_row = cur.fetchone()
                existing = dict(existing_row) if existing_row else None
            patch = run_discover_for_company(
                row, client, config["HAIKU_MODEL"], config["SERPER_API_KEY"],
                serper_limiter, claude_limiter, cost, existing=existing,
            )
            patch["sampled_for_discovery"] = 1
            db.upsert_company(patch)
            return patch

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker, r) for r in to_do]
            with tqdm(total=len(futures), desc="discover") as pbar:
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        log.warning("worker error: %s", e)
                        cost.inc_error()
                    pbar.update(1)

        # Pull everything we extracted and send to Sonnet
        rows_for_cluster = db.get_sampled_services()
        cluster_input = [
            {"company_number": n, "company_name": name, "services_description": sv}
            for (n, name, sv, _spec) in rows_for_cluster
        ]
        print("Descriptions eligible for clustering: {}".format(len(cluster_input)))

        tax = _cluster(client, config["SONNET_MODEL"], cluster_input, claude_limiter, cost)
        if not tax:
            print("No taxonomy produced. Aborting.")
            exit_code = 2
        else:
            ok, errors = taxonomy_mod.validate_taxonomy(tax)
            if not ok:
                print("Taxonomy validation failed:")
                for e in errors:
                    print("  - {}".format(e))
                print("Saving the taxonomy anyway for inspection, but marking as unvalidated.")

            version_id = taxonomy_mod.save_taxonomy(
                db=db,
                tax=tax,
                taxonomy_path=config["TAXONOMY_PATH"],
                history_dir=config["TAXONOMY_HISTORY_DIR"],
                trigger="initial",
                change_log=None,
                company_count=len(cluster_input),
                notes="validation_errors={}".format(len(errors)) if errors else "",
            )
            print("Saved taxonomy version {}".format(version_id))

    finally:
        elapsed = time.time() - t0
        print("\n=== DISCOVER SUMMARY ===")
        for line in cost.summary_lines():
            print(line)
        print("Elapsed: {:.1f}s".format(elapsed))
        if cost.companies_processed:
            print("Avg per company: {:.2f}s".format(elapsed / cost.companies_processed))
        client.close()
        db.close()

    return exit_code
