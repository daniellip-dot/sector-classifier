"""Mode C: refine the taxonomy when unclassified rate climbs or on schedule.
Callable from classify.py (auto) or the CLI (manual)."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import anthropic

from lib import llm, taxonomy as taxonomy_mod
from lib.concurrency import Database, TokenBucket
from modes._common import CostTracker, classify_company_row


log = logging.getLogger(__name__)

RECLASSIFY_WORKERS = 5
CONTEXT_SAMPLE_N = 200


def _log_changes(change_log: Optional[List[Dict[str, Any]]]) -> None:
    if not change_log:
        print("Changes: (none reported)")
        return
    print("Changes:")
    for c in change_log:
        t = c.get("type", "?")
        reason = c.get("reason", "")
        if t == "added_sub_sector":
            print("  + Added sub-sector '{}' to sector '{}' — {}".format(
                c.get("sub_sector"), c.get("sector"), reason))
        elif t == "renamed_sub_sector":
            print("  ~ Renamed '{}' → '{}' in sector '{}' — {}".format(
                c.get("from"), c.get("to"), c.get("sector"), reason))
        elif t == "merged_sub_sectors":
            print("  ⋈ Merged {} → '{}' in sector '{}' — {}".format(
                c.get("merged"), c.get("into"), c.get("sector"), reason))
        elif t == "new_sector":
            print("  ★ New sector '{}' (replaces '{}') — {}".format(
                c.get("name"), c.get("replaces"), reason))
        else:
            print("  - {}: {}".format(t, c))


def run(
    db: Database,
    client: anthropic.Anthropic,
    config: Dict[str, Any],
    trigger: str,
    cost: CostTracker,
    claude_limiter: TokenBucket,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:

    old_version = db.get_latest_version_id()
    if old_version is None:
        return {"new_version_id": None, "reason": "no current taxonomy in DB"}

    try:
        current = taxonomy_mod.load_taxonomy(config["TAXONOMY_PATH"])
    except FileNotFoundError as e:
        return {"new_version_id": None, "reason": str(e)}

    unclassified = db.get_unclassified_for_current_version(old_version)
    if not unclassified and trigger == "manual":
        print("No UNCLASSIFIED companies at the current version — nothing to refresh against.")
        return {"new_version_id": None, "reason": "no unclassified companies"}

    classified_sample = db.get_random_classified_for_version(old_version, CONTEXT_SAMPLE_N)

    # Total "new companies classified since taxonomy was built" — we approximate as all rows
    # at this taxonomy version_id (unclassified + classified total).
    n_total = len(unclassified) + len(classified_sample)

    print("Refresh training data: {} unclassified, {} classified context.".format(
        len(unclassified), len(classified_sample)
    ))

    new_tax = llm.cluster_taxonomy_refresh(
        client=client,
        model=config["SONNET_MODEL"],
        current_taxonomy=current,
        version_id=old_version,
        unclassified=unclassified,
        classified_sample=classified_sample,
        n_total_since_taxonomy=n_total,
        claude_rate_limiter=claude_limiter,
    )
    cost.inc_sonnet()
    if not new_tax:
        return {"new_version_id": None, "reason": "Sonnet refresh returned nothing"}

    ok, errors = taxonomy_mod.validate_taxonomy(new_tax)
    if not ok:
        print("New taxonomy failed validation:")
        for e in errors:
            print("  - {}".format(e))
        if not force:
            return {
                "new_version_id": None,
                "reason": "validation failed (use --force to accept anyway)",
            }

    # Sector-name safety check
    old_names = set(taxonomy_mod.sector_names(current))
    new_names = set(taxonomy_mod.sector_names(new_tax))
    renamed = old_names - new_names
    added = new_names - old_names
    if renamed and not force:
        print("\nSAFETY ABORT: the following top-level sectors would be renamed/removed:")
        for r in sorted(renamed):
            print("  - {}".format(r))
        print("New sectors that would appear:")
        for a in sorted(added):
            print("  + {}".format(a))
        print("This would break historical classifications.")
        print("Pass --force to accept this change, or adjust prompts/training data and retry.")
        return {"new_version_id": None, "reason": "sector rename blocked"}

    change_log = new_tax.get("change_log")
    if dry_run:
        print("DRY RUN — would create taxonomy version. No changes written.")
        _log_changes(change_log)
        return {"new_version_id": None, "reason": "dry-run"}

    new_version_id = taxonomy_mod.save_taxonomy(
        db=db,
        tax=new_tax,
        taxonomy_path=config["TAXONOMY_PATH"],
        history_dir=config["TAXONOMY_HISTORY_DIR"],
        trigger=trigger,
        change_log=change_log,
        company_count=n_total,
        notes="refresh from v{}".format(old_version),
    )

    print("\n=== TAXONOMY REFRESHED (v{} → v{}) ===".format(old_version, new_version_id))
    print("Trigger: {}".format(trigger))
    print("Unclassified companies fed to Sonnet: {}".format(len(unclassified)))
    _log_changes(change_log)

    # Clear classification fields for previously-UNCLASSIFIED rows so they re-process.
    cleared = db.clear_unclassified_for_reclassify(old_version)
    print("\nRe-classifying {} previously-unclassified companies against new taxonomy...".format(cleared))

    taxonomy_text = taxonomy_mod.format_for_prompt(new_tax)

    # Pull cleared rows and re-classify concurrently
    with db.lock:
        cur = db.conn.execute(
            "SELECT * FROM company_research "
            "WHERE sector IS NULL AND taxonomy_version_id IS NULL "
            "  AND services_description IS NOT NULL"
        )
        cleared_rows = [dict(r) for r in cur.fetchall()]

    def reclassify_one(row: Dict[str, Any]) -> str:
        patch = classify_company_row(
            row, client, config["HAIKU_MODEL"],
            new_tax, taxonomy_text, new_version_id,
            claude_limiter, cost,
        )
        for k in ("domain", "domain_confidence", "domain_method",
                  "website_raw_text_length", "services_description",
                  "specialisms", "sampled_for_discovery"):
            patch.setdefault(k, row.get(k))
        db.upsert_company(patch)
        return patch.get("sector") or "UNCLASSIFIED"

    now_classified = 0
    now_unclassified = 0
    with ThreadPoolExecutor(max_workers=RECLASSIFY_WORKERS) as ex:
        futures = [ex.submit(reclassify_one, r) for r in cleared_rows]
        for f in as_completed(futures):
            try:
                sector = f.result()
                if sector == "UNCLASSIFIED":
                    now_unclassified += 1
                else:
                    now_classified += 1
            except Exception as e:
                log.warning("reclassify worker error: %s", e)
                now_unclassified += 1

    print("Result: {}/{} now classified. {} remain UNCLASSIFIED.".format(
        now_classified, len(cleared_rows), now_unclassified
    ))
    print("Resuming main classification run.\n")

    return {
        "new_version_id": new_version_id,
        "taxonomy": new_tax,
        "change_log": change_log,
        "reclassified_n": now_classified,
        "still_unclassified_n": now_unclassified,
    }


def run_manual(config: Dict[str, Any], force: bool, dry_run: bool) -> int:
    """Entry point for the CLI `refresh-taxonomy` command."""
    db = Database(config["DB_PATH"], config["SCHEMA_PATH"])
    client = anthropic.Anthropic(api_key=config["ANTHROPIC_API_KEY"])
    claude_limiter = TokenBucket(rate_per_sec=50 / 60.0, capacity=50)
    cost = CostTracker()

    try:
        result = run(
            db=db, client=client, config=config,
            trigger="manual", cost=cost, claude_limiter=claude_limiter,
            force=force, dry_run=dry_run,
        )
    finally:
        print("\n=== REFRESH COST ===")
        for line in cost.summary_lines():
            print(line)
        client.close()
        db.close()

    return 0 if result.get("new_version_id") or dry_run else 2
