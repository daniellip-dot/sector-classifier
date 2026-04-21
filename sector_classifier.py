#!/usr/bin/env python3
"""sector-classifier CLI.

Subcommands:
  discover          Build initial taxonomy from a sample of the input.
  classify          Classify the full input against the current taxonomy.
  refresh-taxonomy  Manually refine the current taxonomy.
  status            Show DB counts, taxonomy version, top sectors.
  taxonomy-history  List all taxonomy versions.
  export            Dump the DB to a CSV.
  retry-failed      Re-run any row with an error_log.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import traceback
from typing import Any, Dict

from dotenv import load_dotenv


def _install_debug_signal_handler() -> None:
    """SIGUSR1 → dump all thread stacks to stderr. Useful for diagnosing hangs."""
    def _handler(sig, frame):
        lines = ["\n=== SIGUSR1 THREAD DUMP ===\n"]
        for tid, f in sys._current_frames().items():
            name = "?"
            for t in threading.enumerate():
                if t.ident == tid:
                    name = "{} (daemon={})".format(t.name, t.daemon)
                    break
            lines.append("--- Thread {} [{}] ---\n".format(tid, name))
            lines.append("".join(traceback.format_stack(f)))
        sys.stderr.write("".join(lines))
        sys.stderr.flush()
    signal.signal(signal.SIGUSR1, _handler)


HERE = os.path.abspath(os.path.dirname(__file__))
SCHEMA_PATH = os.path.join(HERE, "schema.sql")
TAXONOMY_HISTORY_DIR = os.path.join(HERE, "taxonomy", "taxonomy_history")


def load_config() -> Dict[str, Any]:
    load_dotenv(os.path.join(HERE, ".env"))
    cfg = {
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "DB_PATH": os.getenv("DB_PATH", os.path.join(HERE, "data", "classifier.db")),
        "TAXONOMY_PATH": os.getenv("TAXONOMY_PATH", os.path.join(HERE, "taxonomy", "taxonomy.json")),
        "WORKERS": int(os.getenv("WORKERS", "5")),
        "REFRESH_UNCLASSIFIED_THRESHOLD": float(os.getenv("REFRESH_UNCLASSIFIED_THRESHOLD", "0.05")),
        "REFRESH_WINDOW": int(os.getenv("REFRESH_WINDOW", "1000")),
        "REFRESH_EVERY": int(os.getenv("REFRESH_EVERY", "10000")),
        "HAIKU_MODEL": os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001"),
        "SONNET_MODEL": os.getenv("SONNET_MODEL", "claude-sonnet-4-6"),
        "SCHEMA_PATH": SCHEMA_PATH,
        "TAXONOMY_HISTORY_DIR": TAXONOMY_HISTORY_DIR,
    }
    return cfg


def _require(cfg: Dict[str, Any], keys: list) -> None:
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        print("ERROR: missing required config: {}".format(", ".join(missing)))
        print("Edit .env and fill in the missing values.")
        sys.exit(2)


def cmd_discover(args, cfg) -> int:
    _require(cfg, ["ANTHROPIC_API_KEY", "SERPER_API_KEY"])
    from modes import discover
    return discover.run(
        input_path=args.input,
        sample_size=args.sample_size,
        workers=args.workers or cfg["WORKERS"],
        config=cfg,
    )


def cmd_classify(args, cfg) -> int:
    _require(cfg, ["ANTHROPIC_API_KEY", "SERPER_API_KEY"])
    from modes import classify
    return classify.run(
        input_path=args.input,
        output_path=args.output,
        workers=args.workers or cfg["WORKERS"],
        limit=args.limit,
        config=cfg,
    )


def cmd_refresh(args, cfg) -> int:
    _require(cfg, ["ANTHROPIC_API_KEY"])
    from modes import refresh
    return refresh.run_manual(cfg, force=args.force, dry_run=args.dry_run)


def cmd_status(args, cfg) -> int:
    from lib.concurrency import Database
    from lib import taxonomy as taxonomy_mod
    db = Database(cfg["DB_PATH"], cfg["SCHEMA_PATH"])
    try:
        counts = db.summary_counts()
        total = counts.get("total", 0) or 0
        print("=== STATUS ===")
        print("Total rows in DB:      {}".format(total))
        print("  domain_found:        {}".format(counts.get("domain_found", 0)))
        print("  services_extracted:  {}".format(counts.get("services_extracted", 0)))
        print("  classified:          {}".format(counts.get("classified", 0)))
        print("  unclassified:        {}".format(counts.get("unclassified", 0)))
        print("  errored:             {}".format(counts.get("errored", 0)))

        version_id = db.get_latest_version_id()
        if version_id is None:
            print("\nTaxonomy: none yet (run `discover`)")
        else:
            hist = db.get_taxonomy_history()
            latest = hist[-1] if hist else {}
            print("\nCurrent taxonomy: v{}  created {}  trigger={}".format(
                version_id, latest.get("created_at", "?"), latest.get("refresh_trigger", "?"),
            ))

        if os.path.exists(cfg["TAXONOMY_PATH"]):
            try:
                tax = taxonomy_mod.load_taxonomy(cfg["TAXONOMY_PATH"])
                print("Sectors in taxonomy.json: {}".format(len(tax.get("sectors", []))))
            except Exception as e:
                print("Could not read taxonomy.json: {}".format(e))

        sectors = db.sector_counts(limit=20)
        if sectors:
            print("\nTop 20 sectors:")
            for sector, n in sectors:
                print("  {:40s} {}".format(sector or "(null)", n))

        if total:
            unclassified_pct = 100.0 * (counts.get("unclassified", 0) or 0) / total
            print("\nUnclassified: {} ({:.1f}%)".format(counts.get("unclassified", 0), unclassified_pct))
    finally:
        db.close()
    return 0


def cmd_history(args, cfg) -> int:
    from lib.concurrency import Database
    db = Database(cfg["DB_PATH"], cfg["SCHEMA_PATH"])
    try:
        hist = db.get_taxonomy_history()
        if not hist:
            print("No taxonomy versions yet.")
            return 0
        print("{:>4s}  {:<20s}  {:<22s}  {:>8s}  {}".format(
            "ver", "created_at", "trigger", "count", "notes"
        ))
        for h in hist:
            print("{:>4d}  {:<20s}  {:<22s}  {:>8d}  {}".format(
                h["version_id"],
                str(h["created_at"])[:19],
                h["refresh_trigger"] or "",
                h["company_count_used"] or 0,
                h.get("notes") or "",
            ))
    finally:
        db.close()
    return 0


def cmd_export(args, cfg) -> int:
    from modes.classify import _export_csv
    from lib.concurrency import Database
    db = Database(cfg["DB_PATH"], cfg["SCHEMA_PATH"])
    try:
        n = _export_csv(db, args.output)
        print("Exported {} rows to {}".format(n, args.output))
    finally:
        db.close()
    return 0


def cmd_retry_failed(args, cfg) -> int:
    _require(cfg, ["ANTHROPIC_API_KEY", "SERPER_API_KEY"])
    from lib.concurrency import Database, TokenBucket
    from modes._common import CostTracker, run_discover_for_company, classify_company_row
    from lib import llm, taxonomy as taxonomy_mod
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    db = Database(cfg["DB_PATH"], cfg["SCHEMA_PATH"])
    client = llm.make_client(cfg["ANTHROPIC_API_KEY"])
    serper_limiter = TokenBucket(rate_per_sec=30, capacity=30)
    claude_limiter = TokenBucket(rate_per_sec=50 / 60.0, capacity=50)
    cost = CostTracker()

    errored = db.get_errored()
    if not errored:
        print("No errored rows.")
        db.close()
        return 0
    print("Retrying {} errored rows...".format(len(errored)))

    try:
        tax = taxonomy_mod.load_taxonomy(cfg["TAXONOMY_PATH"])
        tax_version = db.get_latest_version_id()
        tax_text = taxonomy_mod.format_for_prompt(tax)
        has_taxonomy = True
    except FileNotFoundError:
        tax, tax_version, tax_text = None, None, None
        has_taxonomy = False

    def worker(row):
        # Clear the old error first so we don't keep stale text on success
        db.clear_error(row["company_number"])
        patch = run_discover_for_company(
            {"company_number": row["company_number"], "company_name": row["company_name"], "postcode": row.get("postcode")},
            client, cfg["HAIKU_MODEL"], cfg["SERPER_API_KEY"],
            serper_limiter, claude_limiter, cost,
            existing=None,
        )
        db.upsert_company(patch)
        if has_taxonomy and patch.get("services_description") and patch["services_description"] != "INSUFFICIENT_DATA":
            patch2 = classify_company_row(
                patch, client, cfg["HAIKU_MODEL"],
                tax, tax_text, tax_version,
                claude_limiter, cost,
            )
            for k in ("domain", "domain_confidence", "domain_method",
                      "website_raw_text_length", "services_description",
                      "specialisms", "sampled_for_discovery"):
                patch2.setdefault(k, patch.get(k))
            db.upsert_company(patch2)

    with ThreadPoolExecutor(max_workers=cfg["WORKERS"]) as ex:
        futures = [ex.submit(worker, r) for r in errored]
        with tqdm(total=len(futures), desc="retry") as pbar:
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logging.warning("retry worker error: %s", e)
                    cost.inc_error()
                pbar.update(1)

    print("\n=== RETRY SUMMARY ===")
    for line in cost.summary_lines():
        print(line)
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sector-classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Build initial taxonomy")
    d.add_argument("--input", required=True)
    d.add_argument("--sample-size", type=int, default=2500)
    d.add_argument("--workers", type=int, default=None)

    c = sub.add_parser("classify", help="Classify the full input")
    c.add_argument("--input", required=True)
    c.add_argument("--output", default=None)
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--limit", type=int, default=None)

    r = sub.add_parser("refresh-taxonomy", help="Refine taxonomy against unclassified companies")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--force", action="store_true")

    sub.add_parser("status", help="Show DB + taxonomy status")
    sub.add_parser("taxonomy-history", help="List all taxonomy versions")

    e = sub.add_parser("export", help="Dump DB to CSV")
    e.add_argument("--output", required=True)

    sub.add_parser("retry-failed", help="Re-run any row with error_log set")

    return p


def main() -> int:
    _install_debug_signal_handler()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config()

    if args.cmd == "discover":
        return cmd_discover(args, cfg)
    if args.cmd == "classify":
        return cmd_classify(args, cfg)
    if args.cmd == "refresh-taxonomy":
        return cmd_refresh(args, cfg)
    if args.cmd == "status":
        return cmd_status(args, cfg)
    if args.cmd == "taxonomy-history":
        return cmd_history(args, cfg)
    if args.cmd == "export":
        return cmd_export(args, cfg)
    if args.cmd == "retry-failed":
        return cmd_retry_failed(args, cfg)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
