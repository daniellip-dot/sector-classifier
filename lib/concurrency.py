"""Thread pool support: token-bucket rate limiters and a thread-safe SQLite wrapper."""

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    rate: tokens added per second
    capacity: max tokens in the bucket
    """

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity if capacity is not None else rate_per_sec)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            time.sleep(wait)


class Database:
    """SQLite wrapper. One connection, one write lock, thread-safe for our usage.

    Callers must use the provided methods — don't touch self.conn directly from
    worker threads without holding self.lock.
    """

    def __init__(self, db_path: str, schema_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_schema(schema_path)

    def _init_schema(self, schema_path: str) -> None:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = f.read()
        with self.lock:
            self.conn.executescript(schema)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # ---- company_research ----

    def upsert_company(self, row: Dict[str, Any]) -> None:
        """Insert or replace a company_research row. Preserves sampled_for_discovery
        if it was previously set to 1 and the new row leaves it 0/unset."""
        cols = [
            "company_number",
            "company_name",
            "postcode",
            "domain",
            "domain_confidence",
            "domain_method",
            "website_raw_text_length",
            "services_description",
            "specialisms",
            "sector",
            "sub_sector",
            "classification_confidence",
            "rationale",
            "keywords",
            "sampled_for_discovery",
            "taxonomy_version_id",
            "error_log",
        ]
        values = [row.get(c) for c in cols]
        placeholders = ",".join(["?"] * len(cols))
        set_clause = ",".join("{}=excluded.{}".format(c, c) for c in cols if c != "company_number")
        # preserve sampled_for_discovery if it was 1
        set_clause = set_clause.replace(
            "sampled_for_discovery=excluded.sampled_for_discovery",
            "sampled_for_discovery=MAX(company_research.sampled_for_discovery, excluded.sampled_for_discovery)",
        )
        sql = (
            "INSERT INTO company_research ({}) VALUES ({}) "
            "ON CONFLICT(company_number) DO UPDATE SET {}, processed_at=CURRENT_TIMESTAMP"
        ).format(",".join(cols), placeholders, set_clause)
        with self.lock:
            self.conn.execute(sql, values)
            self.conn.commit()

    def mark_sampled(self, company_numbers: Iterable[str]) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT INTO company_research (company_number, sampled_for_discovery) VALUES (?, 1) "
                "ON CONFLICT(company_number) DO UPDATE SET sampled_for_discovery=1",
                [(cn,) for cn in company_numbers],
            )
            self.conn.commit()

    def get_processed_numbers(self) -> Set[str]:
        """company_numbers that already have a row (for resume)."""
        with self.lock:
            cur = self.conn.execute(
                "SELECT company_number FROM company_research WHERE processed_at IS NOT NULL"
            )
            return {r["company_number"] for r in cur.fetchall()}

    def get_classified_or_attempted(self) -> Set[str]:
        """company_numbers whose classification has at least been attempted
        (sector is set OR error_log is set)."""
        with self.lock:
            cur = self.conn.execute(
                "SELECT company_number FROM company_research "
                "WHERE sector IS NOT NULL OR error_log IS NOT NULL"
            )
            return {r["company_number"] for r in cur.fetchall()}

    def get_sampled_services(self) -> List[Tuple[str, str, str, Optional[str]]]:
        """Return (company_number, company_name, services_description, specialisms) for
        companies used to build the taxonomy."""
        with self.lock:
            cur = self.conn.execute(
                "SELECT company_number, company_name, services_description, specialisms "
                "FROM company_research "
                "WHERE sampled_for_discovery=1 "
                "  AND services_description IS NOT NULL "
                "  AND services_description != 'INSUFFICIENT_DATA'"
            )
            return [
                (r["company_number"], r["company_name"], r["services_description"], r["specialisms"])
                for r in cur.fetchall()
            ]

    def unclassified_rate_in_window(self, window: int) -> Tuple[int, int]:
        """Return (unclassified_count, total) over the last `window` completed rows
        (completed = sector IS NOT NULL)."""
        with self.lock:
            cur = self.conn.execute(
                "SELECT sector FROM company_research "
                "WHERE sector IS NOT NULL "
                "ORDER BY processed_at DESC LIMIT ?",
                (window,),
            )
            rows = cur.fetchall()
        total = len(rows)
        unclassified = sum(1 for r in rows if r["sector"] == "UNCLASSIFIED")
        return unclassified, total

    def get_unclassified_for_current_version(self, version_id: int) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT company_number, company_name, services_description, specialisms "
                "FROM company_research "
                "WHERE sector = 'UNCLASSIFIED' AND taxonomy_version_id = ?",
                (version_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_random_classified_for_version(self, version_id: int, n: int) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT company_number, company_name, services_description, sector, sub_sector "
                "FROM company_research "
                "WHERE sector IS NOT NULL AND sector != 'UNCLASSIFIED' "
                "  AND taxonomy_version_id = ? "
                "ORDER BY RANDOM() LIMIT ?",
                (version_id, n),
            )
            return [dict(r) for r in cur.fetchall()]

    def clear_unclassified_for_reclassify(self, version_id: int) -> int:
        """Wipe classification fields on UNCLASSIFIED rows at the given version so
        they'll be re-processed against the new taxonomy. Returns count affected."""
        with self.lock:
            cur = self.conn.execute(
                "UPDATE company_research SET sector=NULL, sub_sector=NULL, "
                "  classification_confidence=NULL, rationale=NULL, keywords=NULL, "
                "  taxonomy_version_id=NULL "
                "WHERE sector='UNCLASSIFIED' AND taxonomy_version_id=?",
                (version_id,),
            )
            self.conn.commit()
            return cur.rowcount

    def get_errored(self) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT company_number, company_name, postcode, error_log "
                "FROM company_research WHERE error_log IS NOT NULL"
            )
            return [dict(r) for r in cur.fetchall()]

    def clear_error(self, company_number: str) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE company_research SET error_log=NULL WHERE company_number=?",
                (company_number,),
            )
            self.conn.commit()

    # ---- taxonomy_versions ----

    def insert_taxonomy_version(
        self,
        taxonomy_json: str,
        company_count_used: int,
        refresh_trigger: str,
        change_log: Optional[str] = None,
        notes: str = "",
    ) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO taxonomy_versions "
                "(taxonomy_json, company_count_used, refresh_trigger, change_log, notes) "
                "VALUES (?, ?, ?, ?, ?)",
                (taxonomy_json, company_count_used, refresh_trigger, change_log, notes),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_latest_version_id(self) -> Optional[int]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT version_id FROM taxonomy_versions ORDER BY version_id DESC LIMIT 1"
            )
            row = cur.fetchone()
            return row["version_id"] if row else None

    def get_taxonomy_history(self) -> List[Dict[str, Any]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT version_id, company_count_used, refresh_trigger, created_at, notes "
                "FROM taxonomy_versions ORDER BY version_id"
            )
            return [dict(r) for r in cur.fetchall()]

    # ---- status / reporting ----

    def summary_counts(self) -> Dict[str, int]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT "
                "  COUNT(*) AS total, "
                "  SUM(CASE WHEN domain IS NOT NULL THEN 1 ELSE 0 END) AS domain_found, "
                "  SUM(CASE WHEN services_description IS NOT NULL THEN 1 ELSE 0 END) AS services_extracted, "
                "  SUM(CASE WHEN sector IS NOT NULL AND sector != 'UNCLASSIFIED' THEN 1 ELSE 0 END) AS classified, "
                "  SUM(CASE WHEN sector = 'UNCLASSIFIED' THEN 1 ELSE 0 END) AS unclassified, "
                "  SUM(CASE WHEN error_log IS NOT NULL THEN 1 ELSE 0 END) AS errored "
                "FROM company_research"
            )
            row = cur.fetchone()
            return {k: (row[k] or 0) for k in row.keys()}

    def sector_counts(self, limit: int = 20) -> List[Tuple[str, int]]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT sector, COUNT(*) AS n FROM company_research "
                "WHERE sector IS NOT NULL "
                "GROUP BY sector ORDER BY n DESC LIMIT ?",
                (limit,),
            )
            return [(r["sector"], r["n"]) for r in cur.fetchall()]
