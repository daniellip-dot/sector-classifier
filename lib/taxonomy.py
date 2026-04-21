"""Load, validate, and save the sector taxonomy."""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


BANNED_WORDS = [
    "other",
    "misc",
    "miscellaneous",
    "general services",
    "business services",
    "professional services",
    "support services",
    "other support",
]


def load_taxonomy(path: str) -> Dict[str, Any]:
    """Load taxonomy.json. Raises FileNotFoundError with a clear message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "taxonomy.json not found at {}. Run `discover` mode first.".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_taxonomy(tax: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (is_valid, list_of_errors)."""
    errors: List[str] = []

    sectors = tax.get("sectors")
    if not isinstance(sectors, list):
        return False, ["'sectors' key missing or not a list"]

    if len(sectors) != 20:
        errors.append("expected exactly 20 sectors, got {}".format(len(sectors)))

    seen_sector_names = set()
    for i, sector in enumerate(sectors):
        if not isinstance(sector, dict):
            errors.append("sector[{}] is not a dict".format(i))
            continue
        name = sector.get("name")
        if not name or not isinstance(name, str):
            errors.append("sector[{}] has no name".format(i))
            continue

        lower = name.lower()
        for banned in BANNED_WORDS:
            if banned in lower:
                errors.append(
                    "sector '{}' contains banned word '{}'".format(name, banned)
                )

        if name in seen_sector_names:
            errors.append("duplicate sector name '{}'".format(name))
        seen_sector_names.add(name)

        sub_sectors = sector.get("sub_sectors", [])
        if not isinstance(sub_sectors, list):
            errors.append("sector '{}' sub_sectors is not a list".format(name))
            continue
        if len(sub_sectors) > 20:
            errors.append(
                "sector '{}' has {} sub-sectors (max 20)".format(name, len(sub_sectors))
            )

        seen_sub_names = set()
        for j, sub in enumerate(sub_sectors):
            if not isinstance(sub, dict):
                errors.append("sector '{}' sub_sector[{}] not a dict".format(name, j))
                continue
            sub_name = sub.get("name")
            if not sub_name:
                errors.append("sector '{}' sub_sector[{}] has no name".format(name, j))
                continue
            if sub_name in seen_sub_names:
                errors.append(
                    "duplicate sub-sector '{}' in sector '{}'".format(sub_name, name)
                )
            seen_sub_names.add(sub_name)

    return (len(errors) == 0), errors


def save_taxonomy(
    db,
    tax: Dict[str, Any],
    taxonomy_path: str,
    history_dir: str,
    trigger: str,
    change_log: Optional[Any] = None,
    company_count: int = 0,
    notes: str = "",
) -> int:
    """Write taxonomy.json, snapshot to history, insert into taxonomy_versions.

    Returns the new version_id.
    """
    os.makedirs(os.path.dirname(taxonomy_path), exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    tmp_path = taxonomy_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tax, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, taxonomy_path)

    version_id = db.insert_taxonomy_version(
        taxonomy_json=json.dumps(tax, ensure_ascii=False),
        company_count_used=company_count,
        refresh_trigger=trigger,
        change_log=json.dumps(change_log) if change_log is not None else None,
        notes=notes,
    )

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot = os.path.join(
        history_dir, "taxonomy_v{}_{}.json".format(version_id, stamp)
    )
    shutil.copy2(taxonomy_path, snapshot)

    return version_id


def format_for_prompt(tax: Dict[str, Any]) -> str:
    """Pretty-print taxonomy for inclusion in an LLM prompt."""
    lines: List[str] = []
    for sector in tax.get("sectors", []):
        lines.append("SECTOR: {}".format(sector.get("name", "")))
        desc = sector.get("description", "")
        if desc:
            lines.append("  ({})".format(desc))
        for sub in sector.get("sub_sectors", []):
            sub_name = sub.get("name", "")
            sub_desc = sub.get("description", "")
            if sub_desc:
                lines.append("  - {} — {}".format(sub_name, sub_desc))
            else:
                lines.append("  - {}".format(sub_name))
        lines.append("")
    return "\n".join(lines)


def sector_names(tax: Dict[str, Any]) -> List[str]:
    """Return ordered list of top-level sector names."""
    return [s.get("name", "") for s in tax.get("sectors", [])]


def sub_sectors_for(tax: Dict[str, Any], sector_name: str) -> List[str]:
    """Return sub-sector names under a given sector (empty list if sector not found)."""
    for s in tax.get("sectors", []):
        if s.get("name") == sector_name:
            return [sub.get("name", "") for sub in s.get("sub_sectors", [])]
    return []
