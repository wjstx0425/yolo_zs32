# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Shared helpers for C789 YOLO examples."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


FUSION_FIELDNAMES = [
    "part_id",
    "side",
    "view",
    "slot_id",
    "branch",
    "pred_label",
    "score",
    "threshold",
    "defect_type",
    "reason",
    "source_path",
    "evidence_path",
    "status",
]


def parse_overrides(values: list[str]) -> dict[str, Any]:
    """Parse repeated key=value overrides for direct Ultralytics API calls."""
    overrides: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            msg = f"--override must be key=value, got: {value}"
            raise ValueError(msg)
        key, raw = value.split("=", maxsplit=1)
        key = key.strip()
        raw = raw.strip()
        if not key:
            msg = f"--override key cannot be empty: {value}"
            raise ValueError(msg)
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            overrides[key] = lowered == "true"
            continue
        if lowered in {"none", "null"}:
            overrides[key] = None
            continue
        try:
            overrides[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            overrides[key] = float(raw)
            continue
        except ValueError:
            pass
        overrides[key] = raw
    return overrides


def _clean_text(value: Any) -> str | None:
    """Return stripped text or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def path_aliases(path_text: str | None) -> set[str]:
    """Return aliases used to join YOLO result paths with anomalib manifests."""
    strong, weak = path_alias_groups(path_text)
    return strong | weak


def path_alias_groups(path_text: str | None) -> tuple[set[str], set[str]]:
    """Return strong and weak aliases for a manifest image path."""
    if path_text is None:
        return set(), set()
    path = Path(path_text)
    strong = {path_text}
    weak = {path.name, path.stem}
    try:
        strong.add(str(path.resolve(strict=False)))
    except OSError:
        pass
    return {alias for alias in strong if alias}, {alias for alias in weak if alias}


def load_manifest_index(path: Path | None) -> dict[str, dict[str, str]]:
    """Load an anomalib part-crop or YOLO export manifest keyed by path aliases."""
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    index: dict[str, dict[str, str]] = {}
    weak_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strong_aliases: set[str] = set()
        weak_aliases: set[str] = set()
        for name in ("yolo_image_path", "processed_path", "image_path", "path", "file_path"):
            strong, weak = path_alias_groups(_clean_text(row.get(name)))
            strong_aliases.update(strong)
            weak_aliases.update(weak)
        sample_id = _clean_text(row.get("sample_id"))
        if sample_id:
            strong_aliases.add(sample_id)
        for alias in strong_aliases:
            index[alias] = row
        for alias in weak_aliases:
            weak_candidates[alias].append(row)
    for alias, candidates in weak_candidates.items():
        unique_rows = {id(row): row for row in candidates}
        if len(unique_rows) == 1 and alias not in index:
            index[alias] = next(iter(unique_rows.values()))
    return index


def infer_slot_id(path: str | Path, metadata: Mapping[str, str] | None = None) -> str:
    """Infer a slot id from metadata or a filename."""
    if metadata is not None:
        slot = _clean_text(metadata.get("slot_id")) or _clean_text(metadata.get("slot"))
        if slot:
            return slot
    match = re.search(r"(?:^|[_/\-])slot(?P<slot>[0-9]+)(?:$|[_/\-.])", str(path))
    if match is None:
        return ""
    return f"slot{int(match.group('slot')):02d}"


def metadata_for_path(index: Mapping[str, dict[str, str]], image_path: str | Path) -> dict[str, str] | None:
    """Return manifest metadata for an image path."""
    for alias in path_aliases(str(image_path)):
        row = index.get(alias)
        if row is not None:
            return row
    return None


def part_id_for_path(image_path: str | Path, metadata: Mapping[str, str] | None = None) -> str:
    """Return the best available part/sample id."""
    if metadata is not None:
        value = _clean_text(metadata.get("part_id")) or _clean_text(metadata.get("sample_id"))
        if value:
            return value
    return Path(image_path).stem


def defect_type_from_class(class_id: int | None, names: Mapping[int, str] | list[str] | None) -> str:
    """Return a defect type from YOLO class metadata."""
    if class_id is None or names is None:
        return "defect"
    if isinstance(names, Mapping):
        return str(names.get(class_id, "defect"))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return "defect"


def write_fusion_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write anomalib stage-18-compatible YOLO branch predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FUSION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FUSION_FIELDNAMES})
