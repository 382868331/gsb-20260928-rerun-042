"""Compare two frozen multi-region relation reviews without editing either.

A frozen review is the ``multi_region_report.json`` written by
:func:`medreg.relations_batch.save_relations`. This module is purely
read-only with respect to the two reviewed versions: it never reloads a
session, never recomputes masks and never changes a verdict. It pairs
evaluations by their *stable relation identity*
``(region_a, region_b, relation, timepoint)`` and explains why paired
conclusions differ:

- ``manual_correction``: the set of manual corrections of a region
  involved in the pair changed, or a region contributing to the pair
  switched between manual and automatic/source provenance;
- ``candidate_switch``: the adopted registration candidate, selection
  method or transform parameters behind either region changed;
- ``declaration_update``: the pair itself appears in only one of the
  two reviews (a declared relation was added or withdrawn).

The export keeps both sides' timepoint, sources and beyond-voxel basis
(offending voxel counts plus each region's full provenance), together
with content fingerprints of the input reports, so reopening the two
frozen directories reproduces the exact same comparison.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch import _numpy_clean

REPORT_FILENAME = "multi_region_report.json"
COMPARE_FILENAME = "relation_review_comparison.json"
SCHEMA = "medreg-multiregion-comparison-v1"

#: Fields kept verbatim from each side for every paired evaluation,
#: so both timepoints, sources and beyond-voxel evidence stay visible.
_SNAPSHOT_FIELDS = (
    "label",
    "status",
    "actual",
    "offending_voxels",
    "manual_origin",
    "reasons",
    "basis",
)


def relation_identity(evaluation: dict[str, Any]) -> tuple[str, str, str, str]:
    """Stable identity of one declared-relation evaluation.

    ``(region_a, region_b, relation, label)``; the order of regions is
    taken from the declaration (only ``contains`` is ordered).
    """

    regions = evaluation["regions"]
    return (regions[0], regions[1], evaluation["relation"], evaluation["label"])


def format_identity(identity: tuple[str, str, str, str]) -> str:
    return "|".join(identity)


def read_frozen_review(version_dir: str | Path) -> dict[str, Any]:
    """Load a frozen ``multi_region_report.json`` and validate its schema."""

    path = Path(version_dir) / REPORT_FILENAME
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "medreg-multiregion-v1":
        raise ValueError(
            f"{path}: unsupported multi-region schema {report.get('schema')!r}"
        )
    return report


def _file_fingerprint(version_dir: str | Path) -> dict[str, Any]:
    root = Path(version_dir)
    path = root / REPORT_FILENAME
    raw = path.read_bytes()
    files = [path]
    propagation_dir = root / "regions"
    if propagation_dir.is_dir():
        files.extend(
            sorted(propagation_dir.glob("*/propagation_report.json"))
        )
    file_hashes = {}
    for file_path in files:
        file_hashes[str(file_path.relative_to(root))] = hashlib.sha256(
            file_path.read_bytes()
        ).hexdigest()
    return {
        "path": str(root),
        "report_file": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "files": file_hashes,
    }


def _evaluation_map(
    report: dict[str, Any],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {relation_identity(record): record for record in report["evaluations"]}


def _correction_lookup(report: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """region -> label -> correction record for the frozen review."""

    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for region_name, region_record in report.get("regions", {}).items():
        lookup[region_name] = dict(region_record.get("corrections", {}))
    return lookup


def _snapshot(evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    snapshot = {field: evaluation.get(field) for field in _SNAPSHOT_FIELDS}
    regions = evaluation["regions"]
    label = evaluation["label"]
    snapshot["sources"] = [
        {
            "region": regions[index],
            "label": label,
            "kind": evaluation["basis"][regions[index]]["kind"],
            "basis_label": evaluation["basis"][regions[index]]["basis_label"],
            "candidate_id": evaluation["basis"][regions[index]]["candidate_id"],
            "selection_method": evaluation["basis"][regions[index]][
                "selection_method"
            ],
        }
        for index in (0, 1)
    ]
    return snapshot


def _conclusion(evaluation: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        evaluation["status"],
        evaluation["actual"],
        evaluation["offending_voxels"],
    )


def _basis_region(evaluation: dict[str, Any], region: str) -> dict[str, Any]:
    return evaluation["basis"][region]


def _classify_origin(
    left: dict[str, Any],
    right: dict[str, Any],
    left_corrections: dict[str, dict[str, dict[str, Any]]],
    right_corrections: dict[str, dict[str, dict[str, Any]]],
    left_selections: dict[str, dict[str, dict[str, Any]]],
    right_selections: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    """Explain why two paired evaluations differ.

    Returns a sorted list drawn from ``manual_correction`` and
    ``candidate_switch`` (both may apply); an empty list means the
    conclusion cannot be attributed to either recorded change signal.
    """

    regions = left["regions"]
    label = left["label"]
    origins: set[str] = set()
    for region in regions:
        if left_corrections.get(region, {}) != right_corrections.get(region, {}):
            origins.add("manual_correction")
        left_basis = _basis_region(left, region)
        right_basis = _basis_region(right, region)
        if left_basis["kind"] != right_basis["kind"]:
            if {left_basis["kind"], right_basis["kind"]} & {"manual", "source"}:
                # Only a correction edit can move a region into/out of the
                # manual state; auto -> pending is a coverage/propagation
                # consequence handled below.
                origins.add("manual_correction")
        if (
            left_basis.get("basis_label") != right_basis.get("basis_label")
            and (
                left_basis["kind"] == "manual"
                or right_basis["kind"] == "manual"
            )
        ):
            origins.add("manual_correction")
        for field in ("candidate_id", "selection_method", "transform_parameters"):
            if left_basis.get(field) != right_basis.get(field):
                origins.add("candidate_switch")
        # The provenance of a propagated mask also depends on the adopted
        # candidate at every basis timepoint: switching a candidate at the
        # basis label changes the warp even when the target label's own
        # candidate id stays the same.
        for basis_label in {
            left_basis.get("basis_label"),
            right_basis.get("basis_label"),
        }:
            if basis_label is None:
                continue
            left_pick = left_selections.get(region, {}).get(basis_label)
            right_pick = right_selections.get(region, {}).get(basis_label)
            if left_pick is not None and right_pick is not None and left_pick != right_pick:
                origins.add("candidate_switch")
    return sorted(origins)


def _series_selections(
    version_dir: Path, report: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-region adopted candidate decisions from the frozen propagation saves."""

    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for region_name in report.get("regions", {}):
        path = version_dir / "regions" / region_name / "propagation_report.json"
        if not path.exists():
            continue
        propagation_report = json.loads(path.read_text(encoding="utf-8"))
        selections[region_name] = dict(
            propagation_report.get("series_selections", {})
        )
    return selections


def compare_relation_reviews(
    left_dir: str | Path,
    right_dir: str | Path,
) -> dict[str, Any]:
    """Compare two frozen relation review directories read-only.

    Returns a JSON-clean comparison grouped by stable relation identity
    into ``unchanged`` / ``added`` / ``withdrawn`` / ``conclusion_changed``.
    Neither side is reloaded, recomputed or modified.
    """

    left_report = read_frozen_review(left_dir)
    right_report = read_frozen_review(right_dir)
    left_dir = Path(left_dir)
    right_dir = Path(right_dir)
    return _compare_reports(
        left_report,
        right_report,
        _file_fingerprint(left_dir),
        _file_fingerprint(right_dir),
        left_dir,
        right_dir,
    )


def _compare_reports(
    left_report: dict[str, Any],
    right_report: dict[str, Any],
    left_fingerprint: dict[str, Any],
    right_fingerprint: dict[str, Any],
    left_dir: Path,
    right_dir: Path,
) -> dict[str, Any]:
    left_map = _evaluation_map(left_report)
    right_map = _evaluation_map(right_report)
    left_corrections = _correction_lookup(left_report)
    right_corrections = _correction_lookup(right_report)
    left_selections = _series_selections(left_dir, left_report)
    right_selections = _series_selections(right_dir, right_report)

    entries: list[dict[str, Any]] = []
    all_identities = sorted(set(left_map) | set(right_map))
    counts = {"unchanged": 0, "added": 0, "withdrawn": 0, "conclusion_changed": 0}
    for identity in all_identities:
        left = left_map.get(identity)
        right = right_map.get(identity)
        entry: dict[str, Any] = {
            "identity": list(identity),
            "identity_key": format_identity(identity),
            "regions": [identity[0], identity[1]],
            "relation": identity[2],
            "label": identity[3],
            "left": _snapshot(left),
            "right": _snapshot(right),
        }
        if left is None and right is not None:
            entry["kind"] = "added"
            entry["change_origins"] = ["declaration_update"]
        elif left is not None and right is None:
            entry["kind"] = "withdrawn"
            entry["change_origins"] = ["declaration_update"]
        elif _conclusion(left) == _conclusion(right):
            entry["kind"] = "unchanged"
            entry["change_origins"] = []
        else:
            entry["kind"] = "conclusion_changed"
            entry["before"] = {
                "status": left["status"],
                "actual": left["actual"],
                "offending_voxels": left["offending_voxels"],
            }
            entry["after"] = {
                "status": right["status"],
                "actual": right["actual"],
                "offending_voxels": right["offending_voxels"],
            }
            entry["change_origins"] = _classify_origin(
                left,
                right,
                left_corrections,
                right_corrections,
                left_selections,
                right_selections,
            )
        counts[entry["kind"]] += 1
        entries.append(entry)

    grouped = {kind: [] for kind in counts}
    for entry in entries:
        grouped[entry["kind"]].append(entry)

    comparison = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "left": {
            "name": left_report.get("name"),
            "created_utc": left_report.get("created_utc"),
            "frozen_report": left_fingerprint,
        },
        "right": {
            "name": right_report.get("name"),
            "created_utc": right_report.get("created_utc"),
            "frozen_report": right_fingerprint,
        },
        "labels_left": list(left_report.get("labels", [])),
        "labels_right": list(right_report.get("labels", [])),
        "expectations_left": list(left_report.get("expectations", [])),
        "expectations_right": list(right_report.get("expectations", [])),
        "counts": counts,
        "entries": entries,
        "unchanged": grouped["unchanged"],
        "added": grouped["added"],
        "withdrawn": grouped["withdrawn"],
        "conclusion_changed": grouped["conclusion_changed"],
        "read_only": True,
        "note": (
            "comparison only explains differences; no mask, correction, "
            "candidate choice or verdict of either frozen review is modified"
        ),
    }
    return _numpy_clean(comparison)


def export_relation_comparison(
    left_dir: str | Path,
    right_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the read-only comparison and return the comparison report.

    The output directory may be one of the two reviewed version
    directories or any other location; the reviewed reports themselves
    are never written to.
    """

    comparison = compare_relation_reviews(left_dir, right_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / COMPARE_FILENAME).write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return comparison


def verify_relation_comparison(
    output_dir: str | Path,
    left_dir: str | Path | None = None,
    right_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Reopen an exported comparison and check it reproduces identically.

    The comparison is recomputed from the two frozen reports located via
    the explicit arguments or the paths recorded inside the export. The
    input report SHA-256 fingerprints must match; if they do, reopening
    yields exactly the same grouping and change origins.
    """

    output_dir = Path(output_dir)
    saved = json.loads(
        (output_dir / COMPARE_FILENAME).read_text(encoding="utf-8")
    )
    if saved.get("schema") != SCHEMA:
        raise ValueError(f"unsupported comparison schema {saved.get('schema')!r}")
    left_dir = Path(left_dir) if left_dir is not None else Path(
        saved["left"]["frozen_report"]["path"]
    )
    right_dir = Path(right_dir) if right_dir is not None else Path(
        saved["right"]["frozen_report"]["path"]
    )
    for side, directory in (("left", left_dir), ("right", right_dir)):
        fingerprint = _file_fingerprint(directory)
        if fingerprint["sha256"] != saved[side]["frozen_report"]["sha256"]:
            raise ValueError(
                f"{side} frozen report changed since the comparison was exported"
            )
        if fingerprint.get("files") != saved[side]["frozen_report"].get("files"):
            raise ValueError(
                f"{side} frozen provenance files changed since the comparison "
                "was exported"
            )
    rebuilt = compare_relation_reviews(left_dir, right_dir)
    for field in (
        "counts",
        "entries",
        "labels_left",
        "labels_right",
        "expectations_left",
        "expectations_right",
    ):
        if rebuilt[field] != saved[field]:
            raise ValueError(f"comparison field {field!r} is not reproducible")
    return {"verified": True, "counts": saved["counts"]}
