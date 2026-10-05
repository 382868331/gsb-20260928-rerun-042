"""Comparison export between two frozen multi-region review results.

The diff is strictly read-only: it loads two saved relations versions,
compares their declared expectations and evaluations, and writes a
reproducible report. Neither frozen version's masks, candidates nor
conclusions are modified.

A stable relation identity is the ordered tuple
``(region_a, region_b, relation, label)``. For every identity the diff
records one of:

* ``unchanged``           - present on both sides with the same result;
* ``added``               - present only in the right version;
* ``withdrawn``           - present only in the left version;
* ``conclusion_changed``  - present on both but status/actual/evidence differ.

Each non-unchanged identity carries the change origin(s), derived from
provenance saved with the frozen reviews:

* ``manual_correction``        - a region basis differs because of a
  manual correction (kind manual, a changed basis anchor, or a
  correction present on exactly one side);
* ``candidate_switch``         - the adopted propagation candidate or its
  transform parameters differ between the two versions;
* ``relation_declaration_update`` - the expectation exists on only one
  side;
* ``other``                    - no provenance change explains the result.

Both sides' timepoints, sources (region basis, candidate, transform,
coverage) and out-of-container voxel evidence are kept verbatim, so the
report can be reopened and deterministically reproduced.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch import _numpy_clean
from .image import MedicalImage
from .relations_batch import load_relations
from .series import SeriesResult

#: Output schema identifier for frozen review comparisons.
DIFF_SCHEMA = "medreg-multiregion-diff-v1"

#: Origins that can explain why a stable identity changed.
ORIGIN_MANUAL = "manual_correction"
ORIGIN_CANDIDATE = "candidate_switch"
ORIGIN_DECLARATION = "relation_declaration_update"
ORIGIN_OTHER = "other"

# Fields copied verbatim from each side's frozen evaluation.
_SIDE_FIELDS = (
    "label",
    "relation",
    "regions",
    "status",
    "actual",
    "offending_voxels",
    "manual_origin",
    "reasons",
    "basis",
)


def _read_report(version_dir: str | Path) -> dict[str, Any]:
    version_dir = Path(version_dir)
    report = json.loads(
        (version_dir / "multi_region_report.json").read_text(encoding="utf-8")
    )
    if report.get("schema") != "medreg-multiregion-v1":
        raise ValueError(
            f"unsupported multi-region schema {report.get('schema')!r} in {version_dir}"
        )
    return report


def _evaluation_map(
    report: dict[str, Any],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            record["regions"][0],
            record["regions"][1],
            record["relation"],
            record["label"],
        ): record
        for record in report["evaluations"]
    }


def _expectation_pairs(report: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (item["region_a"], item["region_b"], item["relation"])
        for item in report["expectations"]
    }


def _side_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in _SIDE_FIELDS}


def _correction_regions(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (region, label) -> saved correction metadata of one version."""
    corrections: dict[tuple[str, str], dict[str, Any]] = {}
    for region_name, region_record in report.get("regions", {}).items():
        for label, meta in region_record.get("corrections", {}).items():
            corrections[(region_name, label)] = meta
    return corrections


def _changed_origins(
    key: tuple[str, str, str, str],
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
) -> list[str]:
    """Derive change origins strictly from the two frozen reports."""
    region_a, region_b, relation, label = key
    pair = (region_a, region_b, relation)

    if left is None or right is None:
        # The identity is missing on one side: either its declaration was
        # added/withdrawn, or the timepoint did not exist on that side.
        present_pairs = _expectation_pairs(right_report if left is None else left_report)
        return [ORIGIN_DECLARATION] if pair in present_pairs else [ORIGIN_OTHER]

    left_corrections = _correction_regions(left_report)
    right_corrections = _correction_regions(right_report)
    manual = False
    candidate = False
    for region in (region_a, region_b):
        left_basis = left["basis"][region]
        right_basis = right["basis"][region]
        left_corr = left_corrections.get((region, label))
        right_corr = right_corrections.get((region, label))
        kinds = (left_basis.get("kind"), right_basis.get("kind"))
        if kinds[0] != kinds[1] and "manual" in kinds:
            manual = True
        if left_basis.get("basis_label") != right_basis.get("basis_label") and (
            left_corr is not None or right_corr is not None
        ):
            manual = True
        if (left_corr is not None) != (right_corr is not None) or left_corr != right_corr:
            manual = True
        if (
            left_basis.get("candidate_id") != right_basis.get("candidate_id")
            or left_basis.get("transform_parameters")
            != right_basis.get("transform_parameters")
        ):
            candidate = True

    origins: list[str] = []
    if manual:
        origins.append(ORIGIN_MANUAL)
    if candidate:
        origins.append(ORIGIN_CANDIDATE)
    if not origins:
        origins.append(ORIGIN_OTHER)
    return origins


def _entry(
    key: tuple[str, str, str, str],
    disposition: str,
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    origins: list[str],
) -> dict[str, Any]:
    region_a, region_b, relation, label = key
    conclusion_changed = bool(
        left is not None
        and right is not None
        and (left["status"] != right["status"] or left["actual"] != right["actual"])
    )
    evidence_changed = bool(
        left is not None
        and right is not None
        and left["offending_voxels"] != right["offending_voxels"]
    )
    return {
        "relation_id": {
            "region_a": region_a,
            "region_b": region_b,
            "relation": relation,
            "label": label,
        },
        "disposition": disposition,
        "origins": origins,
        "conclusion_changed": conclusion_changed,
        "evidence_changed": evidence_changed,
        "left": _side_snapshot(left) if left is not None else None,
        "right": _side_snapshot(right) if right is not None else None,
    }


def compare_relation_reviews(
    left_dir: str | Path,
    right_dir: str | Path,
) -> dict[str, Any]:
    """Compare two frozen multi-region reviews without touching either.

    Returns a report mapping every stable relation identity to one of
    ``unchanged`` / ``added`` / ``withdrawn`` / ``conclusion_changed``.
    Each entry keeps both sides' timepoints, sources and out-of-voxel
    evidence. The function is pure: nothing in either directory changes.
    """

    left_dir, right_dir = Path(left_dir), Path(right_dir)
    left_report = _read_report(left_dir)
    right_report = _read_report(right_dir)
    left_map = _evaluation_map(left_report)
    right_map = _evaluation_map(right_report)

    unchanged: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    conclusion_changed: list[dict[str, Any]] = []

    for current_key in sorted(set(left_map) | set(right_map)):
        left = left_map.get(current_key)
        right = right_map.get(current_key)
        if left is not None and right is None:
            withdrawn.append(
                _entry(
                    current_key,
                    "withdrawn",
                    left,
                    None,
                    _changed_origins(current_key, left, None, left_report, right_report),
                )
            )
        elif left is None and right is not None:
            added.append(
                _entry(
                    current_key,
                    "added",
                    None,
                    right,
                    _changed_origins(current_key, None, right, left_report, right_report),
                )
            )
        else:
            assert left is not None and right is not None
            differs = (
                left["status"] != right["status"]
                or left["actual"] != right["actual"]
                or left["offending_voxels"] != right["offending_voxels"]
                or left["manual_origin"] != right["manual_origin"]
            )
            if not differs:
                unchanged.append(
                    _entry(
                        current_key,
                        "unchanged",
                        left,
                        right,
                        _changed_origins(
                            current_key, left, right, left_report, right_report
                        ),
                    )
                )
            else:
                conclusion_changed.append(
                    _entry(
                        current_key,
                        "conclusion_changed",
                        left,
                        right,
                        _changed_origins(
                            current_key, left, right, left_report, right_report
                        ),
                    )
                )

    return {
        "left": _version_ref(left_dir, left_report),
        "right": _version_ref(right_dir, right_report),
        "labels": {
            "left": list(left_report["labels"]),
            "right": list(right_report["labels"]),
        },
        "counts": {
            "unchanged": len(unchanged),
            "added": len(added),
            "withdrawn": len(withdrawn),
            "conclusion_changed": len(conclusion_changed),
        },
        "unchanged": unchanged,
        "added": added,
        "withdrawn": withdrawn,
        "conclusion_changed": conclusion_changed,
    }


def _version_ref(version_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "version_dir": str(version_dir),
        "name": report.get("name"),
        "created_utc": report.get("created_utc"),
        "schema": report.get("schema"),
    }


def _canonical_payload(comparison: dict[str, Any]) -> bytes:
    payload = {
        key: value
        for key, value in comparison.items()
        if key not in ("schema", "created_utc", "content_digest")
    }
    return json.dumps(
        _numpy_clean(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def export_relation_diff(
    left_dir: str | Path,
    right_dir: str | Path,
    output_dir: str | Path,
    name: str = "relations_diff",
) -> dict[str, Any]:
    """Compute the frozen-review diff and persist it reproducibly.

    Writes ``relation_diff_report.json`` under ``output_dir``. The report
    carries a SHA-256 content digest over everything except its own
    timestamp; re-running the export over the same two frozen directories
    yields the same digest, so a reopened comparison reproduces exactly.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_relation_reviews(left_dir, right_dir)
    digest = hashlib.sha256(_canonical_payload(comparison)).hexdigest()
    report: dict[str, Any] = {
        "schema": DIFF_SCHEMA,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **comparison,
        "content_digest": digest,
    }
    (output_dir / "relation_diff_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def verify_relation_diff(
    left_dir: str | Path,
    right_dir: str | Path,
    output_dir: str | Path,
    images: list[MedicalImage],
    series_left: SeriesResult,
    series_right: SeriesResult,
) -> dict[str, Any]:
    """Reopen a diff export and confirm it reproduces from both versions.

    Each frozen side is fully verified (masks, candidates, corrections
    and conclusions); the comparison is then recomputed and compared
    against the saved export apart from its export timestamp. Nothing in
    either frozen version is modified.
    """

    from .relations_batch import verify_relations_version

    output_dir = Path(output_dir)
    saved = json.loads(
        (output_dir / "relation_diff_report.json").read_text(encoding="utf-8")
    )
    if saved.get("schema") != DIFF_SCHEMA:
        raise ValueError(f"unsupported relation diff schema {saved.get('schema')!r}")

    left_verified = verify_relations_version(left_dir, images, series_left)
    right_verified = verify_relations_version(right_dir, images, series_right)
    # Reloading also proves every saved evaluation recomputes identically.
    load_relations(left_dir, images, series_left)
    load_relations(right_dir, images, series_right)

    recomputed = compare_relation_reviews(left_dir, right_dir)
    saved_body = {
        key: value
        for key, value in saved.items()
        if key not in ("schema", "name", "created_utc", "content_digest")
    }
    if recomputed != saved_body:
        raise ValueError("relation diff content changed after reload")
    digest = hashlib.sha256(_canonical_payload(recomputed)).hexdigest()
    if digest != saved["content_digest"]:
        raise ValueError(
            f"relation diff digest mismatch: {digest} != saved {saved['content_digest']}"
        )
    return {
        "name": saved["name"],
        "counts": saved["counts"],
        "content_digest": digest,
        "left_verified": left_verified,
        "right_verified": right_verified,
        "verified": True,
    }
