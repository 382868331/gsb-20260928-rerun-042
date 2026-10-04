
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch import _numpy_clean
from .image import MedicalImage
from .propagation_batch import load_propagation, save_propagation, verify_propagation_version
from .relations import MultiRegionSession
from .series import SeriesResult


def save_relations(
    session: MultiRegionSession,
    version_dir: str | Path,
    name: str = "relations",
) -> dict[str, Any]:
    """Persist a multi-region session including relation conflicts.

    Layout::

        <version_dir>/
            multi_region_report.json
            regions/<name>/...        (one full propagation save per region)

    The report stores the declared expectations and every evaluation with
    its provenance, so conflicts caused by manual corrections survive a
    save/reload cycle together with the correction that produced them.
    """

    version_dir = Path(version_dir)
    (version_dir / "regions").mkdir(parents=True, exist_ok=True)
    for region_name in session.region_names:
        save_propagation(
            session.region_session(region_name),
            version_dir / "regions" / region_name,
            name=region_name,
        )
    summary = session.summary()
    report: dict[str, Any] = {
        "schema": "medreg-multiregion-v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "labels": summary["labels"],
        "min_region_coverage": summary["min_region_coverage"],
        "regions": summary["regions"],
        "expectations": summary["expectations"],
        "evaluations": _numpy_clean(summary["evaluations"]),
        "conflicts": _numpy_clean(summary["conflicts"]),
    }
    (version_dir / "multi_region_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_relations(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> MultiRegionSession:
    """Rebuild a multi-region session from a saved version directory.

    Each region's propagation is reloaded (and provenance-checked) via
    load_propagation; declared expectations are restored and the
    recomputed evaluations must reproduce the saved ones, so conflicts
    and their provenance are preserved rather than silently resolved.
    """

    version_dir = Path(version_dir)
    report = json.loads(
        (version_dir / "multi_region_report.json").read_text(encoding="utf-8")
    )
    if report.get("schema") != "medreg-multiregion-v1":
        raise ValueError(f"unsupported multi-region schema {report.get('schema')!r}")
    session = MultiRegionSession(
        list(images),
        series,
        min_region_coverage=float(report["min_region_coverage"]),
    )
    for region_name, region_record in report["regions"].items():
        sub = load_propagation(
            version_dir / "regions" / region_name, list(images), series
        )
        if sub.anchor_label != region_record["anchor_label"]:
            raise ValueError(
                f"region {region_name!r}: anchor label changed after reload"
            )
        session._sessions[region_name] = sub
    for expectation in report["expectations"]:
        session.expect(
            expectation["region_a"], expectation["region_b"], expectation["relation"]
        )
    _check_evaluations_match(report, session)
    return session


def _check_evaluations_match(
    report: dict[str, Any], session: MultiRegionSession
) -> None:
    saved = {
        (e["regions"][0], e["regions"][1], e["relation"], e["label"]): e
        for e in report["evaluations"]
    }
    problems: list[str] = []
    for record in session.evaluate_relations():
        key = (
            record["regions"][0],
            record["regions"][1],
            record["relation"],
            record["label"],
        )
        previous = saved.get(key)
        if previous is None:
            problems.append(f"unexpected evaluation {key}")
            continue
        for field in ("status", "actual", "offending_voxels", "manual_origin"):
            if previous[field] != record[field]:
                problems.append(
                    f"{key}: {field} {record[field]!r} != saved {previous[field]!r}"
                )
    if problems:
        raise ValueError(
            "relation evaluations changed after reload: " + "; ".join(problems)
        )


def verify_relations_version(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> dict[str, Any]:
    """Reload a saved multi-region version and check every part.

    Verifies each region's propagation (masks, corrections, adoption
    decisions) and confirms the saved conflicts are exactly the ones
    reproduced after reload — nothing is dropped or auto-resolved.
    """

    version_dir = Path(version_dir)
    report = json.loads(
        (version_dir / "multi_region_report.json").read_text(encoding="utf-8")
    )
    session = load_relations(version_dir, images, series)
    for region_name in report["regions"]:
        verify_propagation_version(version_dir / "regions" / region_name, images, series)
    saved_conflicts = {
        (c["regions"][0], c["regions"][1], c["relation"], c["label"])
        for c in report["conflicts"]
    }
    current_conflicts = {
        (c["regions"][0], c["regions"][1], c["relation"], c["label"])
        for c in session.conflicts()
    }
    if saved_conflicts != current_conflicts:
        raise ValueError(
            "conflict set changed after reload: "
            f"saved {sorted(saved_conflicts)} != current {sorted(current_conflicts)}"
        )
    return {
        "name": report["name"],
        "labels": report["labels"],
        "regions": sorted(report["regions"]),
        "conflicts": sorted(saved_conflicts),
        "verified": True,
    }

