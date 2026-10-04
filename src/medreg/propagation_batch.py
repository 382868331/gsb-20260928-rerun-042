from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .batch import _numpy_clean
from .image import MedicalImage
from .multi_region import MultiRegionPropagationSession
from .propagation import PropagationSession
from .series import SeriesResult


def _image_fingerprint(image: MedicalImage) -> dict[str, Any]:
    return {"shape": list(image.shape), "affine": np.asarray(image.affine).tolist()}


def save_propagation(
    session: PropagationSession,
    version_dir: str | Path,
    name: str = "propagation",
) -> dict[str, Any]:
    """Persist a propagation session so every region stays traceable.

    Layout::

        <version_dir>/
            propagation_report.json
            anchor_mask.nii.gz
            corrections/<label>.nii.gz
            regions/<label>_mask.nii.gz   (source/auto/manual only)

    Pending timepoints deliberately have no mask file; the report records
    why they are pending instead of fabricating a complete contour.
    """

    version_dir = Path(version_dir)
    (version_dir / "corrections").mkdir(parents=True, exist_ok=True)
    (version_dir / "regions").mkdir(parents=True, exist_ok=True)

    anchor_image = session._image(session.anchor_label)
    MedicalImage(session.anchor_mask.astype(np.float32), anchor_image.affine).save_nifti(
        version_dir / "anchor_mask.nii.gz"
    )

    correction_records: dict[str, Any] = {}
    for label, mask in sorted(session.corrections.items()):
        path = version_dir / "corrections" / f"{label}.nii.gz"
        MedicalImage(mask.astype(np.float32), session._image(label).affine).save_nifti(path)
        correction_records[label] = {
            "use_as_basis": label in session.basis_labels,
            "path": str(path),
        }

    region_records: list[dict[str, Any]] = []
    for label in session.labels:
        region = session.region(label)
        record = _numpy_clean(region.summary())
        record["image"] = _image_fingerprint(session._image(label))
        if region.mask is not None:
            path = version_dir / "regions" / f"{label}_mask.nii.gz"
            MedicalImage(
                region.mask.astype(np.float32), session._image(label).affine
            ).save_nifti(path)
            record["mask_path"] = str(path)
        else:
            record["mask_path"] = None
        region_records.append(record)

    selections = {
        label: {
            "selected_candidate_id": session.series.timepoint(label).summary()[
                "selected_candidate_id"
            ],
            "selection_method": session.series.timepoint(label).summary()["selection_method"],
        }
        for label in session.labels
    }
    report: dict[str, Any] = {
        "schema": "medreg-propagation-v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "labels": list(session.labels),
        "anchor_label": session.anchor_label,
        "anchor_mask_path": str(version_dir / "anchor_mask.nii.gz"),
        "min_region_coverage": session.min_region_coverage,
        "corrections": correction_records,
        "series_selections": selections,
        "regions": region_records,
        "change_curve": _numpy_clean(session.change_curve()),
    }
    report["propagation_graph"] = _numpy_clean(session.propagation_graph())
    (version_dir / "propagation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_propagation(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> PropagationSession:
    """Rebuild a propagation session from a saved version directory.

    The caller supplies the images and the (re-run or reloaded) series
    result; the saved per-label image fingerprints and series selections
    are checked so a region is never re-attached to a different image,
    transform, or adoption decision than the one it was saved with.
    """

    version_dir = Path(version_dir)
    report = json.loads((version_dir / "propagation_report.json").read_text(encoding="utf-8"))
    _check_report_against(report, images, series)

    anchor_mask = np.asarray(
        MedicalImage.load_nifti(report["anchor_mask_path"]).data > 0.5
    )
    session = PropagationSession(
        list(images),
        series,
        report["anchor_label"],
        anchor_mask,
        min_region_coverage=float(report["min_region_coverage"]),
    )
    for label, record in report["corrections"].items():
        mask = np.asarray(MedicalImage.load_nifti(record["path"]).data > 0.5)
        session.apply_correction(label, mask, use_as_basis=bool(record["use_as_basis"]))
    return session


def _check_report_against(
    report: dict[str, Any], images: list[MedicalImage], series: SeriesResult
) -> None:
    labels = report["labels"]
    if list(series.labels) != labels:
        raise ValueError("saved propagation labels do not match the series")
    if len(images) != len(labels):
        raise ValueError("image count does not match the saved propagation")
    problems: list[str] = []
    for label, record in zip(labels, report["regions"]):
        if record["label"] != label:
            problems.append(f"region record order mismatch at {label}")
            continue
        image = images[labels.index(label)]
        fingerprint = record["image"]
        if list(image.shape) != fingerprint["shape"] or not np.allclose(
            image.affine, fingerprint["affine"], atol=1e-9
        ):
            problems.append(f"{label}: image grid differs from the saved provenance")
    for label, saved in report["series_selections"].items():
        current = series.timepoint(label).summary()
        if current["selected_candidate_id"] != saved["selected_candidate_id"]:
            problems.append(
                f"{label}: adopted candidate {current['selected_candidate_id']} does not "
                f"match the saved decision {saved['selected_candidate_id']}"
            )
    if problems:
        raise ValueError("propagation provenance mismatch: " + "; ".join(problems))


def verify_propagation_version(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> dict[str, Any]:
    """Reload a saved propagation and check every region one by one.

    Confirms that after saving and reopening, each region still matches
    its recorded source image, transform, and adoption decision: pending
    timepoints must have no mask file, corrections and the anchor drawing
    must be byte-identical in content, and recomputed automatic regions
    must equal the saved masks. Raises ValueError on the first
    inconsistency.
    """

    version_dir = Path(version_dir)
    report = json.loads((version_dir / "propagation_report.json").read_text(encoding="utf-8"))
    session = load_propagation(version_dir, images, series)
    problems: list[str] = []
    for record in report["regions"]:
        label = record["label"]
        region = session.region(label)
        if region.kind != record["kind"]:
            problems.append(f"{label}: kind {region.kind} != saved {record['kind']}")
        mask_path = record["mask_path"]
        if mask_path is None:
            if region.mask is not None:
                problems.append(f"{label}: pending region unexpectedly has a mask")
            continue
        if not Path(mask_path).exists():
            problems.append(f"{label}: missing region mask {mask_path}")
            continue
        saved_mask = np.asarray(MedicalImage.load_nifti(mask_path).data > 0.5)
        if region.mask is None or not np.array_equal(saved_mask, region.mask):
            problems.append(f"{label}: recomputed region differs from the saved mask")
    anchor_saved = np.asarray(MedicalImage.load_nifti(report["anchor_mask_path"]).data > 0.5)
    if not np.array_equal(anchor_saved, session.anchor_mask):
        problems.append("anchor drawing changed after reload")
    if set(report["corrections"]) != set(session.corrections):
        problems.append("corrections differ after reload")
    if problems:
        raise ValueError("propagation verification failed: " + "; ".join(problems))
    return {
        "name": report["name"],
        "labels": report["labels"],
        "anchor_label": report["anchor_label"],
        "corrections": sorted(report["corrections"]),
        "verified": True,
    }


def save_multi_region_propagation(
    session: MultiRegionPropagationSession,
    version_dir: str | Path,
    name: str = "multi_region_propagation",
) -> dict[str, Any]:
    """Persist every named region, manual correction, and relation evidence."""

    version_dir = Path(version_dir)
    anchors_dir = version_dir / "multi_region" / "anchors"
    corrections_dir = version_dir / "multi_region" / "corrections"
    regions_dir = version_dir / "multi_region" / "regions"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    corrections_dir.mkdir(parents=True, exist_ok=True)
    regions_dir.mkdir(parents=True, exist_ok=True)

    region_records: dict[str, Any] = {}
    for region_id, track in sorted(session.regions.items()):
        anchor_image = track._image(track.anchor_label)
        anchor_path = anchors_dir / f"{region_id}__{track.anchor_label}_anchor.nii.gz"
        MedicalImage(track.anchor_mask.astype(np.float32), anchor_image.affine).save_nifti(
            anchor_path
        )
        correction_records: dict[str, Any] = {}
        for label, mask in sorted(track.corrections.items()):
            correction_path = corrections_dir / f"{region_id}__{label}.nii.gz"
            MedicalImage(mask.astype(np.float32), track._image(label).affine).save_nifti(
                correction_path
            )
            correction_records[label] = {
                "use_as_basis": label in track.basis_labels,
                "path": str(correction_path),
            }
        mask_records: dict[str, Any] = {}
        for label in track.labels:
            region = track.region(label)
            if region.mask is None:
                mask_records[label] = None
                continue
            mask_path = regions_dir / f"{region_id}__{label}_mask.nii.gz"
            MedicalImage(
                region.mask.astype(np.float32), track._image(label).affine
            ).save_nifti(mask_path)
            mask_records[label] = str(mask_path)
        region_records[region_id] = {
            "anchor_label": track.anchor_label,
            "anchor_path": str(anchor_path),
            "corrections": correction_records,
            "mask_paths": mask_records,
        }

    relationship_export = _numpy_clean(session.export_relationships())
    export_path = version_dir / "multi_region" / "relationship_export.json"
    export_path.write_text(
        json.dumps(relationship_export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": "medreg-multi-region-propagation-v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "labels": list(session.series.labels),
        "min_region_coverage": session.min_region_coverage,
        "regions": region_records,
        "declared_relationships": [
            spec.summary() for spec in session.relationships.values()
        ],
        "relationship_export_path": str(export_path),
        "relationship_review": relationship_export["relationship_review"],
    }
    report_path = version_dir / "multi_region_propagation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_multi_region_propagation(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> MultiRegionPropagationSession:
    """Reload named tracks without dropping conflicts or manual masks."""

    version_dir = Path(version_dir)
    report_path = version_dir / "multi_region_propagation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if list(series.labels) != report["labels"]:
        raise ValueError("saved multi-region labels do not match the series")
    session = MultiRegionPropagationSession(
        list(images),
        series,
        min_region_coverage=float(report["min_region_coverage"]),
    )
    for region_id, record in report["regions"].items():
        anchor = np.asarray(MedicalImage.load_nifti(record["anchor_path"]).data > 0.5)
        session.add_region(region_id, record["anchor_label"], anchor)
        track = session.region_session(region_id)
        for label, correction in record["corrections"].items():
            mask = np.asarray(MedicalImage.load_nifti(correction["path"]).data > 0.5)
            track.apply_correction(
                label, mask, use_as_basis=bool(correction["use_as_basis"])
            )
    for spec in report["declared_relationships"]:
        session.define_relationship(
            spec["first_region"],
            spec["second_region"],
            spec["expected"],
            declared_at=spec["declared_at"],
            source=spec["source"],
            note=spec.get("note"),
        )
    return session


def verify_multi_region_propagation_version(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> dict[str, Any]:
    """Check all saved masks and relationship findings after reload."""

    version_dir = Path(version_dir)
    report = json.loads(
        (version_dir / "multi_region_propagation_report.json").read_text(encoding="utf-8")
    )
    session = load_multi_region_propagation(version_dir, images, series)
    problems: list[str] = []
    if set(session.regions) != set(report["regions"]):
        problems.append("region ids differ after reload")
    for region_id, record in report["regions"].items():
        track = session.region_session(region_id)
        anchor_saved = np.asarray(MedicalImage.load_nifti(record["anchor_path"]).data > 0.5)
        if not np.array_equal(anchor_saved, track.anchor_mask):
            problems.append(f"{region_id}: anchor changed after reload")
        for label, mask_path in record["mask_paths"].items():
            current = track.region(label).mask
            if mask_path is None:
                if current is not None:
                    problems.append(f"{region_id}/{label}: pending mask appeared after reload")
                continue
            saved_mask = np.asarray(MedicalImage.load_nifti(mask_path).data > 0.5)
            if current is None or not np.array_equal(saved_mask, current):
                problems.append(f"{region_id}/{label}: region mask differs after reload")

    saved_review = {
        item["relation_id"]: item for item in report["relationship_review"]
    }
    reloaded_review = {
        item["relation_id"]: item for item in session.review_relationships()
    }
    if set(saved_review) != set(reloaded_review):
        problems.append("relationship ids differ after reload")
    for relation_id, saved in saved_review.items():
        current = reloaded_review.get(relation_id)
        if current is None:
            continue
        if current["status"] != saved["status"]:
            problems.append(
                f"{relation_id}: status {current['status']} != saved {saved['status']}"
            )
        if current["conflict_count"] != saved["conflict_count"]:
            problems.append(f"{relation_id}: conflict count changed after reload")
        if current.get("source") != saved.get("source"):
            problems.append(f"{relation_id}: relationship source changed after reload")
    if problems:
        raise ValueError("multi-region propagation verification failed: " + "; ".join(problems))
    return {
        "name": report["name"],
        "regions": sorted(session.regions),
        "relationships": sorted(saved_review),
        "conflicts": {
            relation_id: item["conflict_count"]
            for relation_id, item in saved_review.items()
            if item["conflict_count"]
        },
        "verified": True,
    }
