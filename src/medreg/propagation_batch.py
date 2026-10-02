from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .batch import _numpy_clean
from .image import MedicalImage
from .propagation import (
    MultiRegionPropagationSession,
    PropagationSession,
    RegionRelation,
)
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
    """Persist multiple propagated regions, relationship audits, and conflicts."""

    version_dir = Path(version_dir)
    anchors_dir = version_dir / "anchors"
    corrections_dir = version_dir / "corrections"
    regions_dir = version_dir / "regions"
    conflicts_dir = version_dir / "conflicts"
    for directory in (anchors_dir, corrections_dir, regions_dir, conflicts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    anchors: dict[str, Any] = {}
    for region_id, single in session.sessions.items():
        path = anchors_dir / f"{region_id}.nii.gz"
        MedicalImage(
            single.anchor_mask.astype(np.float32),
            session.images[session.labels.index(single.anchor_label)].affine,
        ).save_nifti(path)
        anchors[region_id] = {
            "anchor_label": single.anchor_label,
            "mask_path": str(path),
        }

    corrections: dict[str, dict[str, Any]] = {}
    regions: dict[str, dict[str, Any]] = {}
    for region_id, single in session.sessions.items():
        corrections[region_id] = {}
        regions[region_id] = {}
        for label in session.labels:
            region = single.region(label)
            record = _numpy_clean(region.summary())
            record["image"] = _image_fingerprint(single._image(label))
            if region.mask is not None:
                mask_path = regions_dir / region_id / f"{label}_mask.nii.gz"
                MedicalImage(
                    region.mask.astype(np.float32), single._image(label).affine
                ).save_nifti(mask_path)
                record["mask_path"] = str(mask_path)
            else:
                record["mask_path"] = None
            regions[region_id][label] = record
        for label, mask in single.corrections.items():
            mask_path = corrections_dir / region_id / f"{label}.nii.gz"
            MedicalImage(mask.astype(np.float32), single._image(label).affine).save_nifti(mask_path)
            corrections[region_id][label] = {
                "use_as_basis": label in single.basis_labels,
                "mask_path": str(mask_path),
            }

    relationship_report = session.relationship_report()
    serialized_relationships: list[dict[str, Any]] = []
    for index, entry in enumerate(relationship_report):
        record = {key: _numpy_clean(value) for key, value in entry.items() if key != "conflict_mask"}
        conflict = entry.get("conflict_mask")
        if conflict is not None:
            label = entry["label"]
            conflict_path = conflicts_dir / f"{index:03d}_{label}_{entry['region_a']}_{entry['region_b']}.nii.gz"
            MedicalImage(
                conflict.astype(np.float32), session.images[session.labels.index(label)].affine
            ).save_nifti(conflict_path)
            record["conflict_mask_path"] = str(conflict_path)
        else:
            record["conflict_mask_path"] = None
        serialized_relationships.append(record)

    selections = {
        label: {
            "selected_candidate_id": session.series.timepoint(label).summary()[
                "selected_candidate_id"
            ],
            "selection_method": session.series.timepoint(label).summary()["selection_method"],
        }
        for label in session.labels
    }
    report = {
        "schema": "medreg-multi-region-propagation-v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "labels": list(session.labels),
        "min_region_coverage": session.min_region_coverage,
        "anchors": anchors,
        "relations": [_numpy_clean(relation.summary()) for relation in session.relations],
        "corrections": corrections,
        "series_selections": selections,
        "regions": regions,
        "relationship_report": serialized_relationships,
        "change_curve": _numpy_clean(session.change_curve()),
    }
    (version_dir / "multi_region_propagation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_multi_region_propagation(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> MultiRegionPropagationSession:
    version_dir = Path(version_dir)
    report_path = version_dir / "multi_region_propagation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    anchors: dict[str, tuple[str, np.ndarray]] = {}
    for region_id, record in report["anchors"].items():
        mask = np.asarray(MedicalImage.load_nifti(record["mask_path"]).data > 0.5)
        anchors[region_id] = (record["anchor_label"], mask)
    relations = [RegionRelation(**item) for item in report["relations"]]
    session = MultiRegionPropagationSession(
        list(images),
        series,
        anchors,
        relations,
        min_region_coverage=float(report["min_region_coverage"]),
    )
    for region_id, corrections in report["corrections"].items():
        for label, record in corrections.items():
            mask = np.asarray(MedicalImage.load_nifti(record["mask_path"]).data > 0.5)
            session.apply_correction(
                region_id, label, mask, use_as_basis=bool(record["use_as_basis"])
            )
    _check_multi_report_against(report, images, series, session)
    return session


def _check_multi_report_against(
    report: dict[str, Any],
    images: list[MedicalImage],
    series: SeriesResult,
    session: MultiRegionPropagationSession,
) -> None:
    problems: list[str] = []
    if list(series.labels) != report["labels"]:
        problems.append("timepoint labels do not match saved propagation")
    for label, saved in report["series_selections"].items():
        current = series.timepoint(label).summary()
        if current["selected_candidate_id"] != saved["selected_candidate_id"]:
            problems.append(f"{label}: adopted candidate does not match saved decision")
    for region_id, labels in report["regions"].items():
        for label, saved in labels.items():
            current = session.region(region_id, label)
            image = images[series.labels.index(label)]
            fingerprint = saved["image"]
            if list(image.shape) != fingerprint["shape"] or not np.allclose(
                image.affine, fingerprint["affine"], atol=1e-9
            ):
                problems.append(f"{region_id}/{label}: image grid provenance mismatch")
            if current.kind != saved["kind"]:
                problems.append(f"{region_id}/{label}: result kind changed after reload")
            if saved["mask_path"] is None:
                if current.mask is not None:
                    problems.append(f"{region_id}/{label}: saved pending result gained a mask")
                continue
            saved_mask = np.asarray(MedicalImage.load_nifti(saved["mask_path"]).data > 0.5)
            if current.mask is None or not np.array_equal(saved_mask, current.mask):
                problems.append(f"{region_id}/{label}: saved mask differs from recomputation")
    current_report = session.relationship_report()
    if len(current_report) != len(report["relationship_report"]):
        problems.append("relationship report length changed after reload")
    else:
        for saved, current in zip(report["relationship_report"], current_report):
            for key in (
                "label",
                "region_a",
                "region_b",
                "kind",
                "start_label",
                "end_label",
                "status",
                "overlap_voxels",
                "contact_voxels",
                "conflict_voxels",
            ):
                if saved.get(key) != _numpy_clean(current.get(key)):
                    problems.append(f"{saved.get('label')}/{saved.get('region_a')}-{saved.get('region_b')}: {key} changed")
                    break
            saved_conflict_path = saved.get("conflict_mask_path")
            if saved_conflict_path is None:
                if current.get("conflict_mask") is not None:
                    problems.append("an unsaved relationship conflict appeared after reload")
            else:
                saved_conflict = np.asarray(MedicalImage.load_nifti(saved_conflict_path).data > 0.5)
                current_conflict = current.get("conflict_mask")
                if current_conflict is None or not np.array_equal(saved_conflict, current_conflict):
                    problems.append("saved conflict mask differs from recomputation")
    if problems:
        raise ValueError("multi-region propagation provenance mismatch: " + "; ".join(problems))


def verify_multi_region_propagation_version(
    version_dir: str | Path,
    images: list[MedicalImage],
    series: SeriesResult,
) -> dict[str, Any]:
    session = load_multi_region_propagation(version_dir, images, series)
    return {
        "verified": True,
        "labels": session.labels,
        "regions": sorted(session.sessions),
        "violated_labels": sorted(
            {
                entry["label"]
                for entry in session.relationship_report()
                if entry["status"] == "violated"
            }
        ),
    }
