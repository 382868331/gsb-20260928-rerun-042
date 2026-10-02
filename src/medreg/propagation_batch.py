from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .batch import _numpy_clean
from .image import MedicalImage
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
