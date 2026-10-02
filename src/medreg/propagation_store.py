from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .batch import _numpy_clean
from .image import MedicalImage
from .propagation import PropagationSession
from .series import SeriesResult

SCHEMA = "medreg-propagation-v1"


def _image_fingerprint(image: MedicalImage) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(image.data.tobytes())
    digest.update(np.asarray(image.affine, dtype=np.float64).tobytes())
    return {
        "shape": list(image.shape),
        "spacing": [float(v) for v in image.spacing],
        "data_sha256": digest.hexdigest(),
    }


def _mask_fingerprint(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest()


def _save_mask(image: MedicalImage, mask: np.ndarray, path: Path) -> None:
    MedicalImage(mask.astype(np.float32), image.affine).save_nifti(path)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(MedicalImage.load_nifti(path).data > 0.5)


def save_propagation(session: PropagationSession, version_dir: str | Path) -> dict[str, Any]:
    """Persist a propagation session so every region stays traceable.

    Each region record stores its mask file, the adopted candidate, the
    transform parameters used, and the decision (original/auto/manual), so
    a reopened version can be audited against the source images.
    """

    if session.original_mask is None:
        raise ValueError("nothing to save: draw the original region first")
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for label in session.series.labels:
        record = session.records[label]
        entry = _numpy_clean(record.summary())
        if record.mask is not None:
            mask_path = version_dir / f"region_{label}.nii.gz"
            _save_mask(session.reference, record.mask, mask_path)
            entry["mask_file"] = mask_path.name
            entry["mask_sha256"] = _mask_fingerprint(record.mask)
        else:
            entry["mask_file"] = None
            entry["mask_sha256"] = None
        records.append(entry)

    original_path = version_dir / "region_original.nii.gz"
    _save_mask(session.reference, session.original_mask, original_path)
    corrections: list[dict[str, Any]] = []
    for correction in session.corrections.values():
        mask_path = version_dir / f"correction_{correction.label}.nii.gz"
        _save_mask(session.reference, correction.mask, mask_path)
        corrections.append({
            "label": correction.label,
            "use_as_basis": correction.use_as_basis,
            "mask_file": mask_path.name,
            "mask_sha256": _mask_fingerprint(correction.mask),
        })

    selections = {
        tp.label: {
            "selected_candidate_id": (
                tp.multi.selected.candidate_id if tp.multi is not None and tp.multi.selected else None
            ),
            "selection_method": tp.multi.selection_method if tp.multi is not None else None,
            "status": tp.status,
        }
        for tp in session.series.timepoints
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_label": session.source_label,
        "labels": list(session.series.labels),
        "config": {"min_region_coverage": session.config.min_region_coverage},
        "reference_image": _image_fingerprint(session.reference),
        "original_region": {
            "mask_file": original_path.name,
            "mask_sha256": _mask_fingerprint(session.original_mask),
        },
        "timepoint_selections": selections,
        "records": records,
        "corrections": corrections,
        "curve": _numpy_clean(session.change_curve()),
        "lineage": _numpy_clean(session.lineage()),
    }
    (version_dir / "propagation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_propagation(
    version_dir: str | Path,
    series: SeriesResult,
    images: list[MedicalImage],
) -> PropagationSession:
    """Reopen a saved propagation session on top of a registered series.

    The stored original drawing and manual corrections are replayed onto the
    given series; automatic propagations are recomputed from the series'
    adopted candidates, never copied from disk.
    """

    version_dir = Path(version_dir)
    report_path = version_dir / "propagation_report.json"
    if not report_path.exists():
        raise ValueError(f"no propagation_report.json in {version_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["schema"] != SCHEMA:
        raise ValueError(f"unsupported propagation schema {report['schema']!r}")
    if list(series.labels) != list(report["labels"]):
        raise ValueError("series labels do not match the saved propagation")

    from .propagation import PropagationConfig

    session = PropagationSession(
        series,
        images,
        source_label=report["source_label"],
        config=PropagationConfig(
            min_region_coverage=float(report["config"]["min_region_coverage"])
        ),
    )
    session.draw_region(_load_mask(version_dir / report["original_region"]["mask_file"]))
    for correction in report["corrections"]:
        session.apply_correction(
            correction["label"],
            _load_mask(version_dir / correction["mask_file"]),
            use_as_basis=bool(correction["use_as_basis"]),
        )
    return session


def verify_propagation(
    version_dir: str | Path,
    series: SeriesResult,
    images: list[MedicalImage],
) -> dict[str, Any]:
    """Reload a saved propagation and check every region is reproducible.

    Recomputes all automatic propagations from the reopened series and
    compares status, coverage, masks, and the change curve against what was
    saved. Raises ValueError on the first inconsistency.
    """

    version_dir = Path(version_dir)
    report = json.loads((version_dir / "propagation_report.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    reference = images[series.config.reference_index]
    fingerprint = _image_fingerprint(reference)
    if fingerprint["data_sha256"] != report["reference_image"]["data_sha256"]:
        problems.append("reference image differs from the one the regions were derived from")

    for label, selection in report["timepoint_selections"].items():
        tp = series.timepoint(label)
        selected = tp.multi.selected.candidate_id if tp.multi is not None and tp.multi.selected else None
        if selected != selection["selected_candidate_id"]:
            problems.append(
                f"{label}: adopted candidate {selected} does not match saved {selection['selected_candidate_id']}"
            )

    session = load_propagation(version_dir, series, images)
    if _mask_fingerprint(session.original_mask) != report["original_region"]["mask_sha256"]:
        problems.append("original drawing hash mismatch after reopening")

    stored_records = {entry["label"]: entry for entry in report["records"]}
    for label in session.series.labels:
        record = session.records[label]
        stored = stored_records.get(label)
        if stored is None:
            problems.append(f"{label}: missing saved record")
            continue
        if record.status != stored["status"] or record.decision != stored["decision"]:
            problems.append(
                f"{label}: status/decision {record.status}/{record.decision} "
                f"!= saved {stored['status']}/{stored['decision']}"
            )
        saved_coverage = stored["coverage_fraction"]
        if (record.coverage_fraction is None) != (saved_coverage is None) or (
            record.coverage_fraction is not None
            and saved_coverage is not None
            and abs(record.coverage_fraction - saved_coverage) > 1e-9
        ):
            problems.append(f"{label}: coverage {record.coverage_fraction} != saved {saved_coverage}")
        if record.mask is None:
            if stored["mask_file"] is not None:
                problems.append(f"{label}: saved a mask but recomputation has none")
        else:
            if stored["mask_file"] is None:
                problems.append(f"{label}: recomputed a mask but none was saved")
            elif _mask_fingerprint(record.mask) != stored["mask_sha256"]:
                problems.append(f"{label}: recomputed region differs from the saved mask")

    if _numpy_clean(session.change_curve()) != report["curve"]:
        problems.append("recomputed change curve differs from the saved curve")

    if problems:
        raise ValueError("propagation verification failed: " + "; ".join(problems))
    return {
        "source_label": report["source_label"],
        "labels": report["labels"],
        "corrections": [c["label"] for c in report["corrections"]],
        "verified": True,
    }

