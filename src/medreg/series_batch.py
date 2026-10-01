from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .batch import _numpy_clean, _utc_stamp, parameter_hash
from .image import MedicalImage
from .register import CandidateResult
from .roi import RegionOfInterest
from .series import SeriesConfig, SeriesResult, register_series, select_series_candidate


@dataclass(frozen=True)
class SeriesItem:
    """One temporal series: a name plus per-series registration settings."""

    name: str
    reference_index: int = 0
    labels: tuple[str, ...] = ()
    levels: tuple[int, ...] = (4, 2, 1)
    max_iterations: int = 120
    min_coverage: float = 0.35
    initial_parameter_sets: list[list[float]] = field(default_factory=list)
    per_timepoint_initial_sets: dict[str, list[list[float]]] = field(default_factory=dict)
    roi_world_min: list[float] | None = None
    roi_world_max: list[float] | None = None
    min_roi_coverage: float = 0.6
    min_roi_ncc: float = 0.7
    jump_translation_mm: float = 5.0
    jump_rotation_deg: float = 8.0
    manual_selections: dict[str, str] = field(default_factory=dict)


def _series_roi(item: SeriesItem) -> RegionOfInterest | None:
    if item.roi_world_min is None or item.roi_world_max is None:
        return None
    return RegionOfInterest(
        tuple(item.roi_world_min),  # type: ignore[arg-type]
        tuple(item.roi_world_max),  # type: ignore[arg-type]
        name=f"{item.name}_roi",
    )


def _series_config(item: SeriesItem) -> SeriesConfig:
    return SeriesConfig(
        reference_index=item.reference_index,
        levels=tuple(item.levels),
        max_iterations=item.max_iterations,
        min_coverage=item.min_coverage,
        min_roi_coverage=item.min_roi_coverage,
        min_roi_ncc=item.min_roi_ncc,
        jump_translation_mm=item.jump_translation_mm,
        jump_rotation_deg=item.jump_rotation_deg,
    )


def _candidate_outputs(candidate: CandidateResult, timepoint_dir: Path) -> dict[str, str | None]:
    outcome = candidate.result
    transformed_path = timepoint_dir / f"{candidate.candidate_id}_moving_transformed.nii.gz"
    mask_path = timepoint_dir / f"{candidate.candidate_id}_overlap_mask.npy"
    if outcome.transformed is not None:
        outcome.transformed.save_nifti(transformed_path)
        np.save(mask_path, outcome.overlap_mask)
        return {"transformed": str(transformed_path), "overlap_mask": str(mask_path)}
    return {"transformed": None, "overlap_mask": None}


def _write_timepoint(
    series_result: SeriesResult,
    label: str,
    moving: MedicalImage,
    timepoint_dir: Path,
) -> dict[str, Any]:
    tp = series_result.timepoint(label)
    timepoint_dir.mkdir(parents=True, exist_ok=True)
    moving.save_nifti(timepoint_dir / "moving_original.nii.gz")
    record: dict[str, Any] = _numpy_clean(tp.summary())
    record["outputs"] = {"moving_original": str(timepoint_dir / "moving_original.nii.gz")}
    if tp.multi is not None:
        candidate_records = []
        for candidate in tp.multi.candidates:
            candidate_record = _numpy_clean(candidate.summary())
            candidate_record["outputs"] = _candidate_outputs(candidate, timepoint_dir)
            candidate_records.append(candidate_record)
        record["candidates"] = candidate_records
        selected = tp.multi.selected
        if selected is not None and selected.result.transformed is not None:
            selected.result.transformed.save_nifti(timepoint_dir / "moving_transformed_selected.nii.gz")
            np.save(timepoint_dir / "overlap_mask_selected.npy", selected.result.overlap_mask)
            record["outputs"]["transformed_selected"] = str(
                timepoint_dir / "moving_transformed_selected.nii.gz"
            )
            record["outputs"]["overlap_mask_selected"] = str(
                timepoint_dir / "overlap_mask_selected.npy"
            )
    (timepoint_dir / "timepoint_report.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record


def _effective_config(item: SeriesItem, labels: list[str]) -> dict[str, Any]:
    return _numpy_clean(
        asdict(item)
        | {
            "labels": labels,
            "levels": list(item.levels),
            "initial_parameter_sets": item.initial_parameter_sets or [[0.0] * 6],
        }
    )


def run_series_batch(
    series_items: list[tuple[SeriesItem, list[MedicalImage]]],
    output_dir: str | Path,
    parameter_file: str | Path | None = None,
) -> dict[str, Any]:
    """Register every temporal series and version the results on disk.

    Layout per series version::

        <output>/<series>/<version_id>/
            series_report.json
            reference_original.nii.gz
            <label>/moving_original.nii.gz
            <label>/<candidate>_moving_transformed.nii.gz
            <label>/moving_transformed_selected.nii.gz
            <label>/timepoint_report.json

    Manual selections and parameter changes produce a new version directory;
    existing versions are never overwritten.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provided = None
    parameter_source = str(parameter_file) if parameter_file else "inline"
    if parameter_file:
        parameter_file = Path(parameter_file)
        provided = json.loads(parameter_file.read_text(encoding="utf-8"))
        parameter_source = f"{parameter_source}@{parameter_hash(provided)}"

    report: dict[str, Any] = {
        "schema": "medreg-series-batch-v1",
        "parameter_source": parameter_source,
        "series": [],
    }
    for item, images in series_items:
        configured = item
        if provided is not None and item.name in provided.get("series", {}):
            configured = _apply_series_overrides(item, provided["series"][item.name])
        labels = list(configured.labels) or [f"t{index:02d}" for index in range(len(images))]
        config = _series_config(configured)
        roi = _series_roi(configured)
        default_sets = configured.initial_parameter_sets or [[0.0] * 6]
        per_index: dict[int, list[list[float]]] = {}
        for index, label in enumerate(labels):
            per_index[index] = configured.per_timepoint_initial_sets.get(label, default_sets)

        result = register_series(images, labels, config, roi, per_index)
        for label, candidate_id in configured.manual_selections.items():
            result = select_series_candidate(result, label, candidate_id)

        effective = _effective_config(configured, labels)
        digest = parameter_hash(effective)[:12]
        if configured.manual_selections:
            choice = "manual-" + "_".join(
                f"{label}:{cid}" for label, cid in sorted(configured.manual_selections.items())
            )
        else:
            choice = "auto"
        version_id = f"{_utc_stamp()}_{digest}_{choice}"[:120]
        version_dir = output_dir / item.name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        images[config.reference_index].save_nifti(version_dir / "reference_original.nii.gz")
        timepoint_records = []
        for index, label in enumerate(labels):
            if index == config.reference_index:
                record = _numpy_clean(result.timepoint(label).summary())
                record["outputs"] = {
                    "moving_original": str(version_dir / "reference_original.nii.gz")
                }
                (version_dir / label).mkdir(parents=True, exist_ok=True)
                (version_dir / label / "timepoint_report.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            else:
                record = _write_timepoint(result, label, images[index], version_dir / label)
            timepoint_records.append(record)

        series_report = {
            "name": item.name,
            "version_id": version_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parameter_source": parameter_source,
            "effective_parameters": effective,
            "effective_parameter_hash": parameter_hash(effective),
            "manual_selections": dict(configured.manual_selections),
            "status": result.status,
            "reference_index": config.reference_index,
            "reference_label": labels[config.reference_index],
            "labels": labels,
            "roi": _numpy_clean(result.roi_summary) if result.roi_summary else None,
            "adjacency": _numpy_clean(result.adjacency),
            "timepoints": timepoint_records,
            "outputs": {
                "reference_original": str(version_dir / "reference_original.nii.gz"),
                "timepoint_dirs": {label: str(version_dir / label) for label in labels},
            },
        }
        (version_dir / "series_report.json").write_text(
            json.dumps(series_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        report["series"].append(series_report)

        versions_path = output_dir / item.name / "versions.json"
        ledger: list[Any] = []
        if versions_path.exists():
            ledger = json.loads(versions_path.read_text(encoding="utf-8"))
        ledger.append(
            {
                "version_id": version_id,
                "effective_parameter_hash": series_report["effective_parameter_hash"],
                "manual_selections": dict(configured.manual_selections),
                "status": result.status,
                "created_utc": series_report["created_utc"],
            }
        )
        versions_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    report["batch_parameter_hash"] = parameter_hash(
        [entry["effective_parameters"] for entry in report["series"]]
    )
    (output_dir / "series_batch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _apply_series_overrides(item: SeriesItem, override: dict[str, Any]) -> SeriesItem:
    return SeriesItem(
        name=item.name,
        reference_index=int(override.get("reference_index", item.reference_index)),
        labels=tuple(override.get("labels", item.labels)),
        levels=tuple(override.get("levels", item.levels)),
        max_iterations=int(override.get("max_iterations", item.max_iterations)),
        min_coverage=float(override.get("min_coverage", item.min_coverage)),
        initial_parameter_sets=[
            list(map(float, values))
            for values in override.get("initial_parameter_sets", item.initial_parameter_sets)
        ],
        per_timepoint_initial_sets={
            label: [list(map(float, values)) for values in sets]
            for label, sets in override.get(
                "per_timepoint_initial_sets", item.per_timepoint_initial_sets
            ).items()
        },
        roi_world_min=override.get("roi_world_min", item.roi_world_min),
        roi_world_max=override.get("roi_world_max", item.roi_world_max),
        min_roi_coverage=float(override.get("min_roi_coverage", item.min_roi_coverage)),
        min_roi_ncc=float(override.get("min_roi_ncc", item.min_roi_ncc)),
        jump_translation_mm=float(override.get("jump_translation_mm", item.jump_translation_mm)),
        jump_rotation_deg=float(override.get("jump_rotation_deg", item.jump_rotation_deg)),
        manual_selections=dict(override.get("manual_selections", item.manual_selections)),
    )


def verify_series_version(version_dir: str | Path) -> dict[str, Any]:
    """Reload a saved series version and check one-to-one correspondence.

    Confirms that every timepoint's original image, its retained candidates,
    the recorded manual/automatic selection, and the selected transformed
    image still match each other after saving and reopening. Raises
    ValueError on the first inconsistency.
    """

    version_dir = Path(version_dir)
    report_path = version_dir / "series_report.json"
    if not report_path.exists():
        raise ValueError(f"no series_report.json in {version_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    reference_path = Path(report["outputs"]["reference_original"])
    reference = None
    if not reference_path.exists():
        problems.append(f"missing reference original {reference_path}")
    else:
        reference = MedicalImage.load_nifti(reference_path)

    labels = report["labels"]
    seen_timepoints: list[str] = []
    for record in report["timepoints"]:
        label = record["label"]
        seen_timepoints.append(label)
        tp_dir = version_dir / label
        original = tp_dir / "moving_original.nii.gz"
        if record["is_reference"]:
            original = version_dir / "reference_original.nii.gz"
        if not original.exists():
            problems.append(f"{label}: missing original image {original}")
        manual = report["manual_selections"].get(label)
        selected_id = record.get("selected_candidate_id")
        if manual is not None and manual != selected_id:
            problems.append(
                f"{label}: recorded manual selection {manual} does not match selected {selected_id}"
            )
        candidates = record.get("candidates", [])
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            problems.append(f"{label}: duplicate candidate ids {candidate_ids}")
        if not record["is_reference"] and record["status"] != "rejected":
            if selected_id is not None and selected_id not in candidate_ids:
                problems.append(f"{label}: selected {selected_id} not among retained candidates")
        for candidate in candidates:
            outputs = candidate.get("outputs", {})
            transformed = outputs.get("transformed")
            if candidate["status"] == "success":
                if not transformed or not Path(transformed).exists():
                    problems.append(f"{label}/{candidate['candidate_id']}: missing transformed output")
            elif transformed is not None:
                problems.append(
                    f"{label}/{candidate['candidate_id']}: failed candidate must not have outputs"
                )
            if (
                candidate["candidate_id"] == selected_id
                and candidate["status"] == "success"
                and reference is not None
            ):
                selected_file = tp_dir / "moving_transformed_selected.nii.gz"
                if not selected_file.exists():
                    problems.append(f"{label}: selected transform image missing")
                else:
                    reloaded_selected = MedicalImage.load_nifti(selected_file)
                    reloaded_candidate = MedicalImage.load_nifti(transformed)
                    if not np.allclose(reloaded_selected.data, reloaded_candidate.data, atol=1e-5):
                        problems.append(
                            f"{label}: selected image does not match candidate {selected_id} output"
                        )
                    if not np.allclose(reloaded_selected.affine, reference.affine, atol=1e-9):
                        problems.append(f"{label}: selected image grid differs from reference grid")
    if seen_timepoints != labels:
        problems.append(f"timepoint order {seen_timepoints} does not match labels {labels}")

    if problems:
        raise ValueError("series version verification failed: " + "; ".join(problems))
    return {
        "version_id": report["version_id"],
        "name": report["name"],
        "labels": labels,
        "manual_selections": report["manual_selections"],
        "status": report["status"],
        "verified": True,
    }

