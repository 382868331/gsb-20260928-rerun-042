from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .image import MedicalImage
from .register import (
    CandidateResult,
    MultiStartResult,
    RegistrationResult,
    apply_manual_selection,
    register_multi,
)
from .roi import RegionOfInterest
from .transform import RigidTransform


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


@dataclass(frozen=True)
class BatchItem:
    name: str
    fixed_path: str | None = None
    moving_path: str | None = None
    initial_parameters: list[float] | None = None
    levels: tuple[int, ...] = (4, 2, 1)
    max_iterations: int = 120
    min_coverage: float = 0.35
    initial_parameter_sets: list[list[float]] = field(default_factory=list)
    roi_world_min: list[float] | None = None
    roi_world_max: list[float] | None = None
    min_roi_coverage: float = 0.6
    min_roi_ncc: float = 0.7
    manual_candidate_id: str | None = None


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parameter_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _numpy_clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _numpy_clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_numpy_clean(item) for item in value]
    return value


def _initial_sets(configured: BatchItem) -> list[list[float]]:
    if configured.initial_parameter_sets:
        return [list(map(float, values)) for values in configured.initial_parameter_sets]
    if configured.initial_parameters is not None:
        return [list(map(float, configured.initial_parameters))]
    return [[0.0] * 6]


def _configured_roi(configured: BatchItem) -> RegionOfInterest | None:
    if configured.roi_world_min is None or configured.roi_world_max is None:
        return None
    return RegionOfInterest(
        tuple(configured.roi_world_min),  # type: ignore[arg-type]
        tuple(configured.roi_world_max),  # type: ignore[arg-type]
        name=f"{configured.name}_roi",
    )


def _candidate_outputs(candidate: CandidateResult, version_dir: Path) -> dict[str, str | None]:
    outcome = candidate.result
    transformed_path = version_dir / f"{candidate.candidate_id}_moving_transformed.nii.gz"
    mask_path = version_dir / f"{candidate.candidate_id}_overlap_mask.npy"
    if outcome.transformed is not None:
        outcome.transformed.save_nifti(transformed_path)
        np.save(mask_path, outcome.overlap_mask)
        return {
            "transformed": str(transformed_path),
            "overlap_mask": str(mask_path),
        }
    return {"transformed": None, "overlap_mask": None}


def _write_version(
    *,
    item: BatchItem,
    fixed: MedicalImage,
    moving: MedicalImage,
    configured: BatchItem,
    multi: MultiStartResult,
    version_dir: Path,
    parameter_source: str,
    manual_requested: str | None,
) -> dict[str, Any]:
    version_dir.mkdir(parents=True, exist_ok=True)
    fixed.save_nifti(version_dir / "fixed_original.nii.gz")
    moving.save_nifti(version_dir / "moving_original.nii.gz")

    candidate_records: list[dict[str, Any]] = []
    for candidate in multi.candidates:
        outputs = _candidate_outputs(candidate, version_dir)
        record = _numpy_clean(candidate.summary())
        record["outputs"] = outputs
        candidate_records.append(record)

    if multi.selected is not None and multi.selected.result.transformed is not None:
        multi.selected.result.transformed.save_nifti(version_dir / "moving_transformed_selected.nii.gz")
        np.save(version_dir / "overlap_mask_selected.npy", multi.selected.result.overlap_mask)

    effective_config = _numpy_clean(
        asdict(configured)
        | {
            "levels": list(configured.levels),
            "initial_parameter_sets": _initial_sets(configured),
        }
    )
    selected = multi.selected
    return {
        "name": item.name,
        "version_id": version_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_source": parameter_source,
        "manual_candidate_requested": manual_requested,
        "effective_parameters": effective_config,
        "effective_parameter_hash": parameter_hash(effective_config),
        "selection": {
            "method": multi.selection_method,
            "selected_candidate_id": selected.candidate_id if selected else None,
            "uncertain": multi.uncertain,
            "uncertainty_reasons": multi.uncertainty_reasons,
            "status": multi.status,
        },
        "outputs": {
            "fixed": str(version_dir / "fixed_original.nii.gz"),
            "moving": str(version_dir / "moving_original.nii.gz"),
            "transformed": str(version_dir / "moving_transformed_selected.nii.gz")
            if selected is not None and selected.result.transformed is not None
            else None,
            "overlap_mask": str(version_dir / "overlap_mask_selected.npy")
            if selected is not None and selected.result.transformed is not None
            else None,
            "candidate_dir": str(version_dir),
        },
        "candidates": candidate_records,
        "result": _numpy_clean(multi.selected.summary()) if selected is not None else _numpy_clean(multi.summary()),
        "multi_start": _numpy_clean(multi.summary()),
    }


def _apply_overrides(item: BatchItem, override: dict[str, Any]) -> BatchItem:
    return BatchItem(
        item.name,
        item.fixed_path,
        item.moving_path,
        override.get("initial_parameters", item.initial_parameters),
        tuple(override.get("levels", item.levels)),
        int(override.get("max_iterations", item.max_iterations)),
        float(override.get("min_coverage", item.min_coverage)),
        [list(map(float, values)) for values in override.get("initial_parameter_sets", item.initial_parameter_sets)],
        override.get("roi_world_min", item.roi_world_min),
        override.get("roi_world_max", item.roi_world_max),
        float(override.get("min_roi_coverage", item.min_roi_coverage)),
        float(override.get("min_roi_ncc", item.min_roi_ncc)),
        override.get("manual_candidate_id", item.manual_candidate_id),
    )


def run_batch(
    items: list[tuple[BatchItem, MedicalImage, MedicalImage]],
    output_dir: str | Path,
    parameter_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run every item and store each configuration or manual choice as a version.

    Existing version directories are never overwritten: a changed parameter
    hash or a changed manual candidate id produces a new directory and is
    appended to ``versions.json``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provided_parameters = None
    parameter_source = str(parameter_file) if parameter_file else "inline"
    if parameter_file:
        parameter_file = Path(parameter_file)
        provided_parameters = json.loads(parameter_file.read_text(encoding="utf-8"))
        parameter_source = f"{parameter_source}@{parameter_hash(provided_parameters)}"

    report: dict[str, Any] = {"schema": "medreg-batch-v2", "parameter_source": parameter_source, "items": []}
    for item, fixed, moving in items:
        configured = item
        if provided_parameters is not None and item.name in provided_parameters.get("items", {}):
            configured = _apply_overrides(item, provided_parameters["items"][item.name])

        initial_sets = _initial_sets(configured)
        configured = BatchItem(
            configured.name,
            configured.fixed_path,
            configured.moving_path,
            configured.initial_parameters,
            configured.levels,
            configured.max_iterations,
            configured.min_coverage,
            initial_sets,
            configured.roi_world_min,
            configured.roi_world_max,
            configured.min_roi_coverage,
            configured.min_roi_ncc,
            configured.manual_candidate_id,
        )
        roi = _configured_roi(configured)
        multi = register_multi(
            fixed,
            moving,
            initial_sets,
            roi,
            configured.levels,
            configured.max_iterations,
            configured.min_coverage,
            configured.min_roi_coverage,
            configured.min_roi_ncc,
        )
        manual_requested = configured.manual_candidate_id
        if manual_requested is not None:
            multi = apply_manual_selection(multi, manual_requested)

        config_preview = _numpy_clean(
            asdict(configured)
            | {"levels": list(configured.levels), "initial_parameter_sets": initial_sets}
        )
        config_digest = parameter_hash(config_preview)[:12]
        choice = (
            f"manual-{manual_requested}" if manual_requested is not None else f"auto-{multi.selection_method}"
        )
        version_id = f"{_utc_stamp()}_{config_digest}_{choice}"
        version_dir = output_dir / item.name / version_id
        item_report = _write_version(
            item=item,
            fixed=fixed,
            moving=moving,
            configured=configured,
            multi=multi,
            version_dir=version_dir,
            parameter_source=parameter_source,
            manual_requested=manual_requested,
        )
        report["items"].append(item_report)

        versions_path = output_dir / item.name / "versions.json"
        ledger: list[Any] = []
        if versions_path.exists():
            ledger = json.loads(versions_path.read_text(encoding="utf-8"))
        ledger.append(
            {
                "version_id": version_id,
                "effective_parameter_hash": item_report["effective_parameter_hash"],
                "selected_candidate_id": item_report["selection"]["selected_candidate_id"],
                "selection_method": item_report["selection"]["method"],
                "status": item_report["selection"]["status"],
                "uncertain": item_report["selection"]["uncertain"],
                "created_utc": item_report["created_utc"],
            }
        )
        versions_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    report["batch_parameter_hash"] = parameter_hash(
        [item["effective_parameters"] for item in report["items"]]
    )
    (output_dir / "batch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
