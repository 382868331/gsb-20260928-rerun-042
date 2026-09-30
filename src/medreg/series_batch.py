from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .batch import _numpy_clean, _utc_stamp, parameter_hash
from .image import MedicalImage
from .roi import RegionOfInterest
from .series import SeriesResult, apply_series_manual_selection, register_series
from .transform import RigidTransform


@dataclass(frozen=True)
class SeriesItem:
    """Configuration for one time-series registration item."""

    name: str
    reference_time_id: str
    time_order: tuple[str, ...] = ()
    initial_parameter_sets: list[list[float]] = field(default_factory=list)
    levels: tuple[int, ...] = (4, 2, 1)
    max_iterations: int = 120
    min_coverage: float = 0.35
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


def _initial_sets(item: SeriesItem) -> list[list[float]]:
    if item.initial_parameter_sets:
        return [list(map(float, values)) for values in item.initial_parameter_sets]
    return [[0.0] * 6]


def _apply_overrides(item: SeriesItem, override: dict[str, Any]) -> SeriesItem:
    return SeriesItem(
        item.name,
        override.get("reference_time_id", item.reference_time_id),
        tuple(override.get("time_order", item.time_order)),
        [list(map(float, values)) for values in override.get("initial_parameter_sets", item.initial_parameter_sets)],
        tuple(override.get("levels", item.levels)),
        int(override.get("max_iterations", item.max_iterations)),
        float(override.get("min_coverage", item.min_coverage)),
        override.get("roi_world_min", item.roi_world_min),
        override.get("roi_world_max", item.roi_world_max),
        float(override.get("min_roi_coverage", item.min_roi_coverage)),
        float(override.get("min_roi_ncc", item.min_roi_ncc)),
        float(override.get("jump_translation_mm", item.jump_translation_mm)),
        float(override.get("jump_rotation_deg", item.jump_rotation_deg)),
        dict(override.get("manual_selections", item.manual_selections)),
    )


def _write_time_point_outputs(
    series: SeriesResult,
    images: dict[str, MedicalImage],
    series_dir: Path,
) -> dict[str, Any]:
    """Persist per-time originals, candidates and selections; return state."""

    times_state: dict[str, Any] = {}
    for point in series.time_points:
        time_dir = series_dir / "times" / point.time_id
        time_dir.mkdir(parents=True, exist_ok=True)
        moving_path = time_dir / "moving_original.nii.gz"
        images[point.time_id].save_nifti(moving_path)
        entry: dict[str, Any] = {
            "time_id": point.time_id,
            "is_reference": point.is_reference,
            "status": point.status,
            "rejection_reasons": list(point.rejection_reasons),
            "moving_original": str(moving_path),
            "selected_candidate_id": None,
            "selected_transform_parameters": None,
            "selected_outputs": None,
            "candidates": [],
        }
        if point.is_reference:
            times_state[point.time_id] = entry
            continue
        multi = point.multi
        assert multi is not None
        for candidate in multi.candidates:
            record = _numpy_clean(candidate.summary())
            transformed = candidate.result.transformed
            if transformed is not None:
                transformed_path = time_dir / f"{candidate.candidate_id}_moving_transformed.nii.gz"
                mask_path = time_dir / f"{candidate.candidate_id}_overlap_mask.npy"
                transformed.save_nifti(transformed_path)
                np.save(mask_path, candidate.result.overlap_mask)
                record["outputs"] = {
                    "transformed": str(transformed_path),
                    "overlap_mask": str(mask_path),
                }
            else:
                record["outputs"] = {"transformed": None, "overlap_mask": None}
            entry["candidates"].append(record)
        selected = multi.selected
        if selected is not None and selected.result.transformed is not None:
            selected_path = time_dir / "moving_transformed_selected.nii.gz"
            selected_mask_path = time_dir / "overlap_mask_selected.npy"
            selected.result.transformed.save_nifti(selected_path)
            np.save(selected_mask_path, selected.result.overlap_mask)
            entry["selected_candidate_id"] = selected.candidate_id
            entry["selected_transform_parameters"] = _numpy_clean(
                selected.result.transform.parameters()
            )
            entry["selected_outputs"] = {
                "transformed": str(selected_path),
                "overlap_mask": str(selected_mask_path),
            }
        elif selected is not None and selected.result.transform is not None:
            entry["selected_candidate_id"] = selected.candidate_id
            entry["selected_transform_parameters"] = _numpy_clean(
                selected.result.transform.parameters()
            )
        times_state[point.time_id] = entry
    return times_state


def run_series_batch(
    items: list[tuple[SeriesItem, dict[str, MedicalImage]]],
    output_dir: str | Path,
    parameter_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run time-series items and persist every time point of every version.

    Each version directory stores the original image, all candidates and the
    selected transform per time point plus a series_state.json that keeps the
    one-to-one mapping between selections, transforms and originals. Existing
    versions are never overwritten.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provided_parameters = None
    parameter_source = str(parameter_file) if parameter_file else "inline"
    if parameter_file:
        parameter_file = Path(parameter_file)
        provided_parameters = json.loads(parameter_file.read_text(encoding="utf-8"))
        parameter_source = f"{parameter_source}@{parameter_hash(provided_parameters)}"

    report: dict[str, Any] = {"schema": "medreg-series-v1", "parameter_source": parameter_source, "items": []}
    for item, images in items:
        configured = item
        if provided_parameters is not None and item.name in provided_parameters.get("items", {}):
            configured = _apply_overrides(item, provided_parameters["items"][item.name])

        roi = _series_roi(configured)
        series = register_series(
            images,
            configured.reference_time_id,
            _initial_sets(configured),
            roi,
            configured.levels,
            configured.max_iterations,
            configured.min_coverage,
            configured.min_roi_coverage,
            configured.min_roi_ncc,
            configured.jump_translation_mm,
            configured.jump_rotation_deg,
            list(configured.time_order) if configured.time_order else None,
        )
        for time_id, candidate_id in configured.manual_selections.items():
            series = apply_series_manual_selection(
                series,
                time_id,
                candidate_id,
                configured.jump_translation_mm,
                configured.jump_rotation_deg,
            )

        effective_config = _numpy_clean(
            asdict(configured)
            | {
                "levels": list(configured.levels),
                "time_order": list(configured.time_order),
                "initial_parameter_sets": _initial_sets(configured),
            }
        )
        config_digest = parameter_hash(effective_config)[:12]
        choice = "manual" if configured.manual_selections else "auto"
        version_id = f"{_utc_stamp()}_{config_digest}_{choice}"
        version_dir = output_dir / item.name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        times_state = _write_time_point_outputs(series, images, version_dir)
        state = {
            "schema": "medreg-series-state-v1",
            "name": item.name,
            "version_id": version_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference_time_id": series.reference_time_id,
            "selections": {
                time_id: entry["selected_candidate_id"] for time_id, entry in times_state.items()
            },
            "times": times_state,
        }
        (version_dir / "series_state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        item_report = {
            "name": item.name,
            "version_id": version_id,
            "created_utc": state["created_utc"],
            "parameter_source": parameter_source,
            "effective_parameters": effective_config,
            "effective_parameter_hash": parameter_hash(effective_config),
            "reference_time_id": series.reference_time_id,
            "status": series.status,
            "jump_warnings": _numpy_clean(series.jump_warnings),
            "manual_selections": dict(configured.manual_selections),
            "series": _numpy_clean(series.summary()),
            "outputs": {
                "version_dir": str(version_dir),
                "series_state": str(version_dir / "series_state.json"),
            },
        }
        report["items"].append(item_report)
        (version_dir / "series_report.json").write_text(
            json.dumps(item_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        versions_path = output_dir / item.name / "versions.json"
        ledger: list[Any] = []
        if versions_path.exists():
            ledger = json.loads(versions_path.read_text(encoding="utf-8"))
        ledger.append(
            {
                "version_id": version_id,
                "effective_parameter_hash": item_report["effective_parameter_hash"],
                "status": series.status,
                "selections": state["selections"],
                "jump_warning_count": len(series.jump_warnings),
                "created_utc": state["created_utc"],
            }
        )
        versions_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    report["batch_parameter_hash"] = parameter_hash(
        [item["effective_parameters"] for item in report["items"]]
    )
    (output_dir / "series_batch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def load_series_state(version_dir: str | Path) -> dict[str, Any]:
    """Reload a persisted series version and verify its internal correspondence.

    The returned mapping links every time id to its original image, the
    selected candidate id and the selected rigid transform. A mismatch
    between selections, transforms and stored originals raises ValueError.
    """

    version_dir = Path(version_dir)
    state_path = version_dir / "series_state.json"
    if not state_path.exists():
        raise ValueError(f"no series_state.json under {version_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    times = state["times"]
    loaded: dict[str, Any] = {
        "reference_time_id": state["reference_time_id"],
        "selections": dict(state["selections"]),
        "times": {},
    }
    for time_id, entry in times.items():
        moving_path = Path(entry["moving_original"])
        if not moving_path.exists():
            raise ValueError(f"original image for time {time_id} is missing: {moving_path}")
        image = MedicalImage.load_nifti(moving_path)
        selected_id = entry["selected_candidate_id"]
        if state["selections"].get(time_id) != selected_id:
            raise ValueError(f"selection mismatch for time {time_id}")
        candidate_ids = [candidate["candidate_id"] for candidate in entry["candidates"]]
        if selected_id is not None and selected_id not in candidate_ids:
            raise ValueError(
                f"selected candidate {selected_id} is not among retained candidates at {time_id}"
            )
        transform = None
        params = entry["selected_transform_parameters"]
        if params is not None:
            transform = RigidTransform.from_parameters(np.asarray(params, dtype=float))
        if entry["selected_outputs"] is not None:
            transformed_path = Path(entry["selected_outputs"]["transformed"])
            if not transformed_path.exists():
                raise ValueError(f"selected transformed image missing for time {time_id}")
        loaded["times"][time_id] = {
            "image": image,
            "status": entry["status"],
            "selected_candidate_id": selected_id,
            "transform": transform,
            "candidate_ids": candidate_ids,
        }
    return loaded
