from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .image import MedicalImage
from .register import RegistrationResult, register_rigid
from .transform import RigidTransform


@dataclass(frozen=True)
class BatchItem:
    name: str
    fixed_path: str | None = None
    moving_path: str | None = None
    initial_parameters: list[float] | None = None
    levels: tuple[int, ...] = (4, 2, 1)
    max_iterations: int = 120
    min_coverage: float = 0.35


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


def run_batch(
    items: list[tuple[BatchItem, MedicalImage, MedicalImage]],
    output_dir: str | Path,
    parameter_file: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provided_parameters = None
    parameter_source = str(parameter_file) if parameter_file else "inline"
    if parameter_file:
        parameter_file = Path(parameter_file)
        provided_parameters = json.loads(parameter_file.read_text(encoding="utf-8"))
        parameter_source = f"{parameter_source}@{parameter_hash(provided_parameters)}"

    report: dict[str, Any] = {
        "schema": "medreg-batch-v1",
        "parameter_source": parameter_source,
        "items": [],
    }
    for item, fixed, moving in items:
        configured = item
        if provided_parameters is not None and item.name in provided_parameters.get("items", {}):
            override = provided_parameters["items"][item.name]
            configured = BatchItem(
                item.name,
                item.fixed_path,
                item.moving_path,
                override.get("initial_parameters", item.initial_parameters),
                tuple(override.get("levels", item.levels)),
                int(override.get("max_iterations", item.max_iterations)),
                float(override.get("min_coverage", item.min_coverage)),
            )
        initial = (
            RigidTransform.from_parameters(np.asarray(configured.initial_parameters, dtype=float))
            if configured.initial_parameters is not None
            else None
        )
        effective_config = _numpy_clean(asdict(configured) | {"levels": list(configured.levels)})
        result = register_rigid(
            fixed,
            moving,
            initial,
            configured.levels,
            configured.max_iterations,
            configured.min_coverage,
        )
        item_dir = output_dir / item.name
        item_dir.mkdir(parents=True, exist_ok=True)
        fixed.save_nifti(item_dir / "fixed_original.nii.gz")
        moving.save_nifti(item_dir / "moving_original.nii.gz")
        if result.transformed is not None:
            result.transformed.save_nifti(item_dir / "moving_transformed.nii.gz")
            np.save(item_dir / "overlap_mask.npy", result.overlap_mask)
        item_report = {
            "name": item.name,
            "effective_parameters": effective_config,
            "effective_parameter_hash": parameter_hash(effective_config),
            "outputs": {
                "fixed": str(item_dir / "fixed_original.nii.gz"),
                "moving": str(item_dir / "moving_original.nii.gz"),
                "transformed": str(item_dir / "moving_transformed.nii.gz") if result.transformed else None,
                "overlap_mask": str(item_dir / "overlap_mask.npy") if result.transformed else None,
            },
            "result": result.summary(),
        }
        report["items"].append(_numpy_clean(item_report))
    report["batch_parameter_hash"] = parameter_hash([item["effective_parameters"] for item in report["items"]])
    report_path = output_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
