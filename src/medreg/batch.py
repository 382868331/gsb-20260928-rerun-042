from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .image import MedicalImage
from .multi import InitializationSpec, register_multi
from .roi import RegionOfInterest


@dataclass(frozen=True)
class BatchItem:
    name: str
    fixed_path: str | None = None
    moving_path: str | None = None
    initial_parameters: list[float] | None = None
    initializations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    levels: tuple[int, ...] = (4, 2, 1)
    max_iterations: int = 120
    min_coverage: float = 0.35
    roi_config: dict[str, Any] | None = None
    min_roi_coverage: float = 0.6
    min_roi_ncc: float = 0.75
    roi_weight: float = 2.0
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


def _build_initializations(configured: BatchItem) -> list[InitializationSpec]:
    specs: list[InitializationSpec] = []
    if configured.initializations:
        for index, entry in enumerate(configured.initializations):
            candidate_id = str(entry.get("candidate_id", f"candidate_{index}"))
            specs.append(
                InitializationSpec(
                    candidate_id=candidate_id,
                    initial_parameters=[float(v) for v in entry["initial_parameters"]],
                    label=str(entry.get("label", "")),
                    source=str(entry.get("source", "automatic")),
                    lock_initial=bool(entry.get("lock_initial", False)),
                )
            )
    else:
        initial = configured.initial_parameters if configured.initial_parameters is not None else [0.0] * 6
        specs.append(InitializationSpec("identity", list(initial), "identity start", "automatic"))
    return specs


def _apply_override(item: BatchItem, override: dict[str, Any]) -> BatchItem:
    return BatchItem(
        item.name,
        item.fixed_path,
        item.moving_path,
        override.get("initial_parameters", item.initial_parameters),
        tuple(override.get("initializations", item.initializations)),
        tuple(override.get("levels", item.levels)),
        int(override.get("max_iterations", item.max_iterations)),
        float(override.get("min_coverage", item.min_coverage)),
        override.get("roi", override.get("roi_config", item.roi_config)),
        float(override.get("min_roi_coverage", item.min_roi_coverage)),
        float(override.get("min_roi_ncc", item.min_roi_ncc)),
        float(override.get("roi_weight", item.roi_weight)),
        override.get("manual_candidate_id", item.manual_candidate_id),
    )


def _load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": "medreg-version-registry-v1", "versions": []}


def run_batch(
    items: list[tuple[BatchItem, MedicalImage, MedicalImage]],
    output_dir: str | Path,
    parameter_file: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    versions_root = output_dir / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)
    registry_path = versions_root / "versions.json"
    registry = _load_registry(registry_path)

    provided_parameters = None
    parameter_source = str(parameter_file) if parameter_file else "inline"
    if parameter_file:
        parameter_file = Path(parameter_file)
        provided_parameters = json.loads(parameter_file.read_text(encoding="utf-8"))
        parameter_source = f"{parameter_source}@{parameter_hash(provided_parameters)}"

    version_index = len(registry["versions"]) + 1
    version_name = f"v{version_index:04d}"
    previous_hash = registry["versions"][-1]["batch_parameter_hash"] if registry["versions"] else None
    version_dir = versions_root / version_name

    report: dict[str, Any] = {
        "schema": "medreg-batch-v2",
        "version": version_name,
        "parameter_source": parameter_source,
        "items": [],
    }
    # Validate every item before writing a new version: a rejected manual
    # selection must never leave a partial version directory behind.
    prepared: list[tuple[BatchItem, RegionOfInterest | None, list[InitializationSpec], dict[str, Any]]] = []
    for item, fixed, moving in items:
        configured = item
        if provided_parameters is not None and item.name in provided_parameters.get("items", {}):
            configured = _apply_override(item, provided_parameters["items"][item.name])
        roi = RegionOfInterest.from_config(configured.roi_config) if configured.roi_config else None
        initializations = _build_initializations(configured)
        effective_config = _numpy_clean(
            asdict(configured)
            | {
                "levels": list(configured.levels),
                "initializations": [
                    {
                        "candidate_id": spec.candidate_id,
                        "initial_parameters": spec.initial_parameters,
                        "label": spec.label,
                        "source": spec.source,
                        "lock_initial": spec.lock_initial,
                    }
                    for spec in initializations
                ],
            }
        )
        if configured.manual_candidate_id is not None:
            valid_ids = {spec.candidate_id for spec in initializations}
            if configured.manual_candidate_id not in valid_ids:
                raise ValueError(f"manual candidate {configured.manual_candidate_id!r} does not exist")
        prepared.append((configured, roi, initializations, effective_config))

    results: list[Any] = []
    for (_, fixed, moving), (configured, roi, initializations, _) in zip(items, prepared):
        results.append(
            register_multi(
                fixed,
                moving,
                initializations,
                levels=configured.levels,
                max_iterations=configured.max_iterations,
                min_coverage=configured.min_coverage,
                roi=roi,
                min_roi_coverage=configured.min_roi_coverage,
                min_roi_ncc=configured.min_roi_ncc,
                roi_weight=configured.roi_weight,
                manual_candidate_id=configured.manual_candidate_id,
            )
        )

    for (item, fixed, moving), (configured, roi, initializations, effective_config), multi in zip(
        items, prepared, results
    ):
        selected = multi.selected_candidate()
        item_dir = version_dir / item.name
        item_dir.mkdir(parents=True, exist_ok=True)
        fixed.save_nifti(item_dir / "fixed_original.nii.gz")
        moving.save_nifti(item_dir / "moving_original.nii.gz")
        outputs: dict[str, Any] = {
            "fixed": str(item_dir / "fixed_original.nii.gz"),
            "moving": str(item_dir / "moving_original.nii.gz"),
            "transformed": None,
            "overlap_mask": None,
            "roi_mask": None,
        }
        if roi is not None:
            roi_mask_path = item_dir / "roi_mask.npy"
            np.save(roi_mask_path, roi.mask(fixed))
            outputs["roi_mask"] = str(roi_mask_path)
        # Only the single selected, acceptance-passing candidate may emit a
        # transformed image. Failed and merely retained candidates never do.
        if selected is not None and selected.result.transformed is not None and selected.result.overlap_mask is not None:
            selected.result.transformed.save_nifti(item_dir / "moving_transformed.nii.gz")
            np.save(item_dir / "overlap_mask.npy", selected.result.overlap_mask)
            outputs["transformed"] = str(item_dir / "moving_transformed.nii.gz")
            outputs["overlap_mask"] = str(item_dir / "overlap_mask.npy")

        item_report = {
            "name": item.name,
            "effective_parameters": effective_config,
            "effective_parameter_hash": parameter_hash(effective_config),
            "selected_candidate_id": multi.selected_candidate_id,
            "outputs": outputs,
            "result": multi.summary(),
        }
        report["items"].append(_numpy_clean(item_report))

    report["batch_parameter_hash"] = parameter_hash([item["effective_parameters"] for item in report["items"]])
    manual_selection = any(item["effective_parameters"].get("manual_candidate_id") for item in report["items"])
    if previous_hash is None:
        trigger = "initial"
    elif manual_selection:
        trigger = "manual_selection"
    elif report["batch_parameter_hash"] != previous_hash:
        trigger = "parameter_change"
    else:
        trigger = "rerun"
    version_dir.mkdir(parents=True, exist_ok=True)
    report_path = version_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    registry["versions"].append(
        {
            "version": version_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "selection_mode": "manual" if manual_selection else "automatic",
            "parameter_source": parameter_source,
            "batch_parameter_hash": report["batch_parameter_hash"],
            "report": str(report_path),
        }
    )
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "LATEST").write_text(version_name + "\n", encoding="utf-8")
    report["version_registry"] = str(registry_path)
    return report
