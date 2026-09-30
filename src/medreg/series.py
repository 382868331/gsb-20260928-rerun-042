from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .image import MedicalImage
from .register import (
    MultiStartResult,
    apply_manual_selection,
    register_multi,
    transform_disagreement,
)
from .roi import RegionOfInterest


@dataclass
class SeriesTimePoint:
    """Registration outcome for one acquisition time in a series.

    Every time point keeps its own candidate list, coverage and error
    diagnostics; a failed or rejected time point never removes or alters
    the reliable results obtained at other times.
    """

    time_id: str
    is_reference: bool
    multi: MultiStartResult | None
    status: str
    rejection_reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "time_id": self.time_id,
            "is_reference": self.is_reference,
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
            "multi_start": self.multi.summary() if self.multi is not None else None,
        }


@dataclass
class SeriesResult:
    """Ordered time-series registration against a user-chosen reference time."""

    reference_time_id: str
    time_points: list[SeriesTimePoint]
    jump_warnings: list[dict[str, object]]
    status: str

    def time_point(self, time_id: str) -> SeriesTimePoint:
        for point in self.time_points:
            if point.time_id == time_id:
                return point
        raise KeyError(f"unknown time_id {time_id!r}")

    def summary(self) -> dict[str, object]:
        return {
            "reference_time_id": self.reference_time_id,
            "status": self.status,
            "jump_warnings": self.jump_warnings,
            "time_points": [point.summary() for point in self.time_points],
        }


def _roi_rejection_reasons(
    fixed: MedicalImage,
    roi: RegionOfInterest | None,
    min_roi_voxels: int = 8,
) -> list[str]:
    """Explicit rejection when the selected region cannot support a conclusion.

    Local signal loss, missing coverage or a completely uninformative region
    must refuse a conclusion instead of hiding behind a smooth curve.
    """

    if roi is None:
        return []
    reasons: list[str] = []
    mask = roi.mask_on(fixed)
    voxels = int(mask.sum())
    if voxels == 0:
        reasons.append("region of interest selects no voxels on the reference grid")
        return reasons
    if voxels < min_roi_voxels:
        reasons.append(f"region of interest covers only {voxels} voxels on the reference grid")
    values = fixed.data[mask]
    if float(np.std(values)) <= 1e-6:
        reasons.append("region of interest contains no image information (constant reference data)")
    return reasons


def compute_jump_warnings(
    time_points: list[SeriesTimePoint],
    jump_translation_mm: float = 5.0,
    jump_rotation_deg: float = 8.0,
) -> list[dict[str, object]]:
    """Flag adjacent time points whose selected transforms change too much.

    Time points without a reliable selected transform are skipped and the
    gap is reported, so a missing result is never smoothed over.
    """

    from .transform import RigidTransform

    warnings: list[dict[str, object]] = []
    previous_id: str | None = None
    previous_transform: RigidTransform | None = None
    for point in time_points:
        if point.is_reference:
            # The reference time is the identity anchor of the series.
            previous_id = point.time_id
            previous_transform = RigidTransform()
            continue
        selected = point.multi.selected if point.multi is not None else None
        transform = selected.result.transform if selected is not None else None
        if transform is None:
            warnings.append(
                {
                    "from_time_id": previous_id,
                    "to_time_id": point.time_id,
                    "kind": "missing_result",
                    "message": (
                        f"no reliable transform at {point.time_id}; the series is "
                        "interrupted here instead of being interpolated"
                    ),
                }
            )
            previous_id = None
            previous_transform = None
            continue
        if previous_transform is not None:
            disagreement = transform_disagreement(previous_transform, transform)
            if (
                disagreement["translation_distance_mm"] > jump_translation_mm
                or disagreement["rotation_angle_deg"] > jump_rotation_deg
            ):
                warnings.append(
                    {
                        "from_time_id": previous_id,
                        "to_time_id": point.time_id,
                        "kind": "large_change",
                        "translation_distance_mm": disagreement["translation_distance_mm"],
                        "rotation_angle_deg": disagreement["rotation_angle_deg"],
                        "message": (
                            f"selected transform changes by "
                            f"{disagreement['translation_distance_mm']:.2f} mm and "
                            f"{disagreement['rotation_angle_deg']:.2f} deg between "
                            f"{previous_id} and {point.time_id}"
                        ),
                    }
                )
        previous_id = point.time_id
        previous_transform = transform
    return warnings


def _series_status(time_points: list[SeriesTimePoint]) -> str:
    non_reference = [point for point in time_points if not point.is_reference]
    if not non_reference:
        return "failed"
    reliable = sum(1 for point in non_reference if point.status in {"success", "uncertain"})
    if reliable == 0:
        return "failed"
    if reliable < len(non_reference):
        return "partial"
    return "success"


def register_series(
    images: dict[str, MedicalImage],
    reference_time_id: str,
    initial_parameters: list[list[float]] | None = None,
    roi: RegionOfInterest | None = None,
    levels: tuple[int, ...] = (4, 2, 1),
    max_iterations: int = 120,
    min_coverage: float = 0.35,
    min_roi_coverage: float = 0.6,
    min_roi_ncc: float = 0.7,
    jump_translation_mm: float = 5.0,
    jump_rotation_deg: float = 8.0,
    time_order: list[str] | None = None,
) -> SeriesResult:
    """Register every acquisition time to the user-chosen reference time.

    Each time point is processed independently and keeps its own candidates,
    coverage and errors. Time points whose region of interest is missing,
    under-covered or uninformative are explicitly rejected.
    """

    if reference_time_id not in images:
        raise ValueError(f"reference time {reference_time_id!r} is not present in the series")
    order = list(time_order) if time_order is not None else list(images.keys())
    missing = [time_id for time_id in order if time_id not in images]
    if missing:
        raise ValueError(f"time_order references unknown time ids: {missing}")
    fixed = images[reference_time_id]

    roi_reasons = _roi_rejection_reasons(fixed, roi)
    time_points: list[SeriesTimePoint] = []
    for time_id in order:
        if time_id == reference_time_id:
            time_points.append(
                SeriesTimePoint(time_id, True, None, "reference", [])
            )
            continue
        if roi_reasons:
            time_points.append(
                SeriesTimePoint(
                    time_id,
                    False,
                    None,
                    "rejected",
                    list(roi_reasons),
                )
            )
            continue
        multi = register_multi(
            fixed,
            images[time_id],
            initial_parameters,
            roi,
            levels,
            max_iterations,
            min_coverage,
            min_roi_coverage,
            min_roi_ncc,
        )
        reasons: list[str] = []
        status = multi.status
        if status == "failed":
            reasons.extend(multi.uncertainty_reasons)
        if roi is not None and multi.selected is not None and multi.selected.result.roi_error is not None:
            roi_error = multi.selected.result.roi_error
            if float(roi_error["region_coverage_fraction"]) < 1.0:
                reasons.append(
                    "region of interest is not fully covered at this time: "
                    f"{float(roi_error['region_coverage_fraction']):.3f}"
                )
        time_points.append(SeriesTimePoint(time_id, False, multi, status, reasons))

    series = SeriesResult(
        reference_time_id,
        time_points,
        [],
        "failed",
    )
    series.jump_warnings = compute_jump_warnings(
        time_points, jump_translation_mm, jump_rotation_deg
    )
    series.status = _series_status(time_points)
    return series


def apply_series_manual_selection(
    series: SeriesResult,
    time_id: str,
    candidate_id: str,
    jump_translation_mm: float = 5.0,
    jump_rotation_deg: float = 8.0,
) -> SeriesResult:
    """Promote a retained candidate at one time and re-review the whole series.

    Geometry is never re-run; the jump warnings across the full time course
    are recomputed so the expert immediately sees the effect of the choice.
    """

    point = series.time_point(time_id)
    if point.is_reference or point.multi is None:
        raise ValueError(f"time {time_id!r} has no candidates to select from")
    updated_multi = apply_manual_selection(point.multi, candidate_id)
    new_points: list[SeriesTimePoint] = []
    for current in series.time_points:
        if current.time_id == time_id:
            status = "success" if not updated_multi.uncertain else "uncertain"
            new_points.append(
                SeriesTimePoint(time_id, False, updated_multi, status, list(current.rejection_reasons))
            )
        else:
            new_points.append(current)
    updated = SeriesResult(
        series.reference_time_id,
        new_points,
        compute_jump_warnings(new_points, jump_translation_mm, jump_rotation_deg),
        _series_status(new_points),
    )
    return updated
