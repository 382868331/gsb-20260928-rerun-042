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
from .transform import RigidTransform


@dataclass(frozen=True)
class SeriesConfig:
    """Shared settings for a temporal registration series.

    Every timepoint is registered independently against the user-chosen
    reference timepoint, so one failed acquisition never changes the
    candidates, coverage, or errors already obtained for the others.
    """

    reference_index: int = 0
    levels: tuple[int, ...] = (4, 2, 1)
    max_iterations: int = 120
    min_coverage: float = 0.35
    min_roi_coverage: float = 0.6
    min_roi_ncc: float = 0.7
    jump_translation_mm: float = 5.0
    jump_rotation_deg: float = 8.0


@dataclass
class TimepointResult:
    """Per-timepoint outcome; candidates and metrics are never shared."""

    index: int
    label: str
    is_reference: bool
    status: str  # reference | success | uncertain | failed | rejected
    multi: MultiStartResult | None
    rejection_reasons: list[str] = field(default_factory=list)
    roi_diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def selected_transform(self) -> RigidTransform | None:
        if self.is_reference:
            return RigidTransform()
        if self.multi is not None and self.multi.selected is not None:
            return self.multi.selected.result.transform
        return None

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "label": self.label,
            "is_reference": self.is_reference,
            "status": self.status,
            "rejection_reasons": list(self.rejection_reasons),
            "roi_diagnostics": self.roi_diagnostics,
        }
        if self.multi is not None:
            payload["multi_start"] = self.multi.summary()
            selected = self.multi.selected
            payload["selected_candidate_id"] = selected.candidate_id if selected else None
            payload["selection_method"] = self.multi.selection_method
        else:
            payload["selected_candidate_id"] = None
            payload["selection_method"] = None
        return payload


@dataclass
class SeriesResult:
    """A whole temporal series with per-timepoint results and adjacency checks."""

    config: SeriesConfig
    labels: list[str]
    timepoints: list[TimepointResult]
    adjacency: list[dict[str, object]]
    roi_summary: dict[str, object] | None

    @property
    def status(self) -> str:
        statuses = {tp.status for tp in self.timepoints}
        if statuses <= {"reference", "success"}:
            return "success"
        if "failed" in statuses or "rejected" in statuses:
            return "partial" if statuses & {"success", "uncertain"} else "failed"
        return "uncertain"

    def timepoint(self, label: str) -> TimepointResult:
        for tp in self.timepoints:
            if tp.label == label:
                return tp
        raise KeyError(f"unknown timepoint label {label!r}")

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reference_index": self.config.reference_index,
            "reference_label": self.labels[self.config.reference_index],
            "labels": list(self.labels),
            "roi": self.roi_summary,
            "adjacency": self.adjacency,
            "timepoints": [tp.summary() for tp in self.timepoints],
        }


def _roi_diagnostics(
    fixed: MedicalImage, roi: RegionOfInterest | None
) -> tuple[dict[str, object], list[str]]:
    """Reference-side ROI checks shared by every timepoint.

    A region with no voxels or no image information cannot support any
    conclusion, so the whole series is rejected explicitly instead of
    drawing a smooth but meaningless curve.
    """

    if roi is None:
        return {}, []
    mask = roi.mask_on(fixed)
    voxels = int(mask.sum())
    diagnostics: dict[str, object] = {
        "roi_voxels_on_reference": voxels,
        "world_min": list(roi.world_min),
        "world_max": list(roi.world_max),
    }
    problems: list[str] = []
    if voxels == 0:
        problems.append("roi_empty_on_reference: the region selects no voxels of the reference image")
        diagnostics["roi_fixed_variance"] = 0.0
        return diagnostics, problems
    variance = float(np.var(fixed.data[mask]))
    diagnostics["roi_fixed_variance"] = variance
    if variance < 1e-12:
        problems.append(
            "roi_has_no_image_information: the reference image is flat inside the region of interest"
        )
    return diagnostics, problems


def _adjacency_report(
    timepoints: list[TimepointResult], config: SeriesConfig
) -> list[dict[str, object]]:
    """Compare selected transforms of neighbouring timepoints.

    Failed or rejected timepoints produce an explicit gap entry; the series
    is never interpolated across them to fake a continuous trajectory.
    """

    report: list[dict[str, object]] = []
    for first, second in zip(timepoints, timepoints[1:]):
        entry: dict[str, object] = {"between": [first.label, second.label]}
        first_transform = first.selected_transform
        second_transform = second.selected_transform
        if first_transform is None or second_transform is None:
            missing = [
                tp.label
                for tp, transform in ((first, first_transform), (second, second_transform))
                if transform is None
            ]
            entry["type"] = "gap"
            entry["message"] = (
                f"no reliable transform for {', '.join(missing)}; the trajectory is broken here "
                "and no values are interpolated across the gap"
            )
            report.append(entry)
            continue
        disagreement = transform_disagreement(first_transform, second_transform)
        entry["type"] = "comparison"
        entry["disagreement"] = disagreement
        jumped = (
            disagreement["translation_distance_mm"] > config.jump_translation_mm
            or disagreement["rotation_angle_deg"] > config.jump_rotation_deg
        )
        entry["jump"] = bool(jumped)
        if jumped:
            entry["message"] = (
                f"adjacent timepoints {first.label} -> {second.label} change by "
                f"{disagreement['translation_distance_mm']:.2f} mm and "
                f"{disagreement['rotation_angle_deg']:.2f} deg, exceeding "
                f"{config.jump_translation_mm:.2f} mm / {config.jump_rotation_deg:.2f} deg"
            )
        report.append(entry)
    return report


def register_series(
    images: list[MedicalImage],
    labels: list[str] | None = None,
    config: SeriesConfig | None = None,
    roi: RegionOfInterest | None = None,
    initial_sets_per_timepoint: dict[int, list[list[float]]] | None = None,
) -> SeriesResult:
    """Register every timepoint of a temporal series to the reference timepoint.

    Each timepoint keeps its own candidates, coverage, and error metrics; a
    failed timepoint is recorded as a gap and never drags down or rewrites
    the results of the others.
    """

    if not images:
        raise ValueError("a series needs at least one timepoint")
    config = config or SeriesConfig()
    if not 0 <= config.reference_index < len(images):
        raise ValueError("reference_index is outside the series")
    labels = labels or [f"t{index:02d}" for index in range(len(images))]
    if len(labels) != len(images):
        raise ValueError("labels must match the number of timepoints")
    if len(set(labels)) != len(labels):
        raise ValueError("timepoint labels must be unique")
    initial_sets_per_timepoint = initial_sets_per_timepoint or {}

    fixed = images[config.reference_index]
    roi_diagnostics, roi_problems = _roi_diagnostics(fixed, roi)
    roi_summary = None
    if roi is not None:
        roi_summary = roi.summary(fixed)
        roi_summary["diagnostics"] = roi_diagnostics
        roi_summary["problems"] = list(roi_problems)

    timepoints: list[TimepointResult] = []
    for index, image in enumerate(images):
        if index == config.reference_index:
            timepoints.append(
                TimepointResult(
                    index=index,
                    label=labels[index],
                    is_reference=True,
                    status="reference",
                    multi=None,
                    roi_diagnostics=dict(roi_diagnostics),
                )
            )
            continue
        if roi_problems:
            timepoints.append(
                TimepointResult(
                    index=index,
                    label=labels[index],
                    is_reference=False,
                    status="rejected",
                    multi=None,
                    rejection_reasons=list(roi_problems),
                    roi_diagnostics=dict(roi_diagnostics),
                )
            )
            continue
        initial_sets = initial_sets_per_timepoint.get(index) or [[0.0] * 6]
        multi = register_multi(
            fixed,
            image,
            initial_sets,
            roi,
            config.levels,
            config.max_iterations,
            config.min_coverage,
            config.min_roi_coverage,
            config.min_roi_ncc,
        )
        rejection: list[str] = []
        status = multi.status
        if multi.selected is None:
            rejection.append(
                "no candidate passed acceptance at this timepoint; other timepoints are unaffected"
            )
        elif roi is not None and multi.selected.result.roi_error is not None:
            roi_coverage = float(multi.selected.result.roi_error["region_coverage_fraction"])
            if roi_coverage < config.min_roi_coverage:
                status = "rejected"
                rejection.append(
                    f"roi_coverage_insufficient: selected candidate covers "
                    f"{roi_coverage:.3f} of the region, below {config.min_roi_coverage:.3f}"
                )
        timepoints.append(
            TimepointResult(
                index=index,
                label=labels[index],
                is_reference=False,
                status=status,
                multi=multi,
                rejection_reasons=rejection,
                roi_diagnostics=dict(roi_diagnostics),
            )
        )

    adjacency = _adjacency_report(timepoints, config)
    return SeriesResult(config, list(labels), timepoints, adjacency, roi_summary)


def select_series_candidate(
    series: SeriesResult, label: str, candidate_id: str
) -> SeriesResult:
    """Apply a manual candidate choice at one timepoint and recompute the series.

    Only the targeted timepoint changes; all other timepoints keep their
    candidates and selections. Adjacency warnings are recomputed so the
    whole trajectory can be reviewed again after the manual choice.
    """

    target = series.timepoint(label)
    if target.is_reference:
        raise ValueError("the reference timepoint has no candidates to select from")
    if target.multi is None:
        raise ValueError(
            f"timepoint {label!r} was rejected before registration ({target.rejection_reasons}); "
            "there are no candidates to select"
        )
    updated_multi = apply_manual_selection(target.multi, candidate_id)
    new_timepoints: list[TimepointResult] = []
    for tp in series.timepoints:
        if tp is target:
            new_timepoints.append(
                TimepointResult(
                    index=tp.index,
                    label=tp.label,
                    is_reference=False,
                    status=updated_multi.status,
                    multi=updated_multi,
                    rejection_reasons=list(tp.rejection_reasons),
                    roi_diagnostics=dict(tp.roi_diagnostics),
                )
            )
        else:
            new_timepoints.append(tp)
    adjacency = _adjacency_report(new_timepoints, series.config)
    return SeriesResult(series.config, list(series.labels), new_timepoints, adjacency, series.roi_summary)

