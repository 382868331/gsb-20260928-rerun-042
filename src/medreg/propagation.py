from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .image import MedicalImage
from .series import SeriesResult, select_series_candidate
from .transform import RigidTransform


@dataclass(frozen=True)
class PropagationConfig:
    """Acceptance settings for propagating a drawn region across a series.

    A target timepoint only receives a usable contour when its adopted
    registration candidate maps enough of the source voxels into validly
    imaged territory; anything else stays explicitly pending.
    """

    min_region_coverage: float = 0.6


@dataclass
class RegionRecord:
    """Per-timepoint region outcome with full adoption trace."""

    label: str
    status: str  # source | propagated | manual | pending
    mask: np.ndarray | None
    coverage_fraction: float | None
    basis: dict[str, object] | None
    decision: str  # original | auto | manual | none
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "label": self.label,
            "status": self.status,
            "decision": self.decision,
            "coverage_fraction": self.coverage_fraction,
            "voxels": int(self.mask.sum()) if self.mask is not None else 0,
            "basis": self.basis,
            "reasons": list(self.reasons),
        }


@dataclass
class Correction:
    """A manual edit at one timepoint; never mutates the original drawing."""

    label: str
    mask: np.ndarray
    use_as_basis: bool


def _warp_mask(
    basis_mask: np.ndarray,
    reference: MedicalImage,
    relative: RigidTransform,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """Warp a reference-grid mask through a rigid transform.

    Returns the warped mask on the reference grid plus the fraction of
    source voxels that land inside validly imaged territory. Missing
    imaging is never filled in to complete the contour.
    """

    coords = np.column_stack(np.nonzero(basis_mask))
    if coords.size == 0:
        return np.zeros(reference.shape, dtype=bool), 0.0
    world = reference.index_to_world(coords.T.astype(float))
    moved = relative.apply(world.T)
    target_index = reference.world_to_index(moved.T)
    nearest = np.rint(target_index).astype(int)
    shape = np.asarray(reference.shape)[:, None]
    inside = np.all((nearest >= 0) & (nearest < shape), axis=0)
    if valid_mask is not None:
        valid = np.zeros(inside.shape, dtype=bool)
        valid[inside] = valid_mask[
            nearest[0][inside], nearest[1][inside], nearest[2][inside]
        ]
        inside &= valid
    out = np.zeros(reference.shape, dtype=bool)
    kept = nearest[:, inside]
    out[kept[0], kept[1], kept[2]] = True
    coverage_fraction = float(inside.sum() / coords.shape[0])
    return out, coverage_fraction


class PropagationSession:
    """Interactive region propagation over a registered temporal series.

    All region masks live on the reference-image grid (the same grid the
    transformed timepoint images are resampled to), so a mask at any
    timepoint is directly comparable with the original drawing.
    """

    def __init__(
        self,
        series: SeriesResult,
        images: list[MedicalImage],
        source_label: str | None = None,
        config: PropagationConfig | None = None,
    ):
        self.series = series
        self.images = list(images)
        self.config = config or PropagationConfig()
        self.reference = self.images[series.config.reference_index]
        self.source_label = source_label or series.labels[series.config.reference_index]
        if self.source_label not in series.labels:
            raise ValueError(f"unknown source label {self.source_label!r}")
        self._original_mask: np.ndarray | None = None
        self._corrections: dict[str, Correction] = {}
        self.records: dict[str, RegionRecord] = {}

    @property
    def original_mask(self) -> np.ndarray | None:
        return None if self._original_mask is None else self._original_mask.copy()

    @property
    def corrections(self) -> dict[str, Correction]:
        return dict(self._corrections)

    def draw_region(self, mask: np.ndarray) -> None:
        """Record the one original hand-drawn region at the source timepoint."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.reference.shape:
            raise ValueError("drawn region must match the reference image grid")
        if not mask.any():
            raise ValueError("the drawn region is empty")
        if self._original_mask is not None:
            raise ValueError(
                "the original drawing already exists; corrections go through "
                "apply_correction and never overwrite the original"
            )
        self._original_mask = mask
        self._recompute()

    def _transform_for(self, label: str) -> RigidTransform | None:
        return self.series.timepoint(label).selected_transform

    def _overlap_for(self, label: str) -> np.ndarray | None:
        tp = self.series.timepoint(label)
        if tp.multi is not None and tp.multi.selected is not None:
            return np.asarray(tp.multi.selected.result.overlap_mask, dtype=bool)
        return None

    def _basis_chain(self) -> dict[str, tuple[str, np.ndarray, str]]:
        """Resolve, per timepoint, which mask it propagates from.

        Timepoints at or after a correction marked as a new basis propagate
        from that correction; everything else propagates from the original
        drawing. Returns label -> (basis_label, basis_mask, basis_kind).
        """

        assert self._original_mask is not None
        order = self.series.labels
        basis_label = self.source_label
        basis_mask = self._original_mask
        basis_kind = "original"
        chain: dict[str, tuple[str, np.ndarray, str]] = {}
        for label in order:
            correction = self._corrections.get(label)
            if correction is not None and correction.use_as_basis and label != self.source_label:
                basis_label, basis_mask, basis_kind = label, correction.mask, "correction"
            chain[label] = (basis_label, basis_mask, basis_kind)
        chain[self.source_label] = (self.source_label, self._original_mask, "original")
        return chain

    def _recompute(self) -> None:
        if self._original_mask is None:
            self.records = {}
            return
        chain = self._basis_chain()
        source_transform = self._transform_for(self.source_label)
        records: dict[str, RegionRecord] = {}
        for label in self.series.labels:
            correction = self._corrections.get(label)
            if correction is not None:
                basis_label, _, _ = chain[label]
                records[label] = RegionRecord(
                    label=label,
                    status="manual",
                    mask=correction.mask.copy(),
                    coverage_fraction=None,
                    basis={
                        "adopted_as_basis": correction.use_as_basis,
                        "downstream_basis_label": basis_label if correction.use_as_basis else None,
                    },
                    decision="manual",
                )
                continue
            if label == self.source_label:
                records[label] = RegionRecord(
                    label=label,
                    status="source",
                    mask=self._original_mask.copy(),
                    coverage_fraction=1.0,
                    basis=None,
                    decision="original",
                )
                continue
            tp = self.series.timepoint(label)
            basis_label, basis_mask, basis_kind = chain[label]
            target_transform = self._transform_for(label)
            reasons: list[str] = []
            if target_transform is None or source_transform is None:
                if tp.status in ("failed", "rejected"):
                    reasons.append(
                        f"timepoint {label} is {tp.status}; no adopted transform exists, "
                        "so the region stays pending instead of being interpolated"
                    )
                else:
                    reasons.append(f"no adopted transform for {label} or {self.source_label}")
                records[label] = RegionRecord(
                    label=label, status="pending", mask=None, coverage_fraction=None,
                    basis={"basis_label": basis_label, "basis_kind": basis_kind},
                    decision="none", reasons=reasons,
                )
                continue
            relative = target_transform.compose(source_transform.inverse())
            candidate_id = (
                tp.multi.selected.candidate_id if tp.multi is not None and tp.multi.selected else None
            )
            valid = self._overlap_for(label)
            mask, coverage_fraction = _warp_mask(basis_mask, self.reference, relative, valid)
            basis = {
                "basis_label": basis_label,
                "basis_kind": basis_kind,
                "candidate_id": candidate_id,
                "selection_method": tp.multi.selection_method if tp.multi else None,
                "transform_parameters": relative.parameters().tolist(),
            }
            if coverage_fraction < self.config.min_region_coverage:
                reasons.append(
                    f"region_coverage_insufficient: only {coverage_fraction:.3f} of the region "
                    f"lands on validly imaged voxels, below {self.config.min_region_coverage:.3f}; "
                    "the uncovered part is left pending, not extrapolated"
                )
                records[label] = RegionRecord(
                    label=label, status="pending", mask=mask, coverage_fraction=coverage_fraction,
                    basis=basis, decision="none", reasons=reasons,
                )
                continue
            records[label] = RegionRecord(
                label=label, status="propagated", mask=mask, coverage_fraction=coverage_fraction,
                basis=basis, decision="auto",
            )
        self.records = records

    def apply_correction(self, label: str, mask: np.ndarray, use_as_basis: bool = False) -> None:
        """Store a manual correction at one timepoint and recompute dependents.

        The original drawing is never touched. When use_as_basis is set,
        later timepoints re-propagate from this correction; earlier ones are
        left exactly as they were.
        """
        if self._original_mask is None:
            raise ValueError("draw the original region before correcting")
        if label not in self.series.labels:
            raise ValueError(f"unknown timepoint label {label!r}")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.reference.shape:
            raise ValueError("correction must match the reference image grid")
        if not mask.any():
            raise ValueError("a correction cannot be empty; revert it instead")
        self._corrections[label] = Correction(label, mask, use_as_basis)
        self._recompute()

    def revert_correction(self, label: str) -> None:
        """Withdraw a correction; the automatic result is recomputed, the
        original drawing is never overwritten."""
        if label not in self._corrections:
            raise ValueError(f"no correction recorded at {label!r}")
        del self._corrections[label]
        self._recompute()

    def _dependents_of(self, label: str) -> list[str]:
        """Labels whose automatic region depends on this timepoint's transform."""
        chain = self._basis_chain()
        dependents: list[str] = []
        for other in self.series.labels:
            if other == label or other in self._corrections:
                continue
            record = self.records.get(other)
            if record is None or record.decision != "auto":
                continue
            basis_label = chain[other][0]
            if basis_label == label:
                dependents.append(other)
        if label in self.records and label not in self._corrections and label != self.source_label:
            if label not in dependents:
                dependents.append(label)
        return dependents

    def preview_candidate_change(self, label: str, candidate_id: str) -> dict[str, object]:
        """List what re-selecting a candidate would touch, before doing it.

        Reports which automatic results would be recomputed and which manual
        corrections sit at or downstream of the change and would conflict
        with the new geometry.
        """
        tp = self.series.timepoint(label)
        if tp.multi is None:
            raise ValueError(f"timepoint {label!r} has no candidates to re-select")
        known = {candidate.candidate_id for candidate in tp.multi.candidates}
        if candidate_id not in known:
            raise ValueError(f"unknown candidate {candidate_id!r} at {label!r}")
        auto_updates = self._dependents_of(label)
        chain = self._basis_chain()
        manual_conflicts = []
        for correction_label, correction in self._corrections.items():
            if correction_label == label:
                manual_conflicts.append(correction_label)
                continue
            basis_label = chain[correction_label][0]
            if basis_label == label or (
                correction.use_as_basis and correction_label in auto_updates
            ):
                manual_conflicts.append(correction_label)
        return {
            "label": label,
            "candidate_id": candidate_id,
            "auto_results_to_update": auto_updates,
            "manual_conflicts": manual_conflicts,
            "manual_kept": list(self._corrections),
        }

    def apply_candidate_change(self, label: str, candidate_id: str) -> dict[str, object]:
        """Re-select a candidate, recompute automatic results, keep manual ones.

        Manual corrections are preserved untouched; only automatic
        propagations are recomputed against the new adopted transform.
        """
        preview = self.preview_candidate_change(label, candidate_id)
        self.series = select_series_candidate(self.series, label, candidate_id)
        self._recompute()
        return preview

    def change_curve(self) -> list[dict[str, object]]:
        """Per-timepoint region size, tagged by how the region was obtained.

        Pending timepoints contribute an explicit gap (volume_mm3=None);
        the curve is never interpolated across missing imaging.
        """
        voxel_mm3 = float(np.prod(self.reference.spacing))
        curve: list[dict[str, object]] = []
        for label in self.series.labels:
            record = self.records.get(label)
            if record is None or record.status == "pending" or record.mask is None:
                curve.append({
                    "label": label,
                    "origin": "pending",
                    "volume_mm3": None,
                    "coverage_fraction": record.coverage_fraction if record else None,
                    "reasons": list(record.reasons) if record else ["no region drawn yet"],
                })
                continue
            curve.append({
                "label": label,
                "origin": {"source": "original", "propagated": "auto", "manual": "manual"}[record.status],
                "volume_mm3": float(record.mask.sum()) * voxel_mm3,
                "coverage_fraction": record.coverage_fraction,
                "reasons": [],
            })
        return curve

    def lineage(self) -> list[dict[str, object]]:
        """Visible propagation relations: who each region was derived from."""
        edges: list[dict[str, object]] = []
        for label in self.series.labels:
            record = self.records.get(label)
            if record is None or record.basis is None:
                continue
            basis = dict(record.basis)
            edges.append({
                "from_label": basis.get("basis_label"),
                "to_label": label,
                "basis_kind": basis.get("basis_kind"),
                "candidate_id": basis.get("candidate_id"),
                "decision": record.decision,
                "status": record.status,
                "transform_parameters": basis.get("transform_parameters"),
            })
        return edges

    def summary(self) -> dict[str, object]:
        return {
            "source_label": self.source_label,
            "config": {"min_region_coverage": self.config.min_region_coverage},
            "records": [self.records[label].summary() for label in self.series.labels if label in self.records],
            "corrections": [
                {"label": c.label, "use_as_basis": c.use_as_basis, "voxels": int(c.mask.sum())}
                for c in self._corrections.values()
            ],
            "curve": self.change_curve(),
            "lineage": self.lineage(),
        }

