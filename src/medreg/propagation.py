from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .image import MedicalImage
from .resample import resample
from .series import SeriesResult, select_series_candidate
from .transform import RigidTransform


@dataclass
class PropagatedRegion:
    """Region outcome at one timepoint of a propagation session.

    ``kind`` is one of:

    - ``source``: the timepoint where the user drew the region by hand.
    - ``auto``: propagated automatically through the adopted registration
      candidate of this timepoint.
    - ``manual``: corrected by hand at this timepoint; the correction is
      stored separately from the automatic result and never overwrites the
      original drawing at the source timepoint.
    - ``pending``: no trustworthy region exists here (failed registration,
      no adopted candidate, or insufficient coverage). Nothing is
      fabricated across missing imaging.
    """

    label: str
    kind: str  # source | auto | manual | pending
    mask: np.ndarray | None
    basis_label: str | None = None
    candidate_id: str | None = None
    selection_method: str | None = None
    transform_parameters: list[float] | None = None
    coverage_fraction: float | None = None
    reasons: list[str] = field(default_factory=list)
    blocked_voxel_range: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "has_mask": self.mask is not None,
            "voxels": int(self.mask.sum()) if self.mask is not None else 0,
            "basis_label": self.basis_label,
            "candidate_id": self.candidate_id,
            "selection_method": self.selection_method,
            "transform_parameters": self.transform_parameters,
            "coverage_fraction": self.coverage_fraction,
            "reasons": list(self.reasons),
            "blocked_voxel_range": copy.deepcopy(self.blocked_voxel_range),
        }


def _check_mask(mask: np.ndarray, image: MedicalImage, label: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError(
            f"region mask for {label!r} has shape {mask.shape}, expected {image.shape}"
        )
    if not mask.any():
        raise ValueError(f"region mask for {label!r} is empty")
    return mask


class PropagationSession:
    """Propagate a hand-drawn region across a registered temporal series.

    The session keeps the original drawing, every manual correction, and
    the per-timepoint automatic results as separate records, so every
    region can be traced back to its source image, the transform and the
    adoption decision that produced it.
    """

    def __init__(
        self,
        images: list[MedicalImage],
        series: SeriesResult,
        anchor_label: str,
        anchor_mask: np.ndarray,
        min_region_coverage: float = 0.6,
    ):
        if len(images) != len(series.labels):
            raise ValueError("images must match the series timepoints")
        self.images = list(images)
        self.series = series
        self.labels = list(series.labels)
        if anchor_label not in self.labels:
            raise KeyError(f"unknown anchor label {anchor_label!r}")
        self.anchor_label = anchor_label
        index = self.labels.index(anchor_label)
        self.anchor_mask = _check_mask(anchor_mask, self.images[index], anchor_label)
        self.min_region_coverage = float(min_region_coverage)
        # Manual corrections live on the grid of their own timepoint. A
        # correction in ``basis_labels`` is additionally used as the new
        # propagation basis for the timepoints nearest to it.
        self.corrections: dict[str, np.ndarray] = {}
        self.basis_labels: set[str] = set()
        self.candidate_overrides: dict[str, str] = {}
        self.regions: dict[str, PropagatedRegion] = {}
        self._recompute()

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _image(self, label: str) -> MedicalImage:
        return self.images[self.labels.index(label)]

    def _selected_transform(
        self, label: str, candidate_id: str | None = None
    ) -> tuple[RigidTransform | None, str | None, str | None, str | None]:
        tp = self.series.timepoint(label)
        if tp.is_reference:
            return RigidTransform(), None, "reference", None
        if tp.multi is None or tp.multi.selected is None:
            return None, None, None, "timepoint has no adopted registration candidate"
        requested_id = candidate_id or self.candidate_overrides.get(label)
        if requested_id is not None:
            chosen = next(
                (item for item in tp.multi.candidates if item.candidate_id == requested_id),
                None,
            )
            if chosen is None:
                return None, requested_id, None, f"unknown candidate {requested_id!r}"
            if chosen.result.transform is None:
                return None, requested_id, None, f"candidate {requested_id!r} failed acceptance"
            return (
                chosen.result.transform,
                chosen.candidate_id,
                f"joint_preview:{chosen.candidate_id}",
                None,
            )
        selected = tp.multi.selected
        return selected.result.transform, selected.candidate_id, tp.multi.selection_method, None

    def _basis_for(self, label: str) -> str:
        """Nearest correction marked as basis, otherwise the original anchor."""
        target_index = self.labels.index(label)
        best_label = self.anchor_label
        best_distance = abs(target_index - self.labels.index(self.anchor_label))
        for basis in sorted(self.basis_labels):
            distance = abs(target_index - self.labels.index(basis))
            if distance < best_distance:
                best_label, best_distance = basis, distance
        return best_label

    def _basis_mask(self, label: str) -> np.ndarray:
        if label == self.anchor_label:
            return self.anchor_mask
        return self.corrections[label]

    def _propagate(
        self, basis_label: str, target_label: str, target_candidate_id: str | None = None
    ) -> PropagatedRegion:
        basis_image = self._image(basis_label)
        target_image = self._image(target_label)
        basis_transform, _, _, basis_blocked = self._selected_transform(
            basis_label, self.candidate_overrides.get(basis_label)
        )
        target_transform, candidate_id, method, target_blocked = self._selected_transform(
            target_label, target_candidate_id
        )
        if basis_transform is None:
            return PropagatedRegion(
                label=target_label,
                kind="pending",
                mask=None,
                basis_label=basis_label,
                reasons=[
                    f"basis timepoint {basis_label!r} has no adopted transform; "
                    "the region cannot be propagated through a failed registration"
                ],
                blocked_voxel_range={
                    "type": "missing_dependency",
                    "basis_label": basis_label,
                    "target_label": target_label,
                    "detail": basis_blocked,
                },
            )
        if target_transform is None:
            return PropagatedRegion(
                label=target_label,
                kind="pending",
                mask=None,
                basis_label=basis_label,
                reasons=[
                    "no adopted registration candidate at this timepoint; "
                    "the region stays pending instead of being interpolated "
                    "across missing imaging"
                ],
                blocked_voxel_range={
                    "type": "missing_candidate",
                    "basis_label": basis_label,
                    "target_label": target_label,
                    "candidate_id": candidate_id,
                    "detail": target_blocked,
                },
            )
        # World correspondence: target -> reference -> basis.
        warp = target_transform.inverse().compose(basis_transform)
        basis_mask = self._basis_mask(basis_label)
        coords = np.column_stack(np.where(basis_mask))
        world = basis_image.index_to_world(coords.T).T
        mapped = warp.apply(world)
        target_indices = target_image.world_to_index(mapped.T).T
        shape = np.asarray(target_image.shape)
        inside = np.all((target_indices >= 0) & (target_indices <= shape - 1), axis=1)
        outside_indices = target_indices[~inside]
        blocked_range = None
        if outside_indices.size:
            clipped = np.clip(outside_indices, 0, shape - 1)
            blocked_range = {
                "type": "outside_field_of_view",
                "mapped_voxels": int(outside_indices.shape[0]),
                "nearest_in_grid_min": [int(value) for value in clipped.min(axis=0)],
                "nearest_in_grid_max": [int(value) for value in clipped.max(axis=0)],
                "grid_shape": [int(value) for value in shape],
            }
        coverage_fraction = float(inside.mean()) if inside.size else 0.0
        transform_parameters = target_transform.parameters().tolist()
        if coverage_fraction < self.min_region_coverage:
            return PropagatedRegion(
                label=target_label,
                kind="pending",
                mask=None,
                basis_label=basis_label,
                candidate_id=candidate_id,
                selection_method=method,
                transform_parameters=transform_parameters,
                coverage_fraction=coverage_fraction,
                reasons=[
                    f"region_coverage_insufficient: only {coverage_fraction:.3f} of the "
                    f"propagated region lands inside the field of view, below "
                    f"{self.min_region_coverage:.3f}; the missing part is left pending"
                ],
                blocked_voxel_range=blocked_range,
            )
        mask_image = MedicalImage(basis_mask.astype(np.float32), basis_image.affine)
        warped, _ = resample(mask_image, target_image, warp, order=0)
        return PropagatedRegion(
            label=target_label,
            kind="auto",
            mask=warped > 0.5,
            basis_label=basis_label,
            candidate_id=candidate_id,
            selection_method=method,
            transform_parameters=transform_parameters,
            coverage_fraction=coverage_fraction,
        )

    def _recompute(self) -> None:
        regions: dict[str, PropagatedRegion] = {}
        for label in self.labels:
            if label == self.anchor_label:
                regions[label] = PropagatedRegion(
                    label=label, kind="source", mask=self.anchor_mask.copy()
                )
                continue
            basis_label = self._basis_for(label)
            regions[label] = self._propagate(
                basis_label, label, self.candidate_overrides.get(label)
            )
        # Manual corrections override the automatic result at their own
        # timepoint; the original drawing at the anchor is never touched.
        for label, mask in self.corrections.items():
            auto = regions[label]
            regions[label] = PropagatedRegion(
                label=label,
                kind="manual",
                mask=mask,
                basis_label=auto.basis_label,
                candidate_id=auto.candidate_id,
                selection_method=auto.selection_method,
                transform_parameters=auto.transform_parameters,
                coverage_fraction=auto.coverage_fraction,
            )
        self.regions = regions

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def region(self, label: str) -> PropagatedRegion:
        return self.regions[label]

    def apply_correction(self, label: str, mask: np.ndarray, use_as_basis: bool = False) -> None:
        """Store a manual correction at one timepoint and recompute the rest.

        With ``use_as_basis=True`` the correction becomes the propagation
        basis for its neighbouring timepoints, making the new propagation
        relationship visible in :meth:`propagation_graph`.
        """
        if label == self.anchor_label:
            raise ValueError(
                "the anchor timepoint holds the original drawing; correct another "
                "timepoint or start a new session instead of overwriting it"
            )
        self.corrections[label] = _check_mask(mask, self._image(label), label)
        if use_as_basis:
            self.basis_labels.add(label)
        else:
            self.basis_labels.discard(label)
        self._recompute()

    def withdraw_correction(self, label: str) -> None:
        """Remove a manual correction; the automatic result is restored.

        The original drawing at the anchor timepoint is never affected.
        """
        if label not in self.corrections:
            raise KeyError(f"no manual correction recorded at {label!r}")
        del self.corrections[label]
        self.basis_labels.discard(label)
        self._recompute()

    def preview_candidate_selection(self, label: str, candidate_id: str) -> dict[str, object]:
        """List the impact of re-selecting a candidate before applying it.

        Returns which automatic results would be recomputed and which
        manual corrections would conflict with the new choice, so the user
        can decide to keep or drop them. Nothing is modified here.
        """
        tp = self.series.timepoint(label)
        if tp.multi is None:
            raise ValueError(f"timepoint {label!r} has no candidates to select from")
        known = {candidate.candidate_id for candidate in tp.multi.candidates}
        if candidate_id not in known:
            raise KeyError(f"unknown candidate {candidate_id!r} at {label!r}")
        auto_to_update = [
            other
            for other, region in self.regions.items()
            if region.kind == "auto"
            and (other == label or region.basis_label == label)
        ]
        manual_conflicts = [
            other
            for other in self.corrections
            if other == label or self._basis_for(other) == label
        ]
        return {
            "label": label,
            "candidate_id": candidate_id,
            "auto_to_update": sorted(auto_to_update),
            "manual_conflicts": sorted(manual_conflicts),
        }

    def apply_candidate_selection(
        self,
        label: str,
        candidate_id: str,
        keep_manual: bool = True,
    ) -> dict[str, object]:
        """Re-select a registration candidate and recompute affected regions.

        With ``keep_manual=True`` existing corrections are preserved and
        only the automatic results are recomputed; with ``keep_manual=False``
        the conflicting corrections reported by the preview are dropped.
        """
        impact = self.preview_candidate_selection(label, candidate_id)
        self.candidate_overrides.pop(label, None)
        self.series = select_series_candidate(self.series, label, candidate_id)
        if not keep_manual:
            for conflict in impact["manual_conflicts"]:
                self.corrections.pop(conflict, None)
                self.basis_labels.discard(conflict)
        self._recompute()
        return impact

    def clone(self) -> "PropagationSession":
        """Deep-copy masks and local choices while sharing read-only images."""
        cloned = copy.copy(self)
        cloned.images = self.images
        cloned.series = self.series
        cloned.labels = list(self.labels)
        cloned.corrections = {key: value.copy() for key, value in self.corrections.items()}
        cloned.basis_labels = set(self.basis_labels)
        cloned.candidate_overrides = dict(self.candidate_overrides)
        cloned.anchor_mask = self.anchor_mask.copy()
        cloned.regions = copy.deepcopy(self.regions)
        return cloned

    def set_region_candidate(self, label: str, candidate_id: str | None) -> None:
        """Set a candidate used only by this region's propagation."""
        tp = self.series.timepoint(label)
        if tp.is_reference:
            raise ValueError("the reference timepoint has no candidates to select from")
        if tp.multi is None:
            raise ValueError(f"timepoint {label!r} has no candidates")
        known = {candidate.candidate_id for candidate in tp.multi.candidates}
        if candidate_id not in known:
            raise KeyError(f"unknown candidate {candidate_id!r} at {label!r}")
        self.candidate_overrides[label] = candidate_id
        self._recompute()

    def propagation_graph(self) -> list[dict[str, object]]:
        """Visible propagation relationships: one edge per non-source region."""
        edges: list[dict[str, object]] = []
        for label in self.labels:
            region = self.regions[label]
            if region.kind == "source":
                continue
            edges.append(
                {
                    "source": region.basis_label,
                    "target": label,
                    "kind": region.kind,
                    "basis_is_correction": region.basis_label in self.basis_labels,
                    "candidate_id": region.candidate_id,
                    "transform_parameters": region.transform_parameters,
                    "coverage_fraction": region.coverage_fraction,
                    "reasons": list(region.reasons),
                }
            )
        return edges

    def change_curve(self) -> list[dict[str, object]]:
        """Per-timepoint region volume/centroid, tagged by how it was obtained.

        Automatic propagation and manual correction are distinguished so a
        smooth-looking curve never hides where a human stepped in; pending
        timepoints contribute explicit gaps (``None`` values).
        """
        curve: list[dict[str, object]] = []
        for label in self.labels:
            region = self.regions[label]
            entry: dict[str, object] = {"label": label, "kind": region.kind}
            if region.mask is None:
                entry["volume_mm3"] = None
                entry["centroid_world"] = None
                entry["reasons"] = list(region.reasons)
            else:
                image = self._image(label)
                voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
                coords = np.column_stack(np.where(region.mask))
                centroid = image.index_to_world(coords.T).T.mean(axis=0)
                entry["volume_mm3"] = float(region.mask.sum() * voxel_volume)
                entry["centroid_world"] = centroid.tolist()
            curve.append(entry)
        return curve

    def summary(self) -> dict[str, object]:
        return {
            "anchor_label": self.anchor_label,
            "min_region_coverage": self.min_region_coverage,
            "corrections": {
                label: {"use_as_basis": label in self.basis_labels}
                for label in sorted(self.corrections)
            },
            "regions": [self.regions[label].summary() for label in self.labels],
            "propagation_graph": self.propagation_graph(),
            "change_curve": self.change_curve(),
        }
