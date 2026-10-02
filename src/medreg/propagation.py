from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

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
        self.regions: dict[str, PropagatedRegion] = {}
        self._recompute()

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _image(self, label: str) -> MedicalImage:
        return self.images[self.labels.index(label)]

    def _selected_transform(
        self, label: str
    ) -> tuple[RigidTransform | None, str | None, str | None]:
        tp = self.series.timepoint(label)
        if tp.is_reference:
            return RigidTransform(), None, "reference"
        if tp.multi is None or tp.multi.selected is None:
            return None, None, None
        selected = tp.multi.selected
        return selected.result.transform, selected.candidate_id, tp.multi.selection_method

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

    def _propagate(self, basis_label: str, target_label: str) -> PropagatedRegion:
        basis_image = self._image(basis_label)
        target_image = self._image(target_label)
        basis_transform, _, _ = self._selected_transform(basis_label)
        target_transform, candidate_id, method = self._selected_transform(target_label)
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
            regions[label] = self._propagate(basis_label, label)
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
        self.series = select_series_candidate(self.series, label, candidate_id)
        if not keep_manual:
            for conflict in impact["manual_conflicts"]:
                self.corrections.pop(conflict, None)
                self.basis_labels.discard(conflict)
        self._recompute()
        return impact

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


@dataclass(frozen=True)
class RegionRelation:
    """A binary relationship that applies over an inclusive label range."""

    region_a: str
    region_b: str
    kind: str  # disjoint | contains
    start_label: str
    end_label: str

    def __post_init__(self) -> None:
        if self.kind not in {"disjoint", "contains"}:
            raise ValueError(f"unsupported region relationship {self.kind!r}")
        if self.region_a == self.region_b:
            raise ValueError("a region relationship requires two different regions")

    def applies_to(self, label: str, labels: list[str]) -> bool:
        start = labels.index(self.start_label)
        end = labels.index(self.end_label)
        current = labels.index(label)
        return min(start, end) <= current <= max(start, end)

    def summary(self) -> dict[str, object]:
        return {
            "region_a": self.region_a,
            "region_b": self.region_b,
            "kind": self.kind,
            "start_label": self.start_label,
            "end_label": self.end_label,
        }


class MultiRegionPropagationSession:
    """Propagate several regions and audit their mutual relationships."""

    def __init__(
        self,
        images: list[MedicalImage],
        series: SeriesResult,
        anchors: dict[str, tuple[str, np.ndarray]],
        relations: list[RegionRelation] | None = None,
        min_region_coverage: float = 0.6,
    ):
        self.images = list(images)
        self.series = series
        self.labels = list(series.labels)
        if not anchors:
            raise ValueError("at least one anchored region is required")
        self.relations = list(relations or [])
        relation_regions = {region for rel in self.relations for region in (rel.region_a, rel.region_b)}
        unknown = relation_regions - set(anchors)
        if unknown:
            raise KeyError(f"relationship references unknown regions: {sorted(unknown)}")
        for relation in self.relations:
            if relation.start_label not in self.labels or relation.end_label not in self.labels:
                raise ValueError("relationship time range contains an unknown timepoint label")
        self.anchor_specs = dict(anchors)
        self.min_region_coverage = float(min_region_coverage)
        self.sessions: dict[str, PropagationSession] = {
            region_id: PropagationSession(
                self.images,
                self.series,
                anchor_label,
                mask,
                min_region_coverage=self.min_region_coverage,
            )
            for region_id, (anchor_label, mask) in self.anchor_specs.items()
        }

    def region(self, region_id: str, label: str) -> PropagatedRegion:
        return self.sessions[region_id].region(label)

    def apply_correction(
        self,
        region_id: str,
        label: str,
        mask: np.ndarray,
        use_as_basis: bool = False,
    ) -> dict[str, object]:
        """Correct one region only, then audit all relationships at that label."""
        self.sessions[region_id].apply_correction(label, mask, use_as_basis=use_as_basis)
        return {
            "corrected_region": region_id,
            "label": label,
            "relationships": [
                result
                for result in self.relationship_report(label)
                if result["status"] in {"violated", "pending"}
            ],
        }

    def withdraw_correction(self, region_id: str, label: str) -> None:
        """Withdraw one region's correction without touching any other region."""
        self.sessions[region_id].withdraw_correction(label)

    def _copy(self) -> "MultiRegionPropagationSession":
        copied = MultiRegionPropagationSession.__new__(MultiRegionPropagationSession)
        copied.images = self.images
        copied.series = deepcopy(self.series)
        copied.labels = self.labels
        copied.relations = self.relations
        copied.anchor_specs = {
            region_id: (session.anchor_label, session.anchor_mask.copy())
            for region_id, session in self.sessions.items()
        }
        copied.min_region_coverage = self.min_region_coverage
        copied.sessions = {}
        for region_id, session in self.sessions.items():
            clone = PropagationSession.__new__(PropagationSession)
            clone.images = copied.images
            clone.series = copied.series
            clone.labels = copied.labels
            clone.anchor_label = session.anchor_label
            clone.anchor_mask = session.anchor_mask.copy()
            clone.min_region_coverage = session.min_region_coverage
            clone.corrections = {label: mask.copy() for label, mask in session.corrections.items()}
            clone.basis_labels = set(session.basis_labels)
            clone.regions = {}
            clone._recompute()
            copied.sessions[region_id] = clone
        return copied

    def preview_correction(
        self,
        region_id: str,
        label: str,
        mask: np.ndarray,
        use_as_basis: bool = False,
    ) -> dict[str, object]:
        trial = self._copy()
        trial.apply_correction(region_id, label, mask, use_as_basis=use_as_basis)
        downstream = [
            edge
            for edge in trial.sessions[region_id].propagation_graph()
            if edge["source"] == label and edge["target"] != label
        ]
        affected_labels = [label, *(edge["target"] for edge in downstream)]
        return {
            "corrected_region": region_id,
            "label": label,
            "use_as_basis": use_as_basis,
            "downstream_propagation": downstream,
            "relationship_review": [
                result
                for affected in dict.fromkeys(affected_labels)
                for result in trial.relationship_report(affected)
            ],
        }

    def preview_candidate_selection(self, label: str, candidate_id: str) -> dict[str, object]:
        trial = self._copy()
        manual_regions = sorted(
            region_id
            for region_id, session in trial.sessions.items()
            if label in session.corrections
            or any(region.basis_label == label for region in session.regions.values())
        )
        trial.series = select_series_candidate(trial.series, label, candidate_id)
        for session in trial.sessions.values():
            session.series = trial.series
            session._recompute()
        affected = [label] + [
            other
            for other in self.labels
            if self.labels.index(other) > self.labels.index(label)
        ]
        return {
            "label": label,
            "candidate_id": candidate_id,
            "manual_regions_at_label": [
                region_id for region_id in manual_regions if label in trial.sessions[region_id].corrections
            ],
            "relationship_review": [
                result
                for affected in affected
                for result in trial.relationship_report(affected)
                if affected == label or result["status"] in {"violated", "pending"}
            ],
        }

    def apply_candidate_selection(
        self,
        label: str,
        candidate_id: str,
        on_manual: str = "abort",
    ) -> dict[str, object]:
        """Change one candidate and re-audit region relationships.

        ``on_manual`` must explicitly be ``abort``, ``keep`` or ``replace``.
        ``replace`` removes only the same region's manual mask at this label;
        masks and original drawings belonging to other regions are retained.
        """
        if on_manual not in {"abort", "keep", "replace"}:
            raise ValueError("on_manual must be 'abort', 'keep', or 'replace'")
        impact = self.preview_candidate_selection(label, candidate_id)
        manual_here = impact["manual_regions_at_label"]
        if manual_here and on_manual == "abort":
            raise ValueError(
                f"manual results exist for {manual_here} at {label}; choose keep or replace"
            )
        self.series = select_series_candidate(self.series, label, candidate_id)
        for region_id, session in self.sessions.items():
            if manual_here and on_manual == "replace" and region_id in manual_here:
                session.withdraw_correction(label)
            session.series = self.series
            session._recompute()
        impact["on_manual"] = on_manual
        impact["unresolved_relationship_labels"] = sorted(
            {
                entry["label"]
                for entry in self.relationship_report()
                if entry["status"] == "violated"
            }
        )
        return impact

    @staticmethod
    def _contact_voxels(first: np.ndarray, second: np.ndarray) -> int:
        from scipy.ndimage import binary_dilation

        if not first.any() or not second.any():
            return 0
        if np.logical_and(first, second).any():
            return 0
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[:, 1, 1] = True
        structure[1, :, 1] = True
        structure[1, 1, :] = True
        return int(np.logical_and(binary_dilation(first, structure=structure), second).sum())

    def relationship_report(self, label: str | None = None) -> list[dict[str, object]]:
        """Return per-region coverage plus every applicable relationship audit."""
        labels = [label] if label is not None else self.labels
        report: list[dict[str, object]] = []
        for current_label in labels:
            for relation in self.relations:
                if not relation.applies_to(current_label, self.labels):
                    continue
                first = self.region(relation.region_a, current_label)
                second = self.region(relation.region_b, current_label)
                entry: dict[str, object] = {
                    "label": current_label,
                    **relation.summary(),
                    "region_kinds": {
                        relation.region_a: first.kind,
                        relation.region_b: second.kind,
                    },
                    "coverage": {
                        relation.region_a: first.coverage_fraction,
                        relation.region_b: second.coverage_fraction,
                    },
                }
                if first.mask is None or second.mask is None:
                    missing = [
                        region_id
                        for region_id, region in (
                            (relation.region_a, first),
                            (relation.region_b, second),
                        )
                        if region.mask is None
                    ]
                    entry.update(
                        {
                            "status": "pending",
                            "conflict_mask": None,
                            "overlap_voxels": None,
                            "contact_voxels": None,
                            "reasons": [f"region {region_id} is pending; relationship cannot be judged" for region_id in missing],
                        }
                    )
                    report.append(entry)
                    continue
                overlap = np.logical_and(first.mask, second.mask)
                entry["overlap_voxels"] = int(overlap.sum())
                entry["contact_voxels"] = self._contact_voxels(first.mask, second.mask)
                if relation.kind == "disjoint":
                    satisfied = not overlap.any()
                    conflict = overlap
                    reason = "regions overlap after propagation; both masks are retained"
                else:
                    outside = np.logical_and(second.mask, np.logical_not(first.mask))
                    satisfied = not outside.any()
                    conflict = outside
                    reason = "contained region extends outside its parent after propagation; no clipping was applied"
                entry.update(
                    {
                        "status": "satisfied" if satisfied else "violated",
                        "conflict_mask": conflict if not satisfied else None,
                        "conflict_voxels": int(conflict.sum()),
                        "reasons": [] if satisfied else [reason],
                    }
                )
                report.append(entry)
        return report

    def change_curve(self) -> list[dict[str, object]]:
        """Volumes and relationship/contact state separated by result origin."""
        relationships = {
            label: [entry for entry in self.relationship_report(label)]
            for label in self.labels
        }
        curve: list[dict[str, object]] = []
        for current_label in self.labels:
            image = self.images[self.labels.index(current_label)]
            voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
            regions: dict[str, object] = {}
            for region_id, session in self.sessions.items():
                region = session.region(current_label)
                regions[region_id] = {
                    "kind": region.kind,
                    "volume_mm3": None if region.mask is None else float(region.mask.sum() * voxel_volume),
                    "coverage_fraction": region.coverage_fraction,
                    "candidate_id": region.candidate_id,
                    "basis_label": region.basis_label,
                }
            curve.append(
                {
                    "label": current_label,
                    "regions": regions,
                    "relationships": [
                        {
                            key: value
                            for key, value in entry.items()
                            if key != "conflict_mask"
                        }
                        for entry in relationships[current_label]
                    ],
                    "has_conflict": any(
                        entry["status"] == "violated" for entry in relationships[current_label]
                    ),
                }
            )
        return curve

    def summary(self) -> dict[str, object]:
        return {
            "labels": self.labels,
            "regions": sorted(self.sessions),
            "anchors": {
                region_id: session.region(session.anchor_label).summary()
                for region_id, session in self.sessions.items()
            },
            "relations": [relation.summary() for relation in self.relations],
            "relationship_report": [
                {key: value for key, value in entry.items() if key != "conflict_mask"}
                for entry in self.relationship_report()
            ],
            "change_curve": self.change_curve(),
        }
