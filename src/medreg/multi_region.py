from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from .image import MedicalImage
from .propagation import PropagationSession
from .series import SeriesResult


RELATION_CONTAINS = "contains"
RELATION_SEPARATE = "separate"
RELATION_CONTACT = "contact"
RELATION_KINDS = {RELATION_CONTAINS, RELATION_SEPARATE, RELATION_CONTACT}


@dataclass(frozen=True)
class RelationshipSpec:
    """A user-declared topological relationship between two region tracks."""

    first_region: str
    second_region: str
    expected: str
    declared_at: str
    source: str = "manual"
    note: str | None = None

    @property
    def relation_id(self) -> str:
        return f"{self.first_region}--{self.expected}--{self.second_region}"

    def involves(self, region_id: str) -> bool:
        return self.first_region == region_id or self.second_region == region_id

    def summary(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "first_region": self.first_region,
            "second_region": self.second_region,
            "expected": self.expected,
            "declared_at": self.declared_at,
            "source": self.source,
            "note": self.note,
        }


class MultiRegionPropagationSession:
    """Propagate several named regions and audit their relationships.

    Each named region keeps its own anchor and corrections. Relationship
    violations are review records only; the session never clips, moves, or
    deletes a manual mask to force the declared topology.
    """

    def __init__(
        self,
        images: list[MedicalImage],
        series: SeriesResult,
        min_region_coverage: float = 0.6,
    ):
        self.images = list(images)
        self.series = series
        self.min_region_coverage = float(min_region_coverage)
        self.regions: dict[str, PropagationSession] = {}
        self.relationships: dict[str, RelationshipSpec] = {}

    def _check_region_id(self, region_id: str) -> None:
        if not isinstance(region_id, str) or not region_id.strip():
            raise ValueError("region id must be a non-empty string")
        if region_id in {".", ".."} or "/" in region_id or "\\" in region_id:
            raise ValueError(f"invalid region id {region_id!r}")

    def region_session(self, region_id: str) -> PropagationSession:
        try:
            return self.regions[region_id]
        except KeyError as exc:
            raise KeyError(f"unknown region {region_id!r}") from exc

    def add_region(
        self,
        region_id: str,
        anchor_label: str,
        anchor_mask: np.ndarray,
    ) -> PropagationSession:
        self._check_region_id(region_id)
        if region_id in self.regions:
            raise ValueError(f"region {region_id!r} already exists")
        session = PropagationSession(
            self.images,
            self.series,
            anchor_label,
            anchor_mask,
            min_region_coverage=self.min_region_coverage,
        )
        self.regions[region_id] = session
        return session

    def define_relationship(
        self,
        first_region: str,
        second_region: str,
        expected: str,
        declared_at: str | None = None,
        source: str = "manual",
        note: str | None = None,
    ) -> RelationshipSpec:
        if first_region == second_region:
            raise ValueError("a relationship requires two different regions")
        self.region_session(first_region)
        self.region_session(second_region)
        if expected not in RELATION_KINDS:
            raise ValueError(
                f"unknown relationship {expected!r}; expected one of {sorted(RELATION_KINDS)}"
            )
        if declared_at is None:
            declared_at = self.region_session(first_region).anchor_label
        if declared_at not in self.series.labels:
            raise KeyError(f"unknown declaration label {declared_at!r}")
        spec = RelationshipSpec(
            first_region,
            second_region,
            expected,
            declared_at,
            source=source,
            note=note,
        )
        self.relationships[spec.relation_id] = spec
        return spec

    def apply_region_correction(
        self,
        region_id: str,
        label: str,
        mask: np.ndarray,
        use_as_basis: bool = False,
    ) -> dict[str, Any]:
        """Correct one region and return before/after evidence for related pairs."""

        before = {entry["relation_id"]: entry for entry in self.review_relationships()}
        self.region_session(region_id).apply_correction(label, mask, use_as_basis=use_as_basis)
        after = {entry["relation_id"]: entry for entry in self.review_relationships()}
        affected = []
        for relation_id, spec in self.relationships.items():
            if not spec.involves(region_id):
                continue
            affected.append(
                self._impact_entry(
                    spec,
                    before.get(relation_id, {}).get("observations", []),
                    after.get(relation_id, {}).get("observations", []),
                    changed_label=label,
                )
            )
        return {
            "corrected_region": region_id,
            "label": label,
            "use_as_basis": use_as_basis,
            "affected_relationships": affected,
        }

    def withdraw_region_correction(self, region_id: str, label: str) -> dict[str, Any]:
        """Withdraw one region's correction without touching another region or anchor."""

        session = self.region_session(region_id)
        before_anchor = session.anchor_mask.copy()
        other_before = {
            other_id: other.region(label).mask.copy() if other.region(label).mask is not None else None
            for other_id, other in self.regions.items()
            if other_id != region_id
        }
        session.withdraw_correction(label)
        unchanged_regions = {}
        for other_id, mask in other_before.items():
            current = self.region_session(other_id).region(label).mask
            unchanged_regions[other_id] = bool(
                (mask is None and current is None)
                or (mask is not None and current is not None and np.array_equal(mask, current))
            )
        return {
            "withdrawn_region": region_id,
            "label": label,
            "anchor_unchanged": np.array_equal(session.anchor_mask, before_anchor),
            "other_regions_unchanged": unchanged_regions,
            "review": self.review_relationships(labels={label}, region_ids={region_id}),
        }

    def apply_candidate_selection(
        self,
        label: str,
        candidate_id: str,
        *,
        region_id: str | None = None,
    ) -> dict[str, Any]:
        """Switch a registration candidate while retaining all manual masks."""

        affected_labels = {label}
        for track in self.regions.values():
            affected_labels.update(
                other
                for other, region in track.regions.items()
                if region.basis_label == label
            )
        for track in self.regions.values():
            track.apply_candidate_selection(label, candidate_id, keep_manual=True)
        self.series = next(iter(self.regions.values())).series
        region_ids = {region_id} if region_id is not None else None
        return {
            "label": label,
            "candidate_id": candidate_id,
            "manual_masks_preserved": True,
            "affected_labels": sorted(affected_labels),
            "review": self.review_relationships(
                labels=affected_labels,
                region_ids=region_ids,
            ),
        }

    def _topology(
        self,
        first_mask: np.ndarray | None,
        second_mask: np.ndarray | None,
    ) -> tuple[str, dict[str, int]]:
        if first_mask is None or second_mask is None:
            return "pending", {}
        intersection = np.logical_and(first_mask, second_mask)
        first_count = int(first_mask.sum())
        second_count = int(second_mask.sum())
        intersection_count = int(intersection.sum())
        if intersection_count == second_count and second_count > 0:
            actual = RELATION_CONTAINS
        elif intersection_count:
            actual = "overlap"
        else:
            touches = int(np.logical_and(binary_dilation(first_mask), second_mask).sum())
            actual = RELATION_CONTACT if touches else RELATION_SEPARATE
        return actual, {
            "first_voxels": first_count,
            "second_voxels": second_count,
            "intersection_voxels": intersection_count,
            "second_outside_first_voxels": int((second_mask & ~first_mask).sum()),
            "contact_voxels": 0 if actual != RELATION_CONTACT else touches,
        }

    def _region_evidence(self, region_id: str, label: str) -> dict[str, Any]:
        track = self.region_session(region_id)
        region = track.region(label)
        return {
            "region_id": region_id,
            "label": label,
            "kind": region.kind,
            "anchor_label": track.anchor_label,
            "basis_label": region.basis_label,
            "candidate_id": region.candidate_id,
            "selection_method": region.selection_method,
            "transform_parameters": region.transform_parameters,
            "coverage_fraction": region.coverage_fraction,
            "manual_correction_present": label in track.corrections,
            "used_as_basis": label in track.basis_labels,
            "reasons": list(region.reasons),
        }

    def review_relationships(
        self,
        *,
        labels: set[str] | None = None,
        region_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        for spec in self.relationships.values():
            if region_ids is not None and not any(
                spec.involves(region_id) for region_id in region_ids
            ):
                continue
            observations = []
            conflict_count = 0
            pending_count = 0
            for label in self.series.labels:
                if labels is not None and label not in labels:
                    continue
                first_region = self.region_session(spec.first_region).region(label)
                second_region = self.region_session(spec.second_region).region(label)
                actual, metrics = self._topology(first_region.mask, second_region.mask)
                if actual == "pending":
                    status = "pending"
                    pending_count += 1
                else:
                    status = "satisfied" if actual == spec.expected else "conflict"
                    if status == "conflict":
                        conflict_count += 1
                observations.append(
                    {
                        "label": label,
                        "expected": spec.expected,
                        "actual": actual,
                        "status": status,
                        "metrics": metrics,
                        "declared_at": spec.declared_at,
                        "declaration_source": spec.source,
                        "regions": {
                            "first": self._region_evidence(spec.first_region, label),
                            "second": self._region_evidence(spec.second_region, label),
                        },
                    }
                )
            reviews.append(
                {
                    **spec.summary(),
                    "status": "conflict" if conflict_count else ("pending" if pending_count else "satisfied"),
                    "conflict_count": conflict_count,
                    "pending_count": pending_count,
                    "observations": observations,
                }
            )
        return reviews

    def export_relationships(self) -> dict[str, Any]:
        return {
            "schema": "medreg-multi-region-relationships-v1",
            "labels": list(self.series.labels),
            "regions": {
                region_id: {
                    "anchor_label": track.anchor_label,
                    "anchor_voxels": int(track.anchor_mask.sum()),
                    "corrections": {
                        label: {"use_as_basis": label in track.basis_labels}
                        for label in sorted(track.corrections)
                    },
                }
                for region_id, track in sorted(self.regions.items())
            },
            "declared_relationships": [spec.summary() for spec in self.relationships.values()],
            "relationship_review": self.review_relationships(),
        }

    def _impact_entry(
        self,
        spec: RelationshipSpec,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        changed_label: str,
    ) -> dict[str, Any]:
        before_map = {entry["label"]: entry for entry in before}
        after_map = {entry["label"]: entry for entry in after}
        return {
            **spec.summary(),
            "changed_label": changed_label,
            "before": before_map.get(changed_label),
            "after": after_map.get(changed_label),
        }

    def snapshot(self) -> "MultiRegionPropagationSession":
        return deepcopy(self)
