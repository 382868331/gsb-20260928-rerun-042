
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image import MedicalImage
from .propagation import PropagationSession
from .series import SeriesResult

#: Relations a user can declare between two regions. "contains" is
#: ordered (first region contains the second); "separate" and
#: "contact" are symmetric.
DECLARABLE_RELATIONS = ("contains", "separate", "contact")


def _face_adjacent(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """True when the two disjoint masks touch across a voxel face."""
    for axis in range(mask_a.ndim):
        for delta in (-1, 1):
            shifted = np.roll(mask_a, delta, axis=axis)
            # np.roll wraps around; clear the wrapped slab.
            edge = [slice(None)] * mask_a.ndim
            edge[axis] = 0 if delta == 1 else -1
            shifted[tuple(edge)] = False
            if np.logical_and(shifted, mask_b).any():
                return True
    return False


def classify_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[str, int]:
    """Classify the spatial relation of an ordered region pair.

    Returns (actual, offending_voxels) where "actual" is one of
    contains / contained_by / coincident / overlap / contact / separate
    and "offending_voxels" counts voxels of the second region lying
    outside the first (relevant when a contains expectation is violated).
    """
    intersection = np.logical_and(mask_a, mask_b)
    outside = int(np.logical_and(mask_b, ~mask_a).sum())
    if intersection.any():
        if outside == 0:
            if int(np.logical_and(mask_a, ~mask_b).sum()) == 0:
                return "coincident", 0
            return "contains", 0
        if int(np.logical_and(mask_a, ~mask_b).sum()) == 0:
            return "contained_by", outside
        return "overlap", outside
    if _face_adjacent(mask_a, mask_b):
        return "contact", outside
    return "separate", outside


@dataclass
class RelationExpectation:
    """A declared relation between two regions, checked at every timepoint."""

    region_a: str
    region_b: str
    relation: str  # contains | separate | contact

    def key(self) -> str:
        return f"{self.region_a}|{self.region_b}|{self.relation}"


class MultiRegionSession:
    """Propagate several named regions and review their mutual relations.

    Each region keeps its own PropagationSession (own anchor, corrections
    and provenance). On top of that the session evaluates user-declared
    relations (contains / separate / contact) at every timepoint. Manual
    corrections are never trimmed to satisfy a relation: a correction
    that violates an expectation is kept as-is and surfaced as a
    "conflict" carrying its full provenance for review.
    """

    def __init__(
        self,
        images: list[MedicalImage],
        series: SeriesResult,
        min_region_coverage: float = 0.6,
    ):
        self.images = list(images)
        self.series = series
        self.labels = list(series.labels)
        self.min_region_coverage = float(min_region_coverage)
        self._sessions: dict[str, PropagationSession] = {}
        self.expectations: list[RelationExpectation] = []

    # ------------------------------------------------------------------ #
    # region management
    # ------------------------------------------------------------------ #

    def add_region(self, name: str, anchor_label: str, anchor_mask: np.ndarray) -> None:
        if name in self._sessions:
            raise ValueError(f"region {name!r} already exists")
        self._sessions[name] = PropagationSession(
            self.images,
            self.series,
            anchor_label,
            anchor_mask,
            min_region_coverage=self.min_region_coverage,
        )

    @property
    def region_names(self) -> list[str]:
        return sorted(self._sessions)

    def region_session(self, name: str) -> PropagationSession:
        if name not in self._sessions:
            raise KeyError(f"unknown region {name!r}")
        return self._sessions[name]

    # ------------------------------------------------------------------ #
    # relation declaration and evaluation
    # ------------------------------------------------------------------ #

    def expect(self, region_a: str, region_b: str, relation: str) -> None:
        """Declare that region_a relates to region_b at all timepoints.

        "contains" is ordered (region_a contains region_b); "separate"
        and "contact" are symmetric. Declaring an expectation never
        modifies any region; violations surface as conflicts in
        evaluate_relations().
        """
        if relation not in DECLARABLE_RELATIONS:
            raise ValueError(
                f"unknown relation {relation!r}; expected one of {DECLARABLE_RELATIONS}"
            )
        for name in (region_a, region_b):
            if name not in self._sessions:
                raise KeyError(f"unknown region {name!r}")
        if region_a == region_b:
            raise ValueError("a relation needs two distinct regions")
        expectation = RelationExpectation(region_a, region_b, relation)
        if any(existing.key() == expectation.key() for existing in self.expectations):
            raise ValueError(f"relation {expectation.key()!r} is already declared")
        self.expectations.append(expectation)

    def _basis_provenance(self, region: str, label: str) -> dict[str, object]:
        propagated = self._sessions[region].region(label)
        return {
            "region": region,
            "kind": propagated.kind,
            "basis_label": propagated.basis_label,
            "candidate_id": propagated.candidate_id,
            "selection_method": propagated.selection_method,
            "transform_parameters": propagated.transform_parameters,
            "coverage_fraction": propagated.coverage_fraction,
        }

    def _evaluate_one(self, expectation: RelationExpectation, label: str) -> dict[str, object]:
        region_a = self._sessions[expectation.region_a].region(label)
        region_b = self._sessions[expectation.region_b].region(label)
        basis = {
            expectation.region_a: self._basis_provenance(expectation.region_a, label),
            expectation.region_b: self._basis_provenance(expectation.region_b, label),
        }
        record: dict[str, object] = {
            "label": label,
            "relation": expectation.relation,
            "regions": [expectation.region_a, expectation.region_b],
            "basis": basis,
        }
        if region_a.mask is None or region_b.mask is None:
            reasons = []
            for name, propagated in (
                (expectation.region_a, region_a),
                (expectation.region_b, region_b),
            ):
                if propagated.mask is None:
                    reasons.append(f"{name}: " + "; ".join(propagated.reasons))
            record.update(
                status="pending",
                actual=None,
                offending_voxels=None,
                manual_origin=False,
                reasons=reasons,
            )
            return record
        actual, offending = classify_pair(region_a.mask, region_b.mask)
        satisfied = actual == expectation.relation
        manual_origin = region_a.kind == "manual" or region_b.kind == "manual"
        record.update(
            status="satisfied" if satisfied else "conflict",
            actual=actual,
            offending_voxels=offending,
            manual_origin=manual_origin,
            reasons=[]
            if satisfied
            else [
                f"expected {expectation.relation} but regions are {actual}"
                + (
                    f" ({offending} voxels of {expectation.region_b} outside "
                    f"{expectation.region_a})"
                    if expectation.relation == "contains"
                    else ""
                )
                + ("; involves a manual correction kept for review" if manual_origin else "")
            ],
        )
        return record

    def evaluate_relations(self) -> list[dict[str, object]]:
        """Evaluate every declared relation at every timepoint with provenance."""
        evaluations: list[dict[str, object]] = []
        for expectation in self.expectations:
            for label in self.labels:
                evaluations.append(self._evaluate_one(expectation, label))
        return evaluations

    def conflicts(self) -> list[dict[str, object]]:
        """All currently violated expectations, kept for review (never trimmed)."""
        return [e for e in self.evaluate_relations() if e["status"] == "conflict"]

    # ------------------------------------------------------------------ #
    # edits with impact comparison
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evaluation_key(record: dict[str, object]) -> tuple[str, str, str, str]:
        regions = record["regions"]
        return (regions[0], regions[1], record["relation"], record["label"])

    def _relation_changes(
        self,
        before: list[dict[str, object]],
        after: list[dict[str, object]],
        region: str | None = None,
        labels: set[str] | None = None,
    ) -> list[dict[str, object]]:
        before_map = {self._evaluation_key(record): record for record in before}
        changes: list[dict[str, object]] = []
        for record in after:
            if region is not None and region not in record["regions"]:
                continue
            if labels is not None and record["label"] not in labels:
                continue
            previous = before_map.get(self._evaluation_key(record))
            if previous is None:
                continue
            if (
                previous["status"] != record["status"]
                or previous["actual"] != record["actual"]
                or previous["offending_voxels"] != record["offending_voxels"]
            ):
                changes.append(
                    {
                        "regions": record["regions"],
                        "relation": record["relation"],
                        "label": record["label"],
                        "before": {"status": previous["status"], "actual": previous["actual"]},
                        "after": {"status": record["status"], "actual": record["actual"]},
                        "offending_voxels": record["offending_voxels"],
                        "manual_origin": record["manual_origin"],
                    }
                )
        return changes

    def apply_correction(
        self,
        region: str,
        label: str,
        mask: np.ndarray,
        use_as_basis: bool = False,
    ) -> dict[str, object]:
        """Correct one region at one timepoint and report the relation impact.

        The correction is stored verbatim; voxels violating a declared
        relation (e.g. a manual block outside its container) are kept and
        reported as conflicts, never clipped to satisfy the expectation.
        """
        before = self.evaluate_relations()
        self.region_session(region).apply_correction(label, mask, use_as_basis=use_as_basis)
        after = self.evaluate_relations()
        return {
            "region": region,
            "label": label,
            "relation_changes": self._relation_changes(before, after, region=region),
            "conflicts": [c for c in after if c["status"] == "conflict"],
        }

    def withdraw_correction(self, region: str, label: str) -> dict[str, object]:
        """Withdraw a correction of one region; other regions stay untouched."""
        before = self.evaluate_relations()
        self.region_session(region).withdraw_correction(label)
        after = self.evaluate_relations()
        return {
            "region": region,
            "label": label,
            "relation_changes": self._relation_changes(before, after, region=region),
            "conflicts": [c for c in after if c["status"] == "conflict"],
        }

    def preview_candidate_selection(self, label: str, candidate_id: str) -> dict[str, object]:
        """Merged preview across regions plus the relations to re-review."""
        per_region = {
            name: session.preview_candidate_selection(label, candidate_id)
            for name, session in self._sessions.items()
        }
        affected_labels: set[str] = {label}
        for preview in per_region.values():
            affected_labels.update(preview["auto_to_update"])
            affected_labels.update(preview["manual_conflicts"])
        return {
            "label": label,
            "candidate_id": candidate_id,
            "per_region": per_region,
            "relations_to_review": sorted(
                {
                    self._evaluation_key(record)
                    for record in self.evaluate_relations()
                    if record["label"] in affected_labels
                }
            ),
        }

    def apply_candidate_selection(
        self,
        label: str,
        candidate_id: str,
        keep_manual: bool = True,
    ) -> dict[str, object]:
        """Re-select a candidate and re-review only the affected relations.

        Relations at timepoints untouched by the new selection are not
        re-evaluated in the report; with keep_manual=True manual
        corrections (and their conflicts) are preserved as-is.
        """
        preview = self.preview_candidate_selection(label, candidate_id)
        affected_labels = {key[3] for key in preview["relations_to_review"]}
        before = self.evaluate_relations()
        per_region_impact = {}
        for name, session in self._sessions.items():
            per_region_impact[name] = session.apply_candidate_selection(
                label, candidate_id, keep_manual=keep_manual
            )
        self.series = self._sessions[next(iter(self._sessions))].series
        after = self.evaluate_relations()
        return {
            "label": label,
            "candidate_id": candidate_id,
            "per_region": per_region_impact,
            "relations_reviewed": preview["relations_to_review"],
            "relation_changes": self._relation_changes(before, after, labels=affected_labels),
            "conflicts": [c for c in after if c["status"] == "conflict"],
        }

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, object]:
        """Full export: every relation evaluation carries its provenance."""
        return {
            "labels": list(self.labels),
            "min_region_coverage": self.min_region_coverage,
            "regions": {
                name: {
                    "anchor_label": session.anchor_label,
                    "corrections": {
                        label: {"use_as_basis": label in session.basis_labels}
                        for label in sorted(session.corrections)
                    },
                }
                for name, session in sorted(self._sessions.items())
            },
            "expectations": [
                {
                    "region_a": expectation.region_a,
                    "region_b": expectation.region_b,
                    "relation": expectation.relation,
                }
                for expectation in self.expectations
            ],
            "evaluations": self.evaluate_relations(),
            "conflicts": self.conflicts(),
        }

