
from __future__ import annotations

import hashlib
import itertools
import json
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


def _face_boundary_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Count voxel-face contacts without double-counting shared faces."""
    count = 0
    for axis in range(mask_a.ndim):
        a_face = np.take(mask_a, np.arange(mask_a.shape[axis] - 1), axis=axis)
        b_face = np.take(mask_b, np.arange(1, mask_b.shape[axis]), axis=axis)
        count += int(np.logical_and(a_face, b_face).sum())
    return count


def _voxel_range(mask: np.ndarray | None) -> dict[str, object] | None:
    if mask is None or not bool(mask.any()):
        return None
    coords = np.column_stack(np.where(mask))
    lower = coords.min(axis=0)
    upper = coords.max(axis=0)
    return {
        "min": [int(value) for value in lower],
        "max": [int(value) for value in upper],
        "voxels": int(mask.sum()),
    }


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
        intersection = np.logical_and(region_a.mask, region_b.mask)
        outside_mask = np.logical_and(region_b.mask, ~region_a.mask)
        shared_boundary_voxels = _face_boundary_count(region_a.mask, region_b.mask)
        spatial_evidence = {
            "intersection_voxels": int(intersection.sum()),
            "shared_boundary_voxels": shared_boundary_voxels,
            "outside_voxels": int(outside_mask.sum()),
            "outside_voxel_range": _voxel_range(outside_mask),
            "region_voxel_ranges": {
                expectation.region_a: _voxel_range(region_a.mask),
                expectation.region_b: _voxel_range(region_b.mask),
            },
        }
        satisfied = actual == expectation.relation
        manual_origin = region_a.kind == "manual" or region_b.kind == "manual"
        record.update(
            status="satisfied" if satisfied else "conflict",
            actual=actual,
            offending_voxels=offending,
            manual_origin=manual_origin,
            spatial_evidence=spatial_evidence,
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

    def _three_way_conflicts(self, label: str) -> list[dict[str, object]]:
        """Check consequences that follow from actual mask geometry.

        Contact is deliberately not treated as transitive. The two rules
        checked here are transitivity of containment and the fact that a
        region contained in a mask cannot touch/overlap a mask separated
        from its container.
        """
        names = self.region_names
        actual: dict[tuple[str, str], str] = {}
        masks: dict[str, np.ndarray | None] = {}
        for name in names:
            masks[name] = self._sessions[name].region(label).mask
        for left, right in itertools.permutations(names, 2):
            if masks[left] is None or masks[right] is None:
                continue
            actual[(left, right)], _ = classify_pair(masks[left], masks[right])

        conflicts: list[dict[str, object]] = []
        for first, second, third in itertools.permutations(names, 3):
            if (
                actual.get((first, second)) == "contains"
                and actual.get((second, third)) == "contains"
                and actual.get((first, third)) != "contains"
            ):
                conflicts.append(
                    {
                        "label": label,
                        "regions": [first, second, third],
                        "rule": "contains_transitive",
                        "actual": [
                            actual.get((first, second)),
                            actual.get((second, third)),
                            actual.get((first, third)),
                        ],
                        "sources": {
                            name: self._basis_provenance(name, label) for name in (first, second, third)
                        },
                    }
                )
            if (
                actual.get((first, second)) == "contains"
                and actual.get((first, third)) == "separate"
                and actual.get((second, third)) != "separate"
            ):
                conflicts.append(
                    {
                        "label": label,
                        "regions": [first, second, third],
                        "rule": "contained_region_must_remain_separate",
                        "actual": [
                            actual.get((first, second)),
                            actual.get((first, third)),
                            actual.get((second, third)),
                        ],
                        "sources": {
                            name: self._basis_provenance(name, label) for name in (first, second, third)
                        },
                    }
                )
        return conflicts

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

    def three_way_conflicts(self) -> list[dict[str, object]]:
        """All geometrically inconsistent triples across timepoints."""
        conflicts: list[dict[str, object]] = []
        for label in self.labels:
            conflicts.extend(self._three_way_conflicts(label))
        return conflicts

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
    # joint, relation-constrained candidate planning
    # ------------------------------------------------------------------ #

    def _clone_sessions(self) -> dict[str, PropagationSession]:
        return {name: session.clone() for name, session in self._sessions.items()}

    @staticmethod
    def _set_candidate_for_preview(session: PropagationSession, label: str, candidate_id: str) -> None:
        session.candidate_overrides[label] = candidate_id
        session._recompute()

    @staticmethod
    def _digest_clients(clients: dict[str, object]) -> str:
        payload = json.dumps(clients, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _input_fingerprint(self) -> dict[str, object]:
        selections: dict[str, dict[str, object]] = {}
        for label in self.labels:
            tp = self.series.timepoint(label)
            multi = tp.multi
            selections[label] = {
                "status": tp.status,
                "selected": multi.selected.candidate_id if multi and multi.selected else None,
                "candidates": [
                    {
                        "id": candidate.candidate_id,
                        "status": candidate.result.status,
                        "parameters": candidate.result.transform.parameters().tolist()
                        if candidate.result.transform is not None
                        else None,
                    }
                    for candidate in multi.candidates
                ]
                if multi is not None
                else [],
            }
        manual: dict[str, dict[str, dict[str, object]]] = {}
        for region, session in self._sessions.items():
            manual[region] = {}
            for label, mask in sorted(session.corrections.items()):
                manual[region][label] = {
                    "digest": hashlib.sha256(np.asarray(mask, dtype=bool).tobytes()).hexdigest(),
                    "use_as_basis": label in session.basis_labels,
                    "candidate_override": session.candidate_overrides.get(label),
                }
        expectations = [
            (item.region_a, item.region_b, item.relation) for item in self.expectations
        ]
        return {
            "labels": list(self.labels),
            "selections": selections,
            "region_candidate_overrides": {
                region: dict(session.candidate_overrides)
                for region, session in self._sessions.items()
            },
            "manual": manual,
            "expectations": expectations,
        }

    def preview_joint_selection(
        self,
        label: str | None = None,
        labels: list[str] | None = None,
        objective: str = "relation_priority",
        include_current: bool = True,
    ) -> dict[str, object]:
        """Enumerate per-region candidate combinations and test real masks.

        Candidate choices are never OR-ed from per-region pass labels. Each
        complete combination is applied to copied propagation sessions and
        every declared pair, plus geometric three-way consequences, is
        recalculated from the resulting voxel masks. The live session is not
        modified.
        """
        if objective not in {"changed_voxels", "relation_priority"}:
            raise ValueError(
                "objective must be 'changed_voxels' or 'relation_priority'"
            )
        if label is not None and labels is not None:
            raise ValueError("pass either label or labels, not both")
        target_labels = [label] if label is not None else list(labels or self.labels)
        unknown = [item for item in target_labels if item not in self.labels]
        if unknown:
            raise KeyError(f"unknown timepoint labels {unknown!r}")

        domains: list[tuple[str, str, list[dict[str, str]]]] = []
        for target in target_labels:
            tp = self.series.timepoint(target)
            if tp.is_reference or tp.multi is None:
                continue
            successful_ids = [
                candidate.candidate_id
                for candidate in tp.multi.candidates
                if candidate.result.status == "success"
            ]
            current_id = (
                tp.multi.selected.candidate_id if tp.multi.selected is not None else None
            )
            for region, session in self._sessions.items():
                actual_id = session.candidate_overrides.get(target, current_id)
                options: list[dict[str, str]] = [
                    {"kind": "auto", "candidate_id": candidate_id}
                    for candidate_id in successful_ids
                ]
                if target in session.corrections and current_id is not None:
                    options.append({"kind": "manual", "candidate_id": current_id})
                if options:
                    domains.append((region, target, options))

        choices: list[dict[str, object]] = []
        if include_current:
            current: dict[str, object] = {
                (region, target): next(
                    option
                    for option in options
                    if option["candidate_id"] == actual_id
                    and (
                        option["kind"] == "manual"
                        if target in self._sessions[region].corrections
                        else option["kind"] == "auto"
                    )
                )
                for region, target, options in domains
            }
            current["_current"] = True
            choices.append(current)
        for values in itertools.product(*(options for _, _, options in domains)):
            choice = {
                (region, target): option
                for (region, target, _), option in zip(domains, values)
            }
            choice["_current"] = False
            choices.append(choice)

        solutions: list[dict[str, object]] = []
        blocked: list[dict[str, object]] = []
        target_set = set(target_labels)
        for choice in choices:
            tentative_sessions = self._clone_sessions()
            selections: dict[str, dict[str, str]] = {}
            for region, session in tentative_sessions.items():
                selections[region] = {}
                for target_region, target, _ in domains:
                    if target_region != region:
                        continue
                    option = choice[(region, target)]
                    candidate_id = option["candidate_id"]
                    self._set_candidate_for_preview(session, target, candidate_id)
                    if option["kind"] == "auto" and target in session.corrections:
                        session.corrections.pop(target, None)
                        session.basis_labels.discard(target)
                        session._recompute()
                    selections[region][target] = {
                        "source_kind": option["kind"],
                        "candidate_id": candidate_id,
                        "uses_manual_block": option["kind"] == "manual",
                    }

            original_sessions = self._sessions
            self._sessions = tentative_sessions
            try:
                evaluations = self.evaluate_relations()
                three_way = [
                    conflict
                    for conflict in self.three_way_conflicts()
                    if conflict["label"] in target_set
                ]
                relevant = [record for record in evaluations if record["label"] in target_set]
                pending = [record for record in relevant if record["status"] == "pending"]
                pair_conflicts = [
                    record for record in relevant if record["status"] == "conflict"
                ]
                changed_voxels = 0
                changed_ranges: dict[str, dict[str, object]] = {}
                for region, tentative in tentative_sessions.items():
                    changed_ranges[region] = {}
                    for target in target_labels:
                        old_region = original_sessions[region].region(target)
                        new_region = tentative.region(target)
                        old_mask = old_region.mask
                        new_mask = new_region.mask
                        if old_mask is None and new_mask is None:
                            continue
                        if old_mask is None or new_mask is None:
                            changed = old_mask if old_mask is not None else new_mask
                            count = int(changed.sum())
                        else:
                            changed = np.logical_xor(old_mask, new_mask)
                            count = int(changed.sum())
                        changed_voxels += count
                        changed_ranges[region][target] = {
                            "voxels": count,
                            "range": _voxel_range(changed),
                        }
            finally:
                self._sessions = original_sessions

            solution = {
                "selections": selections,
                "satisfied_relations": sum(
                    1 for record in relevant if record["status"] == "satisfied"
                ),
                "conflict_relations": len(pair_conflicts),
                "pending_relations": len(pending),
                "three_way_conflicts": three_way,
                "changed_voxels": changed_voxels,
                "changed_voxel_ranges": changed_ranges,
                "evaluations": relevant,
                "pending": pending,
                "compatible": (
                    not pending and not pair_conflicts and not three_way
                ),
                "is_current": bool(choice["_current"]),
            }
            if pending:
                blocked.append(
                    {
                        "selections": selections,
                        "pending_relations": pending,
                        "blocked_ranges": self._blocked_ranges(tentative_sessions, target_set),
                    }
                )
            solutions.append(solution)

        def score(item: dict[str, object]) -> tuple[float, int, int, int]:
            relation_score = item["satisfied_relations"] - (
                10 * item["conflict_relations"]
                + 3 * item["pending_relations"]
                + 10 * len(item["three_way_conflicts"])
            )
            if objective == "changed_voxels":
                return (
                    int(item["compatible"]),
                    -int(item["changed_voxels"]),
                    relation_score,
                    -int(item["pending_relations"]),
                )
            return (
                int(item["compatible"]),
                relation_score,
                -int(item["changed_voxels"]),
                -int(item["pending_relations"]),
            )

        ranked = sorted(solutions, key=score, reverse=True)
        def plan_key(item: dict[str, object]) -> str:
            return self._digest_clients(item["selections"])
        for item in ranked:
            item["solution_id"] = plan_key(item)
        fingerprint = self._input_fingerprint()
        return {
            "labels": list(target_labels),
            "objective": objective,
            "input_digest": self._digest_clients(fingerprint),
            "compatible_solutions": [item for item in ranked if item["compatible"]],
            "all_solutions": ranked,
            "blocked": blocked,
            "recommended": ranked[0] if ranked and ranked[0]["compatible"] else None,
        }

    def _blocked_ranges(
        self, sessions: dict[str, PropagationSession], labels: set[str]
    ) -> list[dict[str, object]]:
        blocked: list[dict[str, object]] = []
        for region, session in sessions.items():
            for label in labels:
                propagated = session.region(label)
                if propagated.mask is None and propagated.blocked_voxel_range is not None:
                    blocked.append(
                        {
                            "region": region,
                            "label": label,
                            "reasons": list(propagated.reasons),
                            "range": propagated.blocked_voxel_range,
                        }
                    )
        return blocked

    def confirm_joint_selection(
        self,
        preview: dict[str, object],
        solution: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Validate a preview against current inputs, then adopt one plan.

        Adopted candidate IDs remain per-region propagation overrides; the
        underlying shared series is rewritten only when the same candidate is
        chosen at a label by every region participating in that label.
        """
        fingerprint = self._input_fingerprint()
        current_digest = self._digest_clients(fingerprint)
        if preview.get("input_digest") != current_digest:
            raise ValueError(
                "joint selection inputs changed; call preview_joint_selection() again "
                "before confirming a plan"
            )
        chosen = solution or preview.get("recommended")
        if not chosen:
            raise ValueError("no joint candidate solution is available")
        if not chosen.get("compatible"):
            raise ValueError("refusing to confirm a plan with unresolved relation conflicts")
        known_ids = {item.get("solution_id") for item in preview["all_solutions"]}
        if chosen.get("solution_id") not in known_ids:
            raise ValueError("solution does not belong to the supplied preview")

        before = self.evaluate_relations()
        target_labels = list(preview["labels"])
        selections: dict[str, dict[str, str]] = chosen["selections"]
        per_label: dict[str, set[str]] = {}
        for region, region_selections in selections.items():
            if region not in self._sessions:
                raise KeyError(f"unknown region {region!r}")
            for label, item in region_selections.items():
                candidate_id = item["candidate_id"]
                self._sessions[region].set_region_candidate(label, candidate_id)
                if not item.get("uses_manual_block", False):
                    self._sessions[region].corrections.pop(label, None)
                    self._sessions[region].basis_labels.discard(label)
                    self._sessions[region]._recompute()
                per_label.setdefault(label, set()).add(candidate_id)

        # Keep the shared provenance representative when there is no disagreement.
        for label, candidate_ids in per_label.items():
            if len(candidate_ids) == 1:
                candidate_id = next(iter(candidate_ids))
                if self.series.timepoint(label).multi.selected.candidate_id != candidate_id:
                    self.series = select_series_candidate(
                        self.series, label, candidate_id
                    )
                for session in self._sessions.values():
                    session.series = self.series
                    session.candidate_overrides.pop(label, None)
                    session._recompute()
        after = self.evaluate_relations()
        return {
            "confirmed": True,
            "labels": target_labels,
            "selections": selections,
            "changed_voxels": chosen["changed_voxels"],
            "relation_changes": self._relation_changes(
                before, after, labels=set(target_labels)
            ),
            "final_evaluations": [
                record for record in after if record["label"] in set(target_labels)
            ],
            "three_way_conflicts": [
                conflict
                for conflict in self.three_way_conflicts()
                if conflict["label"] in set(target_labels)
            ],
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
            "three_way_conflicts": self.three_way_conflicts(),
        }
