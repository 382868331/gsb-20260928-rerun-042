"""Joint constraint selection of multi-region correction candidates.

A researcher declares the spatial relations to maintain (contains /
separate / contact, see medreg.relations) and offers, per region,
alternative propagation candidates and manual blocks at one timepoint.
The planner evaluates every combination of options against the declared
relations using the actual voxel masks -- never by merging per-candidate
pass labels -- and reports which combinations satisfy all constraints.

Options that cannot be used (failed candidate, missing propagation
dependency, region landing outside the field of view) are kept as
explicit blocked entries with their concrete blocking range; the other,
independent regions still take part in the search.

Two rankings are reported side by side so the trade-off is visible:
fewest changed voxels versus prioritising the declared relations. Nothing is
written to any region mask during planning; only confirm() applies
a chosen plan, and it refuses to run when any input changed since the
preview was computed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field

import numpy as np

from .relations import MultiRegionSession, classify_pair
from .series import select_series_candidate

#: Ranking objectives reported by the planner.
OBJECTIVES = ("min_changed_voxels", "prioritize_relations")


def _mask_hash(mask: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _bounds(indices: np.ndarray) -> dict[str, list[int]]:
    return {
        "index_min": [int(v) for v in indices.min(axis=0)],
        "index_max": [int(v) for v in indices.max(axis=0)],
    }


def _shared_boundary_voxels(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Voxels of mask_b sharing a face with mask_a."""
    count = 0
    for axis in range(mask_a.ndim):
        for delta in (-1, 1):
            shifted = np.roll(mask_a, delta, axis=axis)
            edge = [slice(None)] * mask_a.ndim
            edge[axis] = 0 if delta == 1 else -1
            shifted[tuple(edge)] = False
            count += int(np.logical_and(shifted, mask_b).sum())
    return count


@dataclass
class JointOption:
    """One selectable option for a region at the planned timepoint."""

    region: str
    option_id: str
    kind: str  # current | candidate | manual
    candidate_id: str | None = None
    mask: np.ndarray | None = None
    blocked: bool = False
    blocking: dict[str, object] = field(default_factory=dict)
    changed_voxels: int = 0

    def report(self) -> dict[str, object]:
        return {
            "region": self.region,
            "option_id": self.option_id,
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "blocked": self.blocked,
            "blocking": dict(self.blocking),
            "changed_voxels": self.changed_voxels,
            "voxels": int(self.mask.sum()) if self.mask is not None else 0,
        }


@dataclass
class JointPreview:
    """Read-only planning result; confirm() applies a chosen plan."""

    label: str
    fingerprint: str
    options: dict[str, list[JointOption]]
    plans: list[dict[str, object]]
    rankings: dict[str, list[int]]
    excluded_regions: dict[str, object]
    manual_options: dict[str, list[np.ndarray]]

    def option(self, region: str, option_id: str) -> JointOption:
        for opt in self.options[region]:
            if opt.option_id == option_id:
                return opt
        raise KeyError(f"unknown option {option_id!r} for region {region!r}")

    def report(self) -> dict[str, object]:
        return {
            "label": self.label,
            "fingerprint": self.fingerprint,
            "options": {
                region: [opt.report() for opt in options]
                for region, options in sorted(self.options.items())
            },
            "excluded_regions": self.excluded_regions,
            "plans": self.plans,
            "rankings": self.rankings,
            "best_by_objective": {
                objective: (plan_ids[0] if plan_ids else None)
                for objective, plan_ids in self.rankings.items()
            },
        }


class JointSelectionPlanner:
    """Search compatible per-region options under declared relations."""

    def __init__(self, session: MultiRegionSession, max_combinations: int = 4096):
        self.session = session
        self.max_combinations = int(max_combinations)

    # ------------------------------------------------------------------ #
    # option construction
    # ------------------------------------------------------------------ #

    def _changed_voxels(self, mask: np.ndarray | None, current: np.ndarray | None) -> int:
        if mask is None:
            return 0
        if current is None:
            return int(mask.sum())
        return int(np.logical_xor(mask, current).sum())

    def _options_for(
        self, region: str, label: str, manual_masks: list[np.ndarray]
    ) -> list[JointOption]:
        propagation = self.session.region_session(region)
        current = propagation.region(label)
        current_mask = current.mask
        options: list[JointOption] = []

        current_option = JointOption(
            region=region,
            option_id="current",
            kind="current",
            mask=current_mask,
            changed_voxels=0,
        )
        if current_mask is None:
            current_option.blocked = True
            current_option.blocking = {
                "reason": "region is pending at this timepoint",
                "reasons": list(current.reasons),
            }
        options.append(current_option)

        tp = self.session.series.timepoint(label)
        if not tp.is_reference and tp.multi is not None:
            for candidate in tp.multi.candidates:
                preview = propagation.preview_candidate_region(
                    label, candidate.candidate_id
                )
                option = JointOption(
                    region=region,
                    option_id=f"candidate:{candidate.candidate_id}",
                    kind="candidate",
                    candidate_id=candidate.candidate_id,
                    mask=preview.mask,
                    changed_voxels=self._changed_voxels(preview.mask, current_mask),
                )
                if preview.out_of_fov:
                    option.blocking["out_of_fov"] = dict(preview.out_of_fov)
                if preview.mask is None:
                    option.blocked = True
                    option.blocking["reasons"] = list(preview.reasons)
                options.append(option)

        image = propagation.images[propagation.labels.index(label)]
        for index, mask in enumerate(manual_masks):
            array = np.asarray(mask, dtype=bool)
            option = JointOption(
                region=region,
                option_id=f"manual:{index}",
                kind="manual",
                mask=array,
                changed_voxels=self._changed_voxels(array, current_mask),
            )
            if array.shape != image.shape:
                option.blocked = True
                option.blocking["reason"] = (
                    f"manual block has shape {array.shape}, expected {image.shape}; "
                    "it does not lie on the image grid of this timepoint"
                )
                option.mask = None
            elif not array.any():
                option.blocked = True
                option.blocking["reason"] = "manual block is empty"
                option.mask = None
            options.append(option)
        return options

    # ------------------------------------------------------------------ #
    # joint evaluation on actual voxel masks
    # ------------------------------------------------------------------ #

    def _evaluate_combination(
        self, label: str, masks: dict[str, np.ndarray | None]
    ) -> tuple[list[dict[str, object]], int]:
        records: list[dict[str, object]] = []
        satisfied = 0
        for expectation in self.session.expectations:
            mask_a = masks.get(expectation.region_a)
            mask_b = masks.get(expectation.region_b)
            record: dict[str, object] = {
                "label": label,
                "relation": expectation.relation,
                "regions": [expectation.region_a, expectation.region_b],
            }
            if mask_a is None or mask_b is None:
                missing = [
                    name
                    for name, mask in (
                        (expectation.region_a, mask_a),
                        (expectation.region_b, mask_b),
                    )
                    if mask is None
                ]
                record.update(
                    status="pending",
                    actual=None,
                    reasons=[
                        f"no usable mask for {name} in this combination" for name in missing
                    ],
                )
                records.append(record)
                continue
            actual, offending = classify_pair(mask_a, mask_b)
            ok = actual == expectation.relation
            satisfied += int(ok)
            record.update(
                status="satisfied" if ok else "conflict",
                actual=actual,
                offending_voxels=offending,
            )
            outside = np.logical_and(mask_b, ~mask_a)
            if outside.any():
                record["offending_bounds"] = _bounds(
                    np.column_stack(np.where(outside))
                )
            if actual == "contact":
                record["shared_boundary_voxels"] = _shared_boundary_voxels(mask_a, mask_b)
            records.append(record)
        return records, satisfied

    # ------------------------------------------------------------------ #
    # fingerprint of every input the preview depends on
    # ------------------------------------------------------------------ #

    def _fingerprint(self, label: str, manual_options: dict[str, list[np.ndarray]]) -> str:
        regions: dict[str, object] = {}
        for name in self.session.region_names:
            propagation = self.session.region_session(name)
            regions[name] = {
                "anchor_label": propagation.anchor_label,
                "anchor_mask": _mask_hash(propagation.anchor_mask),
                "corrections": {
                    correction_label: _mask_hash(mask)
                    for correction_label, mask in sorted(propagation.corrections.items())
                },
                "basis_labels": sorted(propagation.basis_labels),
            }
        series_state = []
        for tp in self.session.series.timepoints:
            selected = tp.multi.selected if tp.multi is not None else None
            series_state.append(
                {
                    "label": tp.label,
                    "status": tp.status,
                    "selected_candidate_id": selected.candidate_id if selected else None,
                    "selected_transform": (
                        selected.result.transform.parameters().tolist() if selected else None
                    ),
                    "candidates": [
                        [c.candidate_id, c.result.status]
                        for c in (tp.multi.candidates if tp.multi is not None else [])
                    ],
                }
            )
        payload = {
            "label": label,
            "expectations": [
                [e.region_a, e.region_b, e.relation] for e in self.session.expectations
            ],
            "regions": regions,
            "series": series_state,
            "manual_options": {
                region: [_mask_hash(mask) for mask in masks]
                for region, masks in sorted(manual_options.items())
            },
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------ #
    # planning (read-only) and confirmation (the only write path)
    # ------------------------------------------------------------------ #

    def plan(
        self,
        label: str,
        manual_options: dict[str, list[np.ndarray]] | None = None,
    ) -> JointPreview:
        """Evaluate every combination of per-region options at the given label.

        The search is purely observational: no region mask, correction or
        candidate adoption is modified. Blocked options keep their concrete
        blocking range and regions without any usable option are excluded
        from the product while the remaining regions are still searched.
        """
        if label not in self.session.labels:
            raise KeyError(f"unknown timepoint label {label!r}")
        manual_options = {
            region: [np.asarray(mask, dtype=bool) for mask in masks]
            for region, masks in (manual_options or {}).items()
        }
        for region in manual_options:
            if region not in self.session.region_names:
                raise KeyError(f"unknown region {region!r}")

        options: dict[str, list[JointOption]] = {}
        excluded: dict[str, object] = {}
        searchable: list[str] = []
        for region in self.session.region_names:
            region_options = self._options_for(
                region, label, manual_options.get(region, [])
            )
            options[region] = region_options
            usable = [opt for opt in region_options if not opt.blocked]
            if usable:
                searchable.append(region)
            else:
                excluded[region] = {
                    "reason": "no usable option at this timepoint; the region "
                    "is excluded from the joint search but other regions continue",
                    "options": [opt.report() for opt in region_options],
                }

        combination_count = 1
        for region in searchable:
            combination_count *= sum(1 for opt in options[region] if not opt.blocked)
        if combination_count > self.max_combinations:
            raise ValueError(
                f"joint search would evaluate {combination_count} combinations, "
                f"above the limit of {self.max_combinations}; narrow the offered "
                "candidates or manual blocks"
            )

        plans: list[dict[str, object]] = []
        usable_per_region = [
            [opt for opt in options[region] if not opt.blocked] for region in searchable
        ]
        total = len(self.session.expectations)
        for plan_id, combination in enumerate(itertools.product(*usable_per_region)):
            masks = {region: None for region in excluded}
            choices: dict[str, str] = {}
            changed = 0
            for region, opt in zip(searchable, combination):
                masks[region] = opt.mask
                choices[region] = opt.option_id
                changed += opt.changed_voxels
            records, satisfied = self._evaluate_combination(label, masks)
            plans.append(
                {
                    "plan_id": plan_id,
                    "choices": choices,
                    "satisfied_relations": satisfied,
                    "total_relations": total,
                    "all_satisfied": satisfied == total,
                    "changed_voxels": changed,
                    "relations": records,
                }
            )

        rankings = {
            "min_changed_voxels": [
                plan["plan_id"]
                for plan in sorted(
                    plans,
                    key=lambda p: (p["changed_voxels"], -p["satisfied_relations"]),
                )
            ],
            "prioritize_relations": [
                plan["plan_id"]
                for plan in sorted(
                    plans,
                    key=lambda p: (-p["satisfied_relations"], p["changed_voxels"]),
                )
            ],
        }
        return JointPreview(
            label=label,
            fingerprint=self._fingerprint(label, manual_options),
            options=options,
            plans=plans,
            rankings=rankings,
            excluded_regions=excluded,
            manual_options=manual_options,
        )

    def confirm(self, preview: JointPreview, plan_id: int) -> dict[str, object]:
        """Apply one planned combination; the only path that writes masks.

        The fingerprint of every input is recomputed; if anything changed
        since the preview (corrections, candidate adoptions, expectations,
        offered manual blocks) the confirmation is refused and the plan
        must be re-validated with a fresh plan() call.
        """
        current = self._fingerprint(preview.label, preview.manual_options)
        if current != preview.fingerprint:
            raise ValueError(
                "inputs changed since this preview was computed; re-run plan() "
                "and review the fresh candidates before confirming"
            )
        plan = next((p for p in preview.plans if p["plan_id"] == plan_id), None)
        if plan is None:
            raise KeyError(f"unknown plan_id {plan_id!r} in this preview")

        candidate_choices: list[tuple[str, str]] = []
        manual_choices: list[tuple[str, np.ndarray]] = []
        for region, option_id in plan["choices"].items():
            option = preview.option(region, option_id)
            if option.kind == "candidate":
                candidate_choices.append((region, option.candidate_id))
            elif option.kind == "manual":
                manual_choices.append((region, option.mask))

        # Adopt all candidate choices into one consistent series first, so
        # every region recomputes against the same final adoption state.
        series = self.session.series
        for _, candidate_id in candidate_choices:
            series = select_series_candidate(series, preview.label, candidate_id)
        for name in self.session.region_names:
            self.session.region_session(name).set_series(series)
        self.session.series = series
        for region, mask in manual_choices:
            self.session.apply_correction(region, preview.label, mask)

        relations = [
            record
            for record in self.session.evaluate_relations()
            if record["label"] == preview.label
        ]
        return {
            "label": preview.label,
            "plan_id": plan_id,
            "applied": dict(plan["choices"]),
            "adopted_candidates": {
                region: candidate_id for region, candidate_id in candidate_choices
            },
            "applied_manual_regions": [region for region, _ in manual_choices],
            "relations": relations,
            "conflicts": [r for r in relations if r["status"] == "conflict"],
        }

