from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .image import MedicalImage
from .register import RegistrationResult, register_rigid
from .roi import RegionOfInterest
from .transform import RigidTransform


@dataclass
class InitializationSpec:
    candidate_id: str
    initial_parameters: list[float]
    label: str = ""
    source: str = "automatic"
    lock_initial: bool = False


@dataclass
class CandidateRecord:
    spec: InitializationSpec
    result: RegistrationResult

    def summary(self) -> dict[str, Any]:
        summary = self.result.summary()
        summary["candidate_id"] = self.spec.candidate_id
        summary["label"] = self.spec.label
        summary["source"] = self.spec.source
        summary["initial_parameters"] = list(self.spec.initial_parameters)
        return summary


@dataclass
class MultiRegistrationResult:
    candidates: list[CandidateRecord]
    selected_candidate_id: str | None
    status: str
    uncertainty: dict[str, object]
    roi: RegionOfInterest | None = None
    acceptance: dict[str, float] = field(default_factory=dict)

    def selected_candidate(self) -> CandidateRecord | None:
        for candidate in self.candidates:
            if candidate.spec.candidate_id == self.selected_candidate_id:
                return candidate
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_candidate_id": self.selected_candidate_id,
            "uncertainty": self.uncertainty,
            "acceptance": self.acceptance,
            "candidates": [candidate.summary() for candidate in self.candidates],
        }


def _candidate_score(result: RegistrationResult) -> float:
    ncc = result.final_ncc
    roi_ncc = ncc
    if result.roi_error is not None and "ncc" in result.roi_error:
        roi_ncc = result.roi_error["ncc"]
    return float(2.0 * roi_ncc + ncc)


def _disagreement(results: list[RegistrationResult]) -> dict[str, float]:
    rotations = [Rotation.from_matrix(r.transform.rotation) for r in results]
    translations = np.stack([r.transform.translation for r in results], axis=0)
    anchor = rotations[0]
    angle_degrees = [float((anchor.inv() * rot).magnitude()) * 180.0 / np.pi for rot in rotations[1:]]
    max_translation = float(np.max(np.linalg.norm(translations[1:] - translations[0], axis=1))) if len(results) > 1 else 0.0
    max_angle = float(max(angle_degrees, default=0.0))
    return {"max_rotation_diff_degrees": max_angle, "max_translation_diff_mm": max_translation}


def register_multi(
    fixed: MedicalImage,
    moving: MedicalImage,
    initializations: list[InitializationSpec],
    levels: tuple[int, ...] = (4, 2, 1),
    max_iterations: int = 120,
    min_coverage: float = 0.35,
    roi: RegionOfInterest | None = None,
    min_roi_coverage: float = 0.6,
    min_roi_ncc: float = 0.75,
    roi_weight: float = 2.0,
    rotation_tolerance_degrees: float = 3.0,
    translation_tolerance_mm: float = 3.0,
    score_tolerance: float = 0.05,
    manual_candidate_id: str | None = None,
) -> MultiRegistrationResult:
    if not initializations:
        raise ValueError("at least one initialization is required")
    ids = [spec.candidate_id for spec in initializations]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")

    candidates: list[CandidateRecord] = []
    for spec in initializations:
        initial = RigidTransform.from_parameters(np.asarray(spec.initial_parameters, dtype=float))
        result = register_rigid(
            fixed,
            moving,
            initial,
            levels=levels,
            max_iterations=max_iterations,
            min_coverage=min_coverage,
            roi=roi,
            min_roi_coverage=min_roi_coverage,
            min_roi_ncc=min_roi_ncc,
            roi_weight=roi_weight,
            candidate_id=spec.candidate_id,
            lock_initial=spec.lock_initial,
        )
        candidates.append(CandidateRecord(spec, result))

    accepted = [candidate for candidate in candidates if candidate.result.accepted]
    scores = {candidate.spec.candidate_id: _candidate_score(candidate.result) for candidate in accepted}

    if manual_candidate_id is not None:
        if manual_candidate_id not in ids:
            raise ValueError(f"manual candidate {manual_candidate_id!r} does not exist")
        chosen = next(candidate for candidate in candidates if candidate.spec.candidate_id == manual_candidate_id)
        if not chosen.result.accepted:
            raise ValueError(f"manual candidate {manual_candidate_id!r} failed acceptance and cannot be selected")

    selected_id: str | None = None
    status = "failed"
    uncertainty: dict[str, object] = {
        "ambiguous": False,
        "reason": "no accepted candidate" if not accepted else "",
    }
    acceptance = {
        "rotation_tolerance_degrees": float(rotation_tolerance_degrees),
        "translation_tolerance_mm": float(translation_tolerance_mm),
        "score_tolerance": float(score_tolerance),
    }

    if accepted:
        if manual_candidate_id is not None:
            selected_id = manual_candidate_id
            status = "selected_manual"
            uncertainty = {
                "ambiguous": True,
                "reason": "manually selected candidate retained; verify target alignment",
            }
        else:
            ordered = sorted(accepted, key=lambda candidate: scores[candidate.spec.candidate_id], reverse=True)
            best = ordered[0]
            best_score = scores[best.spec.candidate_id]
            near_best = [
                candidate
                for candidate in ordered
                if best_score - scores[candidate.spec.candidate_id] <= score_tolerance
            ]
            disagreement = _disagreement([candidate.result for candidate in near_best])
            consistent = (
                len(near_best) == 1
                or (
                    disagreement["max_rotation_diff_degrees"] <= rotation_tolerance_degrees
                    and disagreement["max_translation_diff_mm"] <= translation_tolerance_mm
                )
            )
            if consistent:
                selected_id = best.spec.candidate_id
                status = "agreed"
                uncertainty = {
                    "ambiguous": False,
                    "reason": "",
                    "near_best_candidate_ids": [candidate.spec.candidate_id for candidate in near_best],
                    **disagreement,
                }
            else:
                selected_id = best.spec.candidate_id
                status = "uncertain"
                uncertainty = {
                    "ambiguous": True,
                    "reason": (
                        "accepted candidates have similar scores but disagreeing transforms; "
                        "candidates retained for manual review"
                    ),
                    "near_best_candidate_ids": [candidate.spec.candidate_id for candidate in near_best],
                    **disagreement,
                }

    return MultiRegistrationResult(candidates, selected_id, status, uncertainty, roi, acceptance)
