from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .image import MedicalImage
from .metrics import image_error, normalized_cross_correlation, region_error
from .roi import RegionOfInterest
from .resample import coverage, resample
from .transform import RigidTransform


@dataclass
class RegistrationResult:
    status: str
    transform: RigidTransform | None
    transformed: MedicalImage | None
    overlap_mask: np.ndarray | None
    coverage: dict[str, object] | None
    error: dict[str, float] | None
    iterations: int
    converged: bool
    message: str
    initial_ncc: float
    final_ncc: float
    elapsed_seconds: float
    roi_error: dict[str, object] | None = None
    initial_parameters: list[float] | None = None

    def summary(self) -> dict[str, object]:
        params = self.transform.parameters().tolist() if self.transform else None
        return {
            "status": self.status,
            "converged": self.converged,
            "message": self.message,
            "iterations": self.iterations,
            "parameters_rotvec_xyz_translation_xyz": params,
            "rotation_degrees_xyz": Rotation.from_matrix(self.transform.rotation).as_euler("xyz", degrees=True).tolist()
            if self.transform
            else None,
            "coverage": self.coverage,
            "error": self.error,
            "roi_error": self.roi_error,
            "initial_ncc": self.initial_ncc,
            "final_ncc": self.final_ncc,
            "elapsed_seconds": self.elapsed_seconds,
            "initial_parameters": self.initial_parameters,
        }


@dataclass
class CandidateResult:
    """One registration started from a distinct initial position."""

    candidate_id: str
    initial_parameters: list[float]
    result: RegistrationResult

    def summary(self) -> dict[str, object]:
        payload = self.result.summary()
        payload["candidate_id"] = self.candidate_id
        return payload


@dataclass
class MultiStartResult:
    """All candidates are retained, including failed and ambiguous ones."""

    status: str
    selected: CandidateResult | None
    candidates: list[CandidateResult]
    uncertain: bool
    uncertainty_reasons: list[str]
    transformed: MedicalImage | None
    selection_method: str

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "uncertain": self.uncertain,
            "uncertainty_reasons": self.uncertainty_reasons,
            "selection_method": self.selection_method,
            "selected_candidate_id": self.selected.candidate_id if self.selected else None,
            "candidates": [candidate.summary() for candidate in self.candidates],
        }


def transform_disagreement(first: RigidTransform, second: RigidTransform) -> dict[str, float]:
    """Physical-space disagreement between two rigid solutions.

    Rotation distance is the geodesic angle; translation distance is reported
    directly in physical units together with the residual point error that
    the rotation induces at a 50 mm radius, so nearly identical scores with
    inconsistent geometry can be flagged.
    """

    relative = first.compose(second.inverse())
    angle = float(np.linalg.norm(Rotation.from_matrix(relative.rotation).as_rotvec()))
    translation = float(np.linalg.norm(relative.translation))
    induced = 2.0 * 50.0 * np.sin(angle / 2.0)
    return {
        "rotation_angle_deg": float(np.degrees(angle)),
        "translation_distance_mm": translation,
        "induced_point_error_mm_at_50mm": float(induced),
    }


def _downsample(image: MedicalImage, factor: int) -> MedicalImage:
    if factor <= 1:
        return image
    sigma = (2 * factor / 6.0) * np.ones(3)
    blurred = gaussian_filter(image.data, sigma=sigma, mode="constant")
    slices = tuple(slice(None, None, factor) for _ in range(3))
    data = blurred[slices]
    affine = image.affine.copy()
    affine[:3, :3] = image.affine[:3, :3] * factor
    affine[:3, 3] = image.affine[:3, 3]
    return MedicalImage(data, affine, image.metadata)


def register_rigid(
    fixed: MedicalImage,
    moving: MedicalImage,
    initial_transform: RigidTransform | None = None,
    levels: tuple[int, ...] = (4, 2, 1),
    max_iterations: int = 120,
    min_coverage: float = 0.35,
    roi: RegionOfInterest | None = None,
    min_roi_coverage: float = 0.6,
    min_roi_ncc: float = 0.7,
) -> RegistrationResult:
    start = perf_counter()
    initial_params = initial_transform.parameters() if initial_transform else np.zeros(6)
    current = initial_params.copy()
    iterations = 0
    first_fixed = _downsample(fixed, levels[0])
    first_moving = _downsample(moving, levels[0])
    first_roi_mask = roi.mask_on(first_fixed) if roi is not None else None
    initial_array, initial_mask = resample(
        first_moving, first_fixed, RigidTransform.from_parameters(initial_params), order=1
    )
    initial_ncc = normalized_cross_correlation(first_fixed.data, initial_array, initial_mask)

    for level in levels:
        f = _downsample(fixed, level)
        m = _downsample(moving, level)
        level_roi_mask = roi.mask_on(f) if roi is not None else None

        def objective(params: np.ndarray) -> float:
            transformed, mask = resample(m, f, RigidTransform.from_parameters(params), order=1)
            fraction = np.count_nonzero(mask) / mask.size
            if fraction < min_coverage:
                return 2.0 - fraction
            global_ncc = normalized_cross_correlation(f.data, transformed, mask)
            if level_roi_mask is None:
                return 1.05 - global_ncc
            region = mask & level_roi_mask
            roi_fraction = np.count_nonzero(region) / max(np.count_nonzero(level_roi_mask), 1)
            if roi_fraction < min_roi_coverage:
                return 2.0 - roi_fraction
            roi_ncc = normalized_cross_correlation(f.data, transformed, region)
            if roi_ncc <= -1.0:
                return 2.0
            # The selected region drives the fit while global agreement acts
            # only as a weak regularizer, so background cannot dominate.
            return 1.05 - (0.75 * roi_ncc + 0.25 * global_ncc)

        result = minimize(
            objective,
            current,
            method="Powell",
            options={"maxiter": max_iterations, "xtol": 2e-4, "ftol": 2e-5, "disp": False},
        )
        iterations += int(getattr(result, "nit", 0))
        current = np.asarray(result.x, dtype=np.float64)

    transform = RigidTransform.from_parameters(current)
    transformed_array, mask = resample(moving, fixed, transform, order=1)
    info = coverage(mask, fixed)
    fraction = float(info["coverage_fraction"])
    errors = image_error(fixed.data, transformed_array, mask)
    ncc = errors["ncc"]
    roi_errors = None
    roi_ncc = ncc
    roi_fraction = 1.0
    if roi is not None:
        roi_mask = roi.mask_on(fixed)
        roi_errors = region_error(fixed.data, transformed_array, mask, roi_mask)
        roi_ncc = float(roi_errors["ncc"])
        roi_fraction = float(roi_errors["region_coverage_fraction"])
    converged = bool(
        result.success
        and fraction >= min_coverage
        and ncc >= 0.75
        and np.isfinite(current).all()
        and (roi is None or (roi_ncc >= min_roi_ncc and roi_fraction >= min_roi_coverage))
    )
    status = "success" if converged else "failed"
    if converged:
        message = str(result.message)
    else:
        message = (
            f"registration did not meet acceptance: optimizer={result.success}, ncc={ncc:.4f}, "
            f"coverage={fraction:.4f}"
        )
        if roi is not None:
            message += f", roi_ncc={roi_ncc:.4f}, roi_coverage={roi_fraction:.4f}"
    transformed = MedicalImage(transformed_array, fixed.affine, {"resampled_to": "fixed"}) if converged else None
    return RegistrationResult(
        status,
        transform,
        transformed,
        mask,
        info,
        errors,
        iterations,
        converged,
        message,
        initial_ncc,
        ncc,
        perf_counter() - start,
        roi_errors,
        initial_params.tolist(),
    )


def register_multi(
    fixed: MedicalImage,
    moving: MedicalImage,
    initial_parameters: list[list[float] | np.ndarray] | None = None,
    roi: RegionOfInterest | None = None,
    levels: tuple[int, ...] = (4, 2, 1),
    max_iterations: int = 120,
    min_coverage: float = 0.35,
    min_roi_coverage: float = 0.6,
    min_roi_ncc: float = 0.7,
    disagreement_translation_mm: float = 3.0,
    disagreement_rotation_deg: float = 5.0,
    ambiguous_score_gap: float = 0.03,
) -> MultiStartResult:
    """Run registration from several initial positions and keep all results.

    Candidates with similar quality but inconsistent geometry are retained and
    reported as uncertain instead of silently picking one transform.
    """

    if not initial_parameters:
        initial_parameters = [np.zeros(6)]
    candidates: list[CandidateResult] = []
    for index, params in enumerate(initial_parameters):
        initial = RigidTransform.from_parameters(np.asarray(params, dtype=float))
        outcome = register_rigid(
            fixed,
            moving,
            initial,
            levels,
            max_iterations,
            min_coverage,
            roi,
            min_roi_coverage,
            min_roi_ncc,
        )
        candidates.append(CandidateResult(f"candidate_{index:02d}", np.asarray(params, dtype=float).tolist(), outcome))

    successful = [candidate for candidate in candidates if candidate.result.status == "success"]
    uncertainty: list[str] = []
    selected: CandidateResult | None = None

    if successful:

        def score(candidate: CandidateResult) -> float:
            outcome = candidate.result
            if roi is not None and outcome.roi_error is not None:
                return 0.75 * float(outcome.roi_error["ncc"]) + 0.25 * outcome.final_ncc
            return outcome.final_ncc

        ranked = sorted(successful, key=score, reverse=True)
        selected = ranked[0]
        best_score = score(selected)
        close = [candidate for candidate in ranked[1:] if best_score - score(candidate) <= ambiguous_score_gap]
        for candidate in close:
            disagreement = transform_disagreement(selected.result.transform, candidate.result.transform)
            if (
                disagreement["translation_distance_mm"] > disagreement_translation_mm
                or disagreement["rotation_angle_deg"] > disagreement_rotation_deg
            ):
                uncertainty.append(
                    f"{selected.candidate_id} and {candidate.candidate_id} score within {ambiguous_score_gap:.3f} "
                    f"but disagree by {disagreement['translation_distance_mm']:.2f} mm and "
                    f"{disagreement['rotation_angle_deg']:.2f} deg"
                )
        if roi is not None and selected.result.roi_error is not None:
            if float(selected.result.roi_error["region_coverage_fraction"]) < 1.0:
                uncertainty.append(
                    "selected candidate does not fully cover the region of interest: "
                    f"{float(selected.result.roi_error['region_coverage_fraction']):.3f}"
                )

    if selected is None:
        status = "failed"
        uncertainty.append(f"none of {len(candidates)} initial positions met the acceptance criteria")
    elif uncertainty:
        status = "uncertain"
    else:
        status = "success"

    return MultiStartResult(
        status,
        selected,
        candidates,
        bool(uncertainty and selected is not None),
        uncertainty,
        selected.result.transformed if selected else None,
        "auto_best_score",
    )


def apply_manual_selection(multi: MultiStartResult, candidate_id: str) -> MultiStartResult:
    """Promote a retained candidate after expert review.

    The choice is recorded as a manual selection; geometry is never re-run and
    failed candidates cannot be promoted into a successful image.
    """

    chosen = next((candidate for candidate in multi.candidates if candidate.candidate_id == candidate_id), None)
    if chosen is None:
        raise ValueError(f"unknown candidate_id {candidate_id!r}; candidates are retained in the report")
    if chosen.result.status != "success":
        raise ValueError(
            f"candidate {candidate_id} did not pass acceptance; it is retained as a failed candidate "
            "and cannot produce a success image"
        )
    reasons = list(multi.uncertainty_reasons)
    reasons.append(f"manual selection of {candidate_id} from {len(multi.candidates)} retained candidates")
    return MultiStartResult(
        multi.status,
        chosen,
        multi.candidates,
        multi.uncertain,
        reasons,
        chosen.result.transformed,
        f"manual:{candidate_id}",
    )
