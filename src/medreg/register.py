from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .image import MedicalImage
from .metrics import image_error, masked_error, normalized_cross_correlation
from .resample import coverage, resample
from .roi import RegionOfInterest
from .transform import RigidTransform


@dataclass
class RegistrationResult:
    status: str
    transform: RigidTransform
    transformed: MedicalImage | None
    overlap_mask: np.ndarray | None
    coverage: dict[str, object] | None
    error: dict[str, float] | None
    roi_coverage: dict[str, object] | None
    roi_error: dict[str, float] | None
    iterations: int
    converged: bool
    message: str
    initial_ncc: float
    final_ncc: float
    elapsed_seconds: float
    accepted: bool
    candidate_id: str | None = None

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "converged": self.converged,
            "message": self.message,
            "iterations": self.iterations,
            "candidate_id": self.candidate_id,
            "parameters_rotvec_xyz_translation_xyz": self.transform.parameters().tolist(),
            "rotation_degrees_xyz": Rotation.from_matrix(self.transform.rotation).as_euler("xyz", degrees=True).tolist(),
            "coverage": self.coverage,
            "error": self.error,
            "roi_coverage": self.roi_coverage,
            "roi_error": self.roi_error,
            "initial_ncc": self.initial_ncc,
            "final_ncc": self.final_ncc,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @property
    def candidate_parameters(self) -> np.ndarray:
        return self.transform.parameters()


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
    min_roi_ncc: float = 0.75,
    roi_weight: float = 2.0,
    candidate_id: str | None = None,
    lock_initial: bool = False,
) -> RegistrationResult:
    start = perf_counter()
    initial_params = initial_transform.parameters() if initial_transform else np.zeros(6)
    current = initial_params.copy()
    iterations = 0
    first_fixed = _downsample(fixed, levels[0])
    first_moving = _downsample(moving, levels[0])
    initial_array, initial_mask = resample(
        first_moving, first_fixed, RigidTransform.from_parameters(initial_params), order=1
    )
    initial_ncc = normalized_cross_correlation(first_fixed.data, initial_array, initial_mask)

    roi_full_mask = roi.mask(fixed) if roi is not None else None
    roi_voxels = int(np.count_nonzero(roi_full_mask)) if roi is not None else 0
    if roi is not None and roi_voxels == 0:
        raise ValueError("region of interest contains no fixed-image voxels")
    roi_down_masks: dict[int, np.ndarray] = {}
    if roi is not None:
        for level in dict.fromkeys(levels):
            roi_down_masks[level] = roi.mask(_downsample(fixed, level))

    result = None
    if lock_initial:
        current = initial_params.copy()
    else:
        for level in levels:
            f = _downsample(fixed, level)
            m = _downsample(moving, level)
            roi_down_mask = roi_down_masks.get(level)
            roi_down_voxels = int(np.count_nonzero(roi_down_mask)) if roi_down_mask is not None else 0

            def objective(params: np.ndarray) -> float:
                transformed, mask = resample(m, f, RigidTransform.from_parameters(params), order=1)
                fraction = np.count_nonzero(mask) / mask.size
                if fraction < min_coverage:
                    return 2.0 - fraction
                ncc = normalized_cross_correlation(f.data, transformed, mask)
                cost = 1.05 - ncc
                if roi_down_mask is not None and roi_down_voxels > 0:
                    covered = roi_down_mask & mask
                    roi_fraction = np.count_nonzero(covered) / roi_down_voxels
                    if roi_fraction < min_roi_coverage:
                        return max(cost, 1.25) + (1.0 - roi_fraction)
                    roi_ncc = normalized_cross_correlation(f.data, transformed, covered)
                    cost = (cost + roi_weight * (1.05 - roi_ncc)) / (1.0 + roi_weight)
                return cost

            result = minimize(
                objective,
                current,
                method="Powell",
                options={"maxiter": max_iterations, "xtol": 2e-4, "ftol": 2e-5, "disp": False},
            )
            iterations += int(getattr(result, "nit", 0))
            current = np.asarray(result.x, dtype=np.float64)

    return evaluate_transform(
        fixed,
        moving,
        RigidTransform.from_parameters(current),
        roi=roi,
        min_coverage=min_coverage,
        min_roi_coverage=min_roi_coverage,
        min_roi_ncc=min_roi_ncc,
        optimizer_ok=True if lock_initial else bool(getattr(result, "success", False)),
        optimizer_message="initial transform locked" if lock_initial else str(getattr(result, "message", "")),
        iterations=iterations,
        initial_ncc=initial_ncc,
        elapsed_seconds=perf_counter() - start,
        candidate_id=candidate_id,
    )


def evaluate_transform(
    fixed: MedicalImage,
    moving: MedicalImage,
    transform: RigidTransform,
    *,
    roi: RegionOfInterest | None = None,
    min_coverage: float = 0.35,
    min_roi_coverage: float = 0.6,
    min_roi_ncc: float = 0.75,
    optimizer_ok: bool = True,
    optimizer_message: str = "candidate evaluated directly",
    iterations: int = 0,
    initial_ncc: float | None = None,
    elapsed_seconds: float = 0.0,
    candidate_id: str | None = None,
) -> RegistrationResult:
    """Score a fixed candidate transform without running any optimization.

    Used to verify physical alignment against known ground-truth transforms and
    to reject candidates that agree on large background regions while missing
    the user-selected target region.
    """
    transformed_array, mask = resample(moving, fixed, transform, order=1)
    info = coverage(mask, fixed)
    fraction = float(info["coverage_fraction"])
    errors = image_error(fixed.data, transformed_array, mask)
    ncc = errors["ncc"]

    roi_coverage_info: dict[str, object] | None = None
    roi_errors: dict[str, float] | None = None
    roi_fraction = 1.0
    roi_ncc = ncc
    if roi is not None:
        roi_full_mask = roi.mask(fixed)
        roi_voxels = int(np.count_nonzero(roi_full_mask))
        if roi_voxels == 0:
            raise ValueError("region of interest contains no fixed-image voxels")
        covered_roi = roi_full_mask & mask
        roi_fraction = float(np.count_nonzero(covered_roi) / roi_voxels)
        roi_coverage_info = coverage(covered_roi, fixed)
        roi_coverage_info["roi_voxels"] = roi_voxels
        roi_coverage_info["roi_coverage_fraction"] = roi_fraction
        roi_coverage_info["roi"] = roi.to_config()
        roi_errors = masked_error(fixed.data, transformed_array, covered_roi)
        roi_ncc = roi_errors["ncc"]

    finite = bool(np.all(np.isfinite(transform.parameters())))
    global_ok = fraction >= min_coverage and ncc >= 0.75
    roi_ok = roi is None or (roi_fraction >= min_roi_coverage and roi_ncc >= min_roi_ncc)
    accepted = bool(optimizer_ok and finite and global_ok and roi_ok)
    status = "success" if accepted else "failed"
    if accepted:
        message = optimizer_message
    else:
        message = (
            "registration did not meet acceptance: "
            f"optimizer={optimizer_ok}, ncc={ncc:.4f}, coverage={fraction:.4f}"
        )
        if roi is not None:
            message += f", roi_ncc={roi_ncc:.4f}, roi_coverage={roi_fraction:.4f}"
    transformed = (
        MedicalImage(transformed_array, fixed.affine, {"resampled_to": "fixed", "candidate_id": candidate_id})
        if accepted
        else None
    )
    return RegistrationResult(
        status,
        transform,
        transformed,
        mask if accepted else None,
        info if accepted else None,
        errors if accepted else None,
        roi_coverage_info if accepted else None,
        roi_errors if accepted else None,
        iterations,
        accepted,
        message,
        ncc if initial_ncc is None else initial_ncc,
        ncc,
        elapsed_seconds,
        accepted,
        candidate_id,
    )
