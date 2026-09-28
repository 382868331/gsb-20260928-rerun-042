from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .image import MedicalImage
from .metrics import image_error, normalized_cross_correlation
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
            "initial_ncc": self.initial_ncc,
            "final_ncc": self.final_ncc,
            "elapsed_seconds": self.elapsed_seconds,
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

    for level in levels:
        f = _downsample(fixed, level)
        m = _downsample(moving, level)

        def objective(params: np.ndarray) -> float:
            transformed, mask = resample(m, f, RigidTransform.from_parameters(params), order=1)
            fraction = np.count_nonzero(mask) / mask.size
            if fraction < min_coverage:
                return 2.0 - fraction
            ncc = normalized_cross_correlation(f.data, transformed, mask)
            return 1.05 - ncc

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
    converged = bool(result.success and fraction >= min_coverage and ncc >= 0.75 and np.isfinite(current).all())
    status = "success" if converged else "failed"
    message = str(result.message) if converged else (
        f"registration did not meet acceptance: optimizer={result.success}, ncc={ncc:.4f}, coverage={fraction:.4f}"
    )
    transformed = MedicalImage(transformed_array, fixed.affine, {"resampled_to": "fixed"}) if converged else None
    return RegistrationResult(
        status,
        transform if converged else None,
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
    )
