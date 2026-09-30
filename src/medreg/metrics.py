from __future__ import annotations

import numpy as np


def normalized_cross_correlation(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float:
    fixed = np.asarray(fixed, dtype=np.float64)[mask]
    moving = np.asarray(moving, dtype=np.float64)[mask]
    if fixed.size < 8:
        return -1.0
    fixed = fixed - fixed.mean()
    moving = moving - moving.mean()
    denominator = np.sqrt(np.sum(fixed * fixed) * np.sum(moving * moving))
    if denominator <= np.finfo(float).eps:
        return -1.0
    return float(np.sum(fixed * moving) / denominator)


def mean_squared_error(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(fixed, dtype=np.float64)[mask] - np.asarray(moving, dtype=np.float64)[mask]
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff * diff))


def image_error(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = np.asarray(fixed, dtype=np.float64)[mask] - np.asarray(moving, dtype=np.float64)[mask]
    if diff.size == 0:
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf"), "ncc": -1.0}
    mse = float(np.mean(diff * diff))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
        "ncc": normalized_cross_correlation(fixed, moving, mask),
    }


def region_error(
    fixed: np.ndarray,
    moving: np.ndarray,
    overlap_mask: np.ndarray,
    region_mask: np.ndarray,
) -> dict[str, object]:
    """Error and coverage restricted to a user-selected region.

    ``region_mask`` selects the voxels the user cares about; only voxels that
    are also covered by the resampled moving image enter the error. The
    separate coverage number exposes target misalignment that a large,
    agreeing background would otherwise hide.
    """

    region_mask = np.asarray(region_mask, dtype=bool)
    overlap_mask = np.asarray(overlap_mask, dtype=bool)
    region_total = int(region_mask.sum())
    effective = overlap_mask & region_mask
    covered = int(effective.sum())
    stats = image_error(fixed, moving, effective)
    stats.update(
        {
            "region_voxels": region_total,
            "covered_region_voxels": covered,
            "region_coverage_fraction": float(covered / region_total) if region_total else 0.0,
        }
    )
    return stats
