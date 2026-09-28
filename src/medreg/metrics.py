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
