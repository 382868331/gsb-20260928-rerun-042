"""配准相似性与对齐误差，只在有效覆盖区域内统计。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AlignmentErrors:
    n_valid: int
    mae: float
    rmse: float
    ncc: float
    intensity_delta: float

    def to_dict(self) -> dict:
        return {
            "n_valid": self.n_valid,
            "mae": self.mae,
            "rmse": self.rmse,
            "ncc": self.ncc,
            "intensity_delta": self.intensity_delta,
        }


def _intensities(moving_v: MedicalVolume, fixed: MedicalVolume, mask: np.ndarray):
    a = np.asarray(moving_v.data, dtype=np.float64)[mask]
    b = np.asarray(fixed.data, dtype=np.float64)[mask]
    if a.size < 8:
        raise ValueError(f"有效重叠体素过少（{a.size}），无法可靠评估误差")
    # 各影像独立的稳健归一化，消除整体强度尺度差异后再比较
    lo, hi = np.percentile(a, [1, 99])
    a = (a - lo) / (hi - lo + 1e-12)
    lo, hi = np.percentile(b, [1, 99])
    b = (b - lo) / (hi - lo + 1e-12)
    return a, b


def alignment_errors(
    aligned: MedicalVolume, fixed: MedicalVolume, valid_mask: np.ndarray
) -> AlignmentErrors:
    a, b = _intensities(aligned, fixed, valid_mask)
    diff = a - b
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)))
    ncc = float(np.sum(ac * bc) / denom) if denom > 1e-12 else float("nan")
    return AlignmentErrors(
        n_valid=int(a.size),
        mae=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff * diff))),
        ncc=ncc,
        intensity_delta=float(a.mean() - b.mean()),
    )


def point_target_error(
    fixed_points_mm: np.ndarray,
    moving_points_mm: np.ndarray,
    transform,
) -> dict:
    """已知对应点（物理坐标，mm）经变换后的目标配准误差。"""
    fp = np.asarray(fixed_points_mm, dtype=np.float64)
    mp = np.asarray(moving_points_mm, dtype=np.float64)
    if fp.shape != mp.shape or fp.ndim != 2 or fp.shape[1] != 3:
        raise ValueError("两组点必须是形状相同的 (N, 3) 数组")
    mapped = transform.apply(fp)
    dist = np.linalg.norm(mapped - mp, axis=1)
    return {
        "n_points": int(fp.shape[0]),
        "tre_mean_mm": float(dist.mean()),
        "tre_max_mm": float(dist.max()),
        "tre_rms_mm": float(np.sqrt(np.mean(dist ** 2))),
    }
