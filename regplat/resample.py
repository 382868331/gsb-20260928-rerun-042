"""按刚性变换在物理空间重采样，并计算覆盖范围。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates

from .volume import MedicalVolume
from .transforms import RigidTransform


@dataclass(frozen=True)
class Coverage:
    valid_voxels: int
    total_voxels: int
    fraction: float
    physical_overlap_mm3: float
    target_volume_mm3: float
    missing_fraction: float

    def to_dict(self) -> dict:
        return {
            "valid_voxels": self.valid_voxels,
            "total_voxels": self.total_voxels,
            "fraction": self.fraction,
            "physical_overlap_mm3": self.physical_overlap_mm3,
            "target_volume_mm3": self.target_volume_mm3,
            "missing_fraction": self.missing_fraction,
        }


@dataclass(frozen=True)
class ResampleResult:
    """重采样产物：aligned 与 fixed 同格；valid_mask 标记真正落在源体内的体素。"""

    aligned: MedicalVolume
    valid_mask: np.ndarray
    coverage: Coverage


def resample(
    moving: MedicalVolume,
    fixed: MedicalVolume,
    transform: RigidTransform,
    order: int = 1,
) -> ResampleResult:
    """把 moving 按 transform（fixed 坐标 -> moving 坐标）重采样到 fixed 网格。

    不做任何尺寸拉平：坐标变换贯穿 fixed/ moving 各自的原点、间距与方向。
    落在 moving 域外的位置填充 NaN，并在 valid_mask 中标记为 False。
    """
    fixed_world = fixed.voxel_grid_centers()                      # (K,J,I,3)
    moving_world = transform.apply(fixed_world)
    moving_vox = moving.world_to_voxel(moving_world)

    shape = moving.shape
    inside = np.all(
        (moving_vox >= -0.5)
        & (moving_vox <= (np.array(shape, dtype=np.float64) - 0.5)),
        axis=-1,
    )
    coords = np.moveaxis(moving_vox, -1, 0)
    sampled = map_coordinates(
        moving.data,
        coords,
        order=order,
        mode="constant",
        cval=np.nan,
    )
    sampled = np.where(inside, sampled, np.nan)

    valid = ~np.isnan(sampled)
    voxel_vol = float(np.prod(fixed.spacing))
    valid_count = int(valid.sum())
    overlap = valid_count * voxel_vol
    target_vol = int(np.prod(fixed.shape)) * voxel_vol
    coverage = Coverage(
        valid_voxels=valid_count,
        total_voxels=int(np.prod(fixed.shape)),
        fraction=valid_count / max(1, valid.size),
        physical_overlap_mm3=overlap,
        target_volume_mm3=target_vol,
        missing_fraction=1.0 - valid_count / max(1, valid.size),
    )
    return ResampleResult(
        aligned=fixed.with_data(sampled, allow_all_nan=True),
        valid_mask=valid,
        coverage=coverage,
    )


def box_overlap(a: MedicalVolume, b: MedicalVolume) -> float:
    """两影像物理包围盒的交集体积（mm^3），仅用于诊断。"""
    lo_a, hi_a = a.world_bounds()
    lo_b, hi_b = b.world_bounds()
    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    side = np.maximum(0.0, hi - lo)
    return float(np.prod(side))
