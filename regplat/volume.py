"""医学体数据模型：物理坐标由原点、间距和方向余弦共同决定。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MedicalVolume:
    """三维体数据及其物理空间描述。

    data 形状为 (K, J, I)，索引顺序 (z, y, x)，均采用体素中心坐标。
    world = origin + direction @ diag(spacing) @ voxel

    影像一经创建不在原地修改；配准产物是新的 MedicalVolume，原始影像保留。
    """

    data: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: np.ndarray | None = None  # 3x3，列方向为各体素轴在物理空间中的方向

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float64)
        object.__setattr__(self, "data", arr)
        if arr.ndim != 3:
            raise ValueError(f"data 必须是三维数组，实际 ndim={arr.ndim}")
        if np.any(np.isinf(arr)):
            raise ValueError("data 中存在 Inf，无法参与配准（NaN 可表示覆盖范围外缺失值）")
        if np.all(np.isnan(arr)):
            raise ValueError("data 全部为 NaN")
        spacing = tuple(float(v) for v in self.spacing)
        if len(spacing) != 3 or any(v <= 0 for v in spacing):
            raise ValueError(f"spacing 必须是三个正数，实际为 {spacing}")
        object.__setattr__(self, "spacing", spacing)
        origin = tuple(float(v) for v in self.origin)
        if len(origin) != 3 or not all(np.isfinite(origin)):
            raise ValueError("origin 必须是三个有限数")
        object.__setattr__(self, "origin", origin)
        direction = np.eye(3) if self.direction is None else np.asarray(self.direction, dtype=np.float64)
        if direction.shape != (3, 3) or not np.all(np.isfinite(direction)):
            raise ValueError("direction 必须是 3x3 有限矩阵")
        norms = np.linalg.norm(direction, axis=0)
        if np.any(np.abs(norms - 1.0) > 1e-4):
            raise ValueError(f"direction 各列必须为单位向量，列范数 {norms}")
        gram = direction.T @ direction
        if np.max(np.abs(gram - np.diag(np.diag(gram)))) > 1e-4 or np.any(np.abs(np.diag(gram) - 1) > 1e-3):
            raise ValueError("direction 必须是正交方向余弦矩阵")
        det = np.linalg.det(direction)
        if abs(abs(det) - 1.0) > 1e-3:
            raise ValueError(f"direction 行列式绝对值应为 1，实际 det={det}")
        object.__setattr__(self, "direction", direction)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.data.shape)

    @property
    def affine(self) -> np.ndarray:
        """体素中心 -> 物理坐标 的 3x3 线性部分 A。"""
        return self.direction @ np.diag(self.spacing)

    @classmethod
    def from_affine(cls, data, affine_4x4, spacing=None) -> "MedicalVolume":
        aff = np.asarray(affine_4x4, dtype=np.float64)
        if aff.shape != (4, 4):
            raise ValueError("affine 必须是 4x4 矩阵")
        linear = aff[:3, :3]
        if spacing is None:
            spacing = tuple(np.linalg.norm(linear, axis=0))
        spacing = np.asarray(spacing, dtype=np.float64)
        direction = linear / spacing
        return cls(data, tuple(spacing), tuple(aff[:3, 3]), direction)

    def affine_4x4(self) -> np.ndarray:
        aff = np.eye(4)
        aff[:3, :3] = self.affine
        aff[:3, 3] = self.origin
        return aff

    def voxel_to_world(self, voxel: np.ndarray) -> np.ndarray:
        v = np.asarray(voxel, dtype=np.float64)
        return (self.affine @ v[..., None])[..., 0] + np.asarray(self.origin)

    def world_to_voxel(self, world: np.ndarray) -> np.ndarray:
        w = np.asarray(world, dtype=np.float64) - np.asarray(self.origin)
        inv = np.linalg.inv(self.affine)
        return (inv @ w[..., None])[..., 0]

    def world_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """8 个角点决定的物理空间轴对齐包围盒 (low, high)。"""
        k, j, i = (np.array(s) - 1.0 for s in self.shape)
        corners = np.array(
            [[0, 0, 0], [i, 0, 0], [0, j, 0], [0, 0, k],
             [i, j, 0], [i, 0, k], [0, j, k], [i, j, k]]
        )
        worlds = self.voxel_to_world(corners)
        return worlds.min(axis=0), worlds.max(axis=0)

    def voxel_grid_centers(self) -> np.ndarray:
        """返回形状 (K, J, I, 3) 的全部体素中心物理坐标。"""
        grids = np.meshgrid(
            *[np.arange(s, dtype=np.float64) for s in self.shape], indexing="ij"
        )
        voxels = np.stack(grids, axis=-1)
        return self.voxel_to_world(voxels)

    def with_data(self, data: np.ndarray, origin=None, direction=None, spacing=None,
                  allow_all_nan: bool = False) -> "MedicalVolume":
        """以新数组构造影像，默认沿用原几何（用于存放重采样结果）。"""
        if allow_all_nan and np.all(np.isnan(np.asarray(data))):
            vol = object.__new__(MedicalVolume)
            object.__setattr__(vol, "data", np.asarray(data, dtype=np.float64))
            object.__setattr__(vol, "spacing",
                               tuple(float(v) for v in (self.spacing if spacing is None else spacing)))
            object.__setattr__(vol, "origin",
                               tuple(float(v) for v in (self.origin if origin is None else origin)))
            object.__setattr__(vol, "direction",
                               self.direction if direction is None else np.asarray(direction, dtype=np.float64))
            return vol
        return MedicalVolume(
            np.asarray(data),
            self.spacing if spacing is None else spacing,
            self.origin if origin is None else origin,
            self.direction if direction is None else direction,
        )
