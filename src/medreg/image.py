from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class MedicalImage:
    """A 3D scalar volume with an index-to-world affine.

    The affine maps an index ``(i,j,k,1)`` to physical coordinates, so spacing
    and orientation are never inferred from array shape alone.
    """

    data: np.ndarray
    affine: np.ndarray
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float32)
        affine = np.asarray(self.affine, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("MedicalImage currently supports 3D volumes")
        if affine.shape != (4, 4):
            raise ValueError("affine must be a 4x4 matrix")
        if not np.all(np.isfinite(data)):
            raise ValueError("image data contains non-finite values")
        if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("affine must be finite and invertible")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)

    @property
    def spacing(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def direction(self) -> np.ndarray:
        return self.affine[:3, :3] / self.spacing

    @property
    def origin(self) -> np.ndarray:
        return self.affine[:3, 3].copy()

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.float64)
        return (self.affine[:3, :3] @ indices) + self.affine[:3, 3, None]

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return np.linalg.inv(self.affine[:3, :3]) @ (points - self.affine[:3, 3, None])

    def physical_corners(self) -> np.ndarray:
        ii, jj, kk = np.meshgrid(
            (0, self.shape[0] - 1),
            (0, self.shape[1] - 1),
            (0, self.shape[2] - 1),
            indexing="ij",
        )
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()])
        return self.index_to_world(idx).T

    def save_nifti(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(self.data, self.affine), path)

    @classmethod
    def load_nifti(cls, path: str | Path) -> "MedicalImage":
        image = nib.load(str(path))
        data = np.asarray(image.get_fdata(), dtype=np.float32)
        return cls(data, np.asarray(image.affine, dtype=np.float64), {"path": str(path)})


def make_affine(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Create an affine centered at the volume center."""
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if direction is None:
        direction = np.eye(3)
    direction = np.asarray(direction, dtype=np.float64)
    if np.linalg.det(direction) <= 0:
        raise ValueError("direction must be a right-handed rotation matrix")
    linear = direction @ np.diag(spacing)
    center_offset = linear @ ((np.asarray(shape) - 1) / 2.0)
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = origin - center_offset
    return affine
