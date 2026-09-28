"""刚性变换：物理空间中的旋转（旋转向量参数化）加平移。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigidTransform:
    """x_moving = R @ x_fixed + translation，坐标均为物理坐标。"""

    rotation: Rotation
    translation: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.translation, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(t)):
            raise ValueError("translation 必须有限")
        object.__setattr__(self, "translation", t)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(Rotation.identity(), np.zeros(3))

    @classmethod
    def from_params(cls, params: np.ndarray) -> "RigidTransform":
        p = np.asarray(params, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(p)):
            raise ValueError("变换参数必须有限")
        return cls(Rotation.from_rotvec(p[:3]), p[3:])

    def to_params(self) -> np.ndarray:
        return np.concatenate([self.rotation.as_rotvec(), self.translation])

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.rotation.as_matrix()

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return self.rotation.apply(pts) + self.translation

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        """先 other 再 self：(self∘other)(x)。"""
        r = self.rotation * other.rotation
        t = self.translation + self.rotation.apply(other.translation)
        return RigidTransform(r, t)

    def inverse(self) -> "RigidTransform":
        inv_r = self.rotation.inv()
        return RigidTransform(inv_r, -inv_r.apply(self.translation))

    def to_dict(self) -> dict:
        return {
            "type": "rigid",
            "rotation_vector": self.rotation.as_rotvec().tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_mm": self.translation.tolist(),
            "rotation_angle_deg": float(np.degrees(self.rotation.magnitude())),
        }

    def angular_difference(self, other: "RigidTransform") -> float:
        """与另一刚性变换的旋转夹角（度）。"""
        delta = self.rotation.inv() * other.rotation
        return float(np.degrees(delta.magnitude()))

    def translation_difference(self, other: "RigidTransform") -> float:
        return float(np.linalg.norm(self.translation - other.translation))
