from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class RigidTransform:
    """World-coordinate rigid transform: rotated point plus translation."""

    def __init__(self, rotation: np.ndarray | None = None, translation: np.ndarray | None = None):
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        self.translation = np.zeros(3) if translation is None else np.asarray(translation, dtype=np.float64)
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("expected a 3x3 rotation and length-3 translation")
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-7):
            raise ValueError("rotation must be orthogonal")
        if not np.isfinite(self.translation).all():
            raise ValueError("translation must be finite")

    @classmethod
    def from_parameters(cls, params: np.ndarray) -> "RigidTransform":
        params = np.asarray(params, dtype=np.float64)
        if params.size != 6:
            raise ValueError("rigid parameters are [rx, ry, rz, tx, ty, tz]")
        return cls(Rotation.from_rotvec(params[:3]).as_matrix(), params[3:])

    def parameters(self) -> np.ndarray:
        return np.concatenate([Rotation.from_matrix(self.rotation).as_rotvec(), self.translation])

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (self.rotation @ np.asarray(points, dtype=np.float64).T).T + self.translation

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self.rotation.T, -(self.rotation.T @ self.translation))

    def compose(self, other: "RigidTransform") -> "RigidTransform":
        return RigidTransform(self.rotation @ other.rotation, self.rotation @ other.translation + self.translation)

    def to_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix
