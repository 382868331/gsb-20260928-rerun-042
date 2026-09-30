from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .image import MedicalImage


@dataclass(frozen=True)
class RegionOfInterest:
    """An axis-aligned region of interest defined in physical (world) coordinates.

    ROI bounds are deliberately expressed in millimetres rather than voxel
    indices, so the same user selection remains meaningful across images with
    different spacing, origin and orientation.
    """

    world_min: tuple[float, float, float]
    world_max: tuple[float, float, float]
    name: str = "roi"

    def __post_init__(self) -> None:
        low = np.asarray(self.world_min, dtype=np.float64)
        high = np.asarray(self.world_max, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("ROI bounds must have length 3")
        if np.any(high <= low):
            raise ValueError("ROI world_max must be strictly greater than world_min")
        object.__setattr__(self, "world_min", tuple(low.tolist()))
        object.__setattr__(self, "world_max", tuple(high.tolist()))

    @classmethod
    def from_center_radius(
        cls,
        center: tuple[float, float, float],
        radius: tuple[float, float, float] | float,
        name: str = "roi",
    ) -> "RegionOfInterest":
        center = np.asarray(center, dtype=np.float64)
        radius = np.asarray(radius, dtype=np.float64)
        if radius.ndim == 0:
            radius = np.repeat(radius, 3)
        return cls(tuple((center - radius).tolist()), tuple((center + radius).tolist()), name)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RegionOfInterest":
        if "world_min" in config and "world_max" in config:
            return cls(
                tuple(float(v) for v in config["world_min"]),
                tuple(float(v) for v in config["world_max"]),
                str(config.get("name", "roi")),
            )
        if "center" in config and "radius" in config:
            radius = config["radius"]
            if isinstance(radius, (list, tuple)):
                radius = tuple(float(v) for v in radius)
            else:
                radius = float(radius)
            return cls.from_center_radius(
                tuple(float(v) for v in config["center"]), radius, str(config.get("name", "roi"))
            )
        raise ValueError("ROI config requires world_min/world_max or center/radius")

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "world_min": [float(v) for v in self.world_min],
            "world_max": [float(v) for v in self.world_max],
        }

    def mask(self, image: MedicalImage) -> np.ndarray:
        """Boolean voxel mask marking the ROI on the given image grid."""
        grid = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            np.arange(image.shape[2]),
            indexing="ij",
        )
        indices = np.stack([axis.ravel() for axis in grid])
        world = image.index_to_world(indices).T
        low = np.asarray(self.world_min)
        high = np.asarray(self.world_max)
        inside = np.all((world >= low) & (world < high), axis=1)
        return inside.reshape(image.shape)

    def voxel_count(self, image: MedicalImage) -> int:
        return int(np.count_nonzero(self.mask(image)))
