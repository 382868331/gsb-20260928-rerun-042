from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image import MedicalImage


@dataclass(frozen=True)
class RegionOfInterest:
    """User-selected region of interest defined on the fixed image.

    The selection lives in physical (world) coordinates as an axis-aligned
    box ``[world_min, world_max]``. Background agreement outside this box
    must never hide misalignment of the anatomy the user cares about, so
    every candidate is scored both on the full image and inside the ROI.
    """

    world_min: tuple[float, float, float]
    world_max: tuple[float, float, float]
    name: str = "roi"

    def __post_init__(self) -> None:
        low = np.asarray(self.world_min, dtype=np.float64)
        high = np.asarray(self.world_max, dtype=np.float64)
        if low.shape != (3,) or high.shape != (3,):
            raise ValueError("ROI bounds must be length-3 world coordinates")
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError("ROI bounds must be finite")
        if np.any(high <= low):
            raise ValueError("world_max must be strictly greater than world_min on every axis")
        object.__setattr__(self, "world_min", tuple(low.tolist()))
        object.__setattr__(self, "world_max", tuple(high.tolist()))

    @classmethod
    def from_index_box(
        cls,
        image: MedicalImage,
        index_min: tuple[int, int, int],
        index_max: tuple[int, int, int],
        name: str = "roi",
    ) -> "RegionOfInterest":
        low = image.index_to_world(np.asarray(index_min, dtype=float))
        high = image.index_to_world(np.asarray(index_max, dtype=float))
        box_low = np.minimum(low, high)
        box_high = np.maximum(low, high) + np.abs(image.spacing)
        return cls(tuple(box_low.tolist()), tuple(box_high.tolist()), name)

    def mask_on(self, image: MedicalImage) -> np.ndarray:
        """Voxel mask of the fixed-image grid covered by this ROI box."""
        grid = np.meshgrid(
            np.arange(image.shape[0]),
            np.arange(image.shape[1]),
            np.arange(image.shape[2]),
            indexing="ij",
        )
        indices = np.stack([axis.ravel() for axis in grid])
        world = image.index_to_world(indices)
        low = np.asarray(self.world_min)[:, None]
        high = np.asarray(self.world_max)[:, None]
        inside = np.all((world >= low) & (world < high), axis=0)
        return inside.reshape(image.shape)

    def summary(self, image: MedicalImage) -> dict[str, object]:
        mask = self.mask_on(image)
        return {
            "name": self.name,
            "world_min": list(self.world_min),
            "world_max": list(self.world_max),
            "roi_voxels": int(mask.sum()),
        }
