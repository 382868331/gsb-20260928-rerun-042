from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .image import MedicalImage
from .transform import RigidTransform


def resample(
    moving: MedicalImage,
    reference: MedicalImage,
    transform: RigidTransform | None = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample moving onto reference after applying transform to moving points."""
    transform = transform or RigidTransform()
    grid = np.meshgrid(
        np.arange(reference.shape[0]),
        np.arange(reference.shape[1]),
        np.arange(reference.shape[2]),
        indexing="ij",
    )
    ref_indices = np.stack([axis.ravel() for axis in grid])
    ref_world = reference.index_to_world(ref_indices)
    moving_world = transform.inverse().apply(ref_world.T).T
    moving_indices = moving.world_to_index(moving_world)
    inside = np.all((moving_indices >= 0) & (moving_indices < np.asarray(moving.shape)[:, None] - 1), axis=0)
    values = np.zeros(ref_indices.shape[1], dtype=np.float32)
    values[inside] = map_coordinates(
        moving.data,
        moving_indices[:, inside],
        order=order,
        mode="constant",
        cval=0.0,
    )
    return values.reshape(reference.shape), inside.reshape(reference.shape)


def coverage(mask: np.ndarray, reference: MedicalImage | None = None) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    result: dict[str, object] = {
        "covered_voxels": count,
        "total_reference_voxels": int(mask.size),
        "coverage_fraction": float(count / mask.size) if mask.size else 0.0,
    }
    if count:
        coords = np.where(mask)
        result["bounding_box_index"] = [
            [int(np.min(coords[axis])), int(np.max(coords[axis]))]
            for axis in range(3)
        ]
        if reference is not None:
            first = np.asarray([bounds[0] for bounds in result["bounding_box_index"]])
            last = np.asarray([bounds[1] for bounds in result["bounding_box_index"]])
            result["bounding_box_world_min"] = reference.index_to_world(first).tolist()
            result["bounding_box_world_max"] = reference.index_to_world(last).tolist()
    else:
        result["bounding_box_index"] = None
    return result
