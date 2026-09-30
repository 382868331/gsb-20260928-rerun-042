from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .image import MedicalImage, make_affine
from .transform import RigidTransform


def _bump_volume(shape: tuple[int, int, int], spacing: np.ndarray, affine: np.ndarray) -> np.ndarray:
    grid = np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
    points = np.stack([axis.ravel() for axis in grid], axis=1)
    physical = (points @ affine[:3, :3].T) + affine[:3, 3]
    centers = np.asarray([
        [-10.0, -7.0, -5.0],
        [8.0, 5.0, 6.0],
        [0.0, 10.0, -9.0],
    ])
    scales = np.asarray([5.0, 4.0, 6.0])
    data = np.zeros(shape, dtype=np.float32)
    for center, scale in zip(centers, scales):
        distance = np.linalg.norm((physical - center) / scale, axis=1)
        data += np.exp(-(distance**2)).reshape(shape).astype(np.float32)
    ii, jj, kk = np.indices(shape)
    data += 0.08 * np.sin(ii * spacing[0] / 3.0) * np.cos(jj * spacing[1] / 4.0)
    return data.reshape(shape).astype(np.float32)


def synthetic_pair(
    shape: tuple[int, int, int] = (48, 56, 44),
    fixed_spacing: tuple[float, float, float] = (1.2, 1.0, 1.4),
    moving_spacing: tuple[float, float, float] = (1.5, 1.2, 1.0),
    moving_origin: tuple[float, float, float] = (2.0, -3.0, 1.5),
    rotation_angles_deg: tuple[float, float, float] = (7.0, -9.0, 6.0),
    translation: tuple[float, float, float] = (5.0, -4.0, 2.5),
    moving_direction: np.ndarray | None = None,
    partial_overlap: bool = False,
) -> tuple[MedicalImage, MedicalImage, RigidTransform]:
    fixed_affine = make_affine(shape, fixed_spacing)
    fixed = MedicalImage(_bump_volume(shape, np.asarray(fixed_spacing), fixed_affine), fixed_affine)

    if partial_overlap:
        # A smaller moving FOV deliberately covers only part of the fixed
        # image, so coverage inside a selected region must be reported honestly.
        moving_shape = tuple(max(10, size // 2 + 4) for size in shape)
    else:
        moving_shape = tuple(
            int(round(size * fs / ms)) + 8 for size, fs, ms in zip(shape, fixed_spacing, moving_spacing)
        )
    moving_affine = make_affine(
        moving_shape,
        moving_spacing,
        moving_origin,
        direction=moving_direction,
    )
    truth = RigidTransform(
        Rotation.from_euler("xyz", rotation_angles_deg, degrees=True).as_matrix(),
        np.asarray(translation, dtype=float),
    )
    moving_data = _bump_volume(moving_shape, np.asarray(moving_spacing), moving_affine)
    # Moving is a separately acquired image whose anatomy equals truth(fixed).
    # Build it by sampling fixed through truth while retaining its own geometry.
    from .resample import resample

    warped_fixed, mask = resample(
        fixed,
        MedicalImage(np.zeros(moving_shape, dtype=np.float32), moving_affine),
        truth,
        order=3,
    )
    moving = MedicalImage(np.where(mask, warped_fixed, moving_data * 0).astype(np.float32), moving_affine)
    return fixed, moving, truth.inverse()


def uniform_volume(
    shape: tuple[int, int, int] = (18, 18, 18),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    value: float = 0.0,
    direction: np.ndarray | None = None,
) -> MedicalImage:
    """A volume with no image information (flat region); registration must fail."""
    affine = make_affine(shape, spacing, origin, direction=direction)
    data = np.full(shape, float(value), dtype=np.float32)
    return MedicalImage(data, affine)


def reoriented_direction(axis_permutation: tuple[int, int, int], flips: tuple[bool, bool, bool]) -> np.ndarray:
    """Build a right-handed direction cosine with permuted/flipped axes."""
    matrix = np.zeros((3, 3))
    signs = np.asarray([-1.0 if flip else 1.0 for flip in flips])
    for output_axis, input_axis in enumerate(axis_permutation):
        matrix[output_axis, input_axis] = signs[output_axis]
    if np.linalg.det(matrix) < 0:
        # Preserve a right-handed scanner convention after an odd flip count.
        matrix[:, axis_permutation[0]] *= -1.0
    return matrix


def synthetic_series(
    shape: tuple[int, int, int] = (36, 40, 34),
    time_ids: tuple[str, ...] = ("t0", "t1", "t2", "t3"),
    reference_time_id: str = "t0",
    drift_translation_per_step: tuple[float, float, float] = (1.5, -1.0, 0.8),
    drift_rotation_deg_per_step: tuple[float, float, float] = (1.5, -1.0, 1.0),
    jump_time_id: str | None = None,
    jump_translation: tuple[float, float, float] = (9.0, 7.0, -6.0),
    blank_time_id: str | None = None,
) -> tuple[dict[str, "MedicalImage"], dict[str, "RigidTransform"]]:
    """A time series of acquisitions with a slowly drifting pose.

    jump_time_id adds one abruptly displaced acquisition (to exercise
    large-change warnings) and blank_time_id replaces one acquisition
    with an uninformative flat volume (to exercise per-time failure).
    """

    from .resample import resample

    spacing = (1.2, 1.0, 1.4)
    affine = make_affine(shape, spacing)
    reference = MedicalImage(_bump_volume(shape, np.asarray(spacing), affine), affine)
    moving_shape = tuple(size + 6 for size in shape)
    moving_spacing = (1.4, 1.2, 1.1)
    moving_origin = (1.5, -2.0, 1.0)
    moving_affine = make_affine(moving_shape, moving_spacing, moving_origin)
    images: dict[str, MedicalImage] = {}
    truths: dict[str, RigidTransform] = {}
    step = 0
    for time_id in time_ids:
        if time_id == reference_time_id:
            images[time_id] = reference
            truths[time_id] = RigidTransform()
            continue
        step += 1
        rotation = Rotation.from_euler(
            "xyz", np.asarray(drift_rotation_deg_per_step) * step, degrees=True
        ).as_matrix()
        translation = np.asarray(drift_translation_per_step, dtype=float) * step
        if time_id == jump_time_id:
            translation = translation + np.asarray(jump_translation, dtype=float)
        truth = RigidTransform(rotation, translation)
        warped, mask = resample(
            reference,
            MedicalImage(np.zeros(moving_shape, dtype=np.float32), moving_affine),
            truth,
            order=3,
        )
        if time_id == blank_time_id:
            images[time_id] = uniform_volume(moving_shape, moving_spacing, moving_origin)
        else:
            images[time_id] = MedicalImage(np.where(mask, warped, 0.0).astype(np.float32), moving_affine)
        truths[time_id] = truth.inverse()
    return images, truths
