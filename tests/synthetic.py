"""合成体数据：在物理空间定义结构，可施加已知位移与旋转。"""

from __future__ import annotations
import numpy as np

from regplat.volume import MedicalVolume
from regplat.transforms import RigidTransform
from scipy.spatial.transform import Rotation


def phantom_field(points_world: np.ndarray) -> np.ndarray:
    """若干高斯球 + 平滑梯度，保证旋转/平移可辨识。"""
    p = points_world
    blobs = [
        ((-12, -8, -6), 7.0, 1.0),
        ((10, 6, 8), 5.0, 0.9),
        ((-6, 12, 2), 4.0, 0.8),
        ((14, -10, -10), 3.5, 0.7),
        ((0, 0, 12), 4.5, 0.75),
    ]
    val = 0.15 * (p[..., 0] / 30.0 + p[..., 1] / 40.0)
    for center, sigma, amp in blobs:
        d2 = np.sum(((p - np.array(center)) / sigma) ** 2, axis=-1)
        val += amp * np.exp(-0.5 * d2)
    return val


def make_volume(field, shape, spacing, origin=(0.0, 0.0, 0.0), direction=None,
                world_transform: RigidTransform | None = None) -> MedicalVolume:
    vol = MedicalVolume(np.zeros(shape, dtype=np.float64), spacing, origin, direction)
    centers = vol.voxel_grid_centers()
    sample_at = centers if world_transform is None else world_transform.inverse().apply(centers)
    return vol.with_data(field(sample_at))


def known_pair(translation=(5.0, -3.0, 4.0), angle_deg=12.0):
    """返回 (moving, fixed, ground_truth_transform)。

    fixed 与 moving 使用不同原点、间距和方向余弦；
    ground truth 为 fixed 物理坐标 -> moving 物理坐标。
    """
    # fixed 网格中心位于物理原点，覆盖 phantom 结构区
    shape_f = (40, 44, 36)
    spacing_f = (2.0, 2.0, 2.0)
    aff_f = np.diag(spacing_f)
    origin_f = -aff_f @ ((np.array(shape_f, float) - 1) / 2)
    fixed = make_volume(
        phantom_field, shape_f, spacing_f, origin=tuple(origin_f)
    )
    gt = RigidTransform(
        Rotation.from_euler("z", angle_deg, degrees=True),
        np.asarray(translation, dtype=float),
    )
    # moving 的体素轴相对物理空间旋转 90 度（轴交换），间距各向异性；
    # 其物理网格必须覆盖 gt(fixed 区域)，否则配准没有可用重叠
    dir_m = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    spacing_m = np.array([1.6, 2.1, 2.4])
    aff_m = dir_m @ np.diag(spacing_m)
    f_lo, f_hi = fixed.world_bounds()
    f_corners = np.array([[x, y, z] for x in (f_lo[0], f_hi[0])
                          for y in (f_lo[1], f_hi[1])
                          for z in (f_lo[2], f_hi[2])])
    moved = gt.apply(f_corners)
    m_lo, m_hi = moved.min(0) - 4.0, moved.max(0) + 4.0
    n = np.ceil((m_hi - m_lo) / spacing_m).astype(int) + 1
    # 以 gt(fixed 中心) 为 moving 网格中心反算原点
    f_center = np.zeros(3)
    m_center = gt.apply(f_center)
    origin_m = m_center - aff_m @ ((n - 1) / 2.0)
    moving = make_volume(
        phantom_field, tuple(n), tuple(spacing_m),
        origin=tuple(origin_m), direction=dir_m,
        world_transform=gt,   # moving(x) = phantom(gt^{-1}(x))
    )
    return moving, fixed, gt


def disjoint_pair():
    """两个结构互不重叠的影像，用于覆盖不足场景。"""
    # 两块网格都含同样丰富的结构，但物理上相距 300mm、完全无重叠
    fixed = make_volume(phantom_field, (36, 36, 36), (2.0, 2.0, 2.0))
    moving = make_volume(
        phantom_field, (36, 36, 36), (2.0, 2.0, 2.0), origin=(300.0, 300.0, 300.0)
    )
    return moving, fixed


def unrelated_pair():
    """结构完全不同但重叠充分：棋盘格 vs 高斯球，相似性无法收敛。"""
    fixed = make_volume(phantom_field, (36, 36, 36), (2.0, 2.0, 2.0))

    def checker(p):
        return (np.floor(p[..., 0] / 6).astype(int)
                + np.floor(p[..., 1] / 6).astype(int)
                + np.floor(p[..., 2] / 6).astype(int)) % 2

    moving = make_volume(checker, (36, 36, 36), (2.0, 2.0, 2.0))
    return moving, fixed
