import json
import numpy as np
import pytest

from regplat import (
    MedicalVolume, RigidTransform, resample, rigid_register,
    RegistrationParams, RegistrationStatus, BatchRunner,
)
from regplat.metrics import point_target_error
from tests.synthetic import known_pair, disjoint_pair, unrelated_pair, phantom_field, make_volume


FAST = RegistrationParams(levels=(2, 4), samples_per_axis=22, max_iter=80, min_ncc=0.5)


def test_geometry_not_size_matching():
    """不同间距/方向/原点的影像仍能通过物理坐标对齐（恒等场景）。"""
    from scipy.spatial.transform import Rotation
    fixed = make_volume(phantom_field, (36, 36, 36), (2.0, 2.0, 2.0), origin=(3.0, -2.0, 1.0))
    direction = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    spacing_m = (1.4, 1.5, 1.3)
    affine_m = direction @ np.diag(spacing_m)
    center_f = fixed.voxel_to_world((np.array(fixed.shape, float) - 1) / 2)
    n_m = np.array([54, 50, 58], dtype=float)
    origin_m = center_f - affine_m @ ((n_m - 1) / 2)
    moving = make_volume(
        phantom_field, tuple(int(v) for v in n_m), spacing_m,
        origin=tuple(origin_m), direction=direction,
    )
    assert moving.shape != fixed.shape
    r = resample(moving, fixed, RigidTransform.identity())
    err = np.mean(np.abs(r.aligned.data[r.valid_mask] - fixed.data[r.valid_mask]))
    assert r.coverage.fraction > 0.85
    assert err < 0.05


def test_known_transform_recovery():
    moving, fixed, gt = known_pair(translation=(5.0, -3.0, 4.0), angle_deg=12.0)
    result = rigid_register(moving, fixed, FAST)
    assert result.status is RegistrationStatus.SUCCESS, result.message
    # 参数核对：恢复出的刚性变换与真值比较
    # 允许亚体素级误差（最小间距 1.6mm）
    assert result.transform.translation_difference(gt) < 2.5
    assert result.transform.angular_difference(gt) < 3.0
    # 已知对应点的目标配准误差（mm）
    fixed_pts = np.array([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], [0.0, 5.0, -5.0]])
    moving_pts = gt.apply(fixed_pts)
    tre = point_target_error(fixed_pts, moving_pts, result.transform)
    assert tre["tre_mean_mm"] < 2.5
    assert result.coverage.fraction > FAST.min_overlap_fraction
    assert result.errors.ncc > 0.85


def test_originals_preserved_and_outputs_new():
    moving, fixed, _ = known_pair()
    moving_copy = moving.data.copy()
    fixed_copy = fixed.data.copy()
    result = rigid_register(moving, fixed, FAST)
    assert result.ok
    assert np.array_equal(moving.data, moving_copy)
    assert np.array_equal(fixed.data, fixed_copy)
    assert result.aligned is not moving
    assert result.aligned.data is not moving.data


def test_constant_volume_is_insufficient_information():
    moving, fixed, _ = known_pair()
    flat = moving.with_data(np.ones_like(moving.data))
    result = rigid_register(flat, fixed, FAST)
    assert result.status is RegistrationStatus.INSUFFICIENT_INFORMATION
    assert result.transform is None and result.aligned is None
    assert "常量" in result.message


def test_disjoint_volumes_report_insufficient_information():
    moving, fixed = disjoint_pair()
    # 平移上限阻止优化器通过巨大平移“找回”远处影像
    bounded = RegistrationParams(
        levels=(2, 4), samples_per_axis=22, max_iter=80,
        min_ncc=0.5, max_translation_mm=20.0,
        init_from_centers=False,
    )
    result = rigid_register(moving, fixed, bounded)
    assert result.status is RegistrationStatus.INSUFFICIENT_INFORMATION
    assert result.aligned is not None  # 中间产物保留以便诊断，但状态明确不是成功
    assert not result.ok
    assert result.coverage.fraction < FAST.min_overlap_fraction


def test_unrelated_structure_does_not_converge():
    moving, fixed = unrelated_pair()
    result = rigid_register(moving, fixed, FAST)
    assert result.status in (
        RegistrationStatus.DID_NOT_CONVERGE,
        RegistrationStatus.INSUFFICIENT_INFORMATION,
    )
    assert not result.ok
    # 棋盘格与高斯球结构无关：要么最终 NCC 不达标，要么优化漂移导致覆盖不足
    assert "NCC" in result.message or "覆盖" in result.message


def test_invalid_input_status(tmp_path):
    with pytest.raises(ValueError):
        MedicalVolume(np.zeros((10, 10)), (1.0, 1.0))


def test_batch_provenance_and_rerun(tmp_path):
    moving, fixed, _ = known_pair()
    out = tmp_path / "batch"

    runner = BatchRunner(out, FAST)
    runner.run_pair("pair0", moving, fixed)
    manifest_path = runner.finalize("demo")
    m1 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m1["summary"]["success"] == 1
    fp1 = m1["params_fingerprint"]
    assert "首次运行" in m1["provenance"]
    assert (out / "pair0" / "aligned.nii.gz").exists()

    # 相同参数重跑：manifest 必须解释沿用了哪份参数
    runner2 = BatchRunner(out, FAST)
    runner2.run_pair("pair0", moving, fixed)
    m2 = json.loads(runner2.finalize("demo").read_text(encoding="utf-8"))
    assert m2["params_fingerprint"] == fp1
    assert "沿用" in m2["provenance"]

    # 不同参数重跑：必须说明参数变更及新旧指纹
    changed = RegistrationParams(levels=(2, 4), samples_per_axis=18, max_iter=40, min_ncc=0.5)
    runner3 = BatchRunner(out, changed)
    runner3.run_pair("pair0", moving, fixed)
    m3 = json.loads(runner3.finalize("demo").read_text(encoding="utf-8"))
    assert m3["params_fingerprint"] != fp1
    assert "变更" in m3["provenance"]
    assert m3["previous_params"]["samples_per_axis"] == 22
