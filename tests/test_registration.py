import json
import shutil

import numpy as np
import pytest

from medreg.batch import BatchItem, parameter_hash, run_batch
from medreg.image import MedicalImage, make_affine
from medreg.multi import InitializationSpec, register_multi
from medreg.register import evaluate_transform, register_rigid
from medreg.roi import RegionOfInterest
from medreg.synthetic import synthetic_pair
from medreg.transform import RigidTransform


def test_synthetic_rigid_registration_recovers_known_transform():
    fixed, moving, truth = synthetic_pair(
        shape=(32, 36, 30),
        rotation_angles_deg=(4.0, -5.0, 3.0),
        translation=(3.0, -2.0, 1.5),
    )
    output = "/tmp/medreg-test-batch"
    shutil.rmtree(output, ignore_errors=True)
    report = run_batch(
        [(BatchItem("case", levels=(3, 2, 1), max_iterations=70), fixed, moving)],
        output,
    )
    item = report["items"][0]
    assert item["result"]["status"] == "agreed"
    params = np.asarray(item["result"]["selected_candidate_id"] and _selected(item)["parameters_rotvec_xyz_translation_xyz"])
    estimated = RigidTransform.from_parameters(params)
    true_points = np.asarray([[0.0, 0.0, 0.0], [8.0, -6.0, 4.0], [-5.0, 3.0, 9.0]])
    assert np.allclose(estimated.apply(true_points), truth.apply(true_points), atol=2.0)
    assert _selected(item)["coverage"]["coverage_fraction"] > 0.6
    assert _selected(item)["error"]["ncc"] > 0.85
    assert item["effective_parameter_hash"]
    assert report["batch_parameter_hash"] == parameter_hash([item["effective_parameters"]])


def _selected(item_report):
    selected_id = item_report["selected_candidate_id"]
    return next(
        candidate
        for candidate in item_report["result"]["candidates"]
        if candidate["candidate_id"] == selected_id
    )


def test_uninformative_images_report_failure_without_transformed_output():
    shape = (18, 18, 18)
    affine = make_affine(shape, spacing=(1.0, 1.0, 1.0))
    fixed = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    moving = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    result = register_rigid(fixed, moving, levels=(2, 1), max_iterations=10)
    assert result.status == "failed"
    assert result.converged is False
    assert result.accepted is False
    assert result.transformed is None
    assert result.overlap_mask is None
    assert result.coverage is None
    # The failed candidate parameters are still retained for inspection.
    assert result.transform.parameters().shape == (6,)
    assert "did not meet acceptance" in result.message


def _localized_pair(shift_mm=6.0):
    """Moving equals a laterally shifted copy of fixed (plus broad background).

    The broad background structure still correlates well when the candidate is
    unshifted, so global NCC alone cannot detect that the sharp localised bumps
    are misaligned inside the user-selected ROI.
    """
    from scipy.spatial.transform import Rotation

    from medreg.resample import resample as warp

    shape = (64, 64, 64)
    fixed_affine = make_affine(shape, spacing=(1.0, 1.0, 1.0))
    moving_affine = make_affine(shape, spacing=(1.0, 1.0, 1.0))

    def volume(affine):
        grid = np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
        points = np.stack([axis.ravel() for axis in grid], axis=1)
        world = points @ affine[:3, :3].T + affine[:3, 3]
        data = np.zeros(shape, dtype=np.float32)
        data += np.exp(-(np.linalg.norm(world / 22.0, axis=1) ** 2)).reshape(shape)
        data += 0.05 * np.sin(world[:, 0] / 5.0).reshape(shape)
        for center in ([-14.0, 0.0, 0.0], [14.0, 0.0, 0.0]):
            distance = np.linalg.norm((world - np.asarray(center)) / 2.2, axis=1)
            data += 2.0 * np.exp(-(distance**2)).reshape(shape)
        return data.astype(np.float32)

    fixed_data = volume(fixed_affine)
    fixed = MedicalImage(fixed_data, fixed_affine)
    shifted = RigidTransform(Rotation.identity().as_matrix(), np.asarray([shift_mm, 0.0, 0.0]))
    reference = MedicalImage(np.zeros(shape, dtype=np.float32), moving_affine)
    warped, _ = warp(fixed, reference, shifted, order=1)
    moving = MedicalImage(warped, moving_affine)
    roi = RegionOfInterest.from_center_radius((-14.0, 0.0, 0.0), 3.5, name="left_target")
    return fixed, moving, roi, shifted


def test_roi_constraint_catches_local_misalignment():
    fixed, moving, roi, shifted = _localized_pair(shift_mm=6.0)
    global_only = evaluate_transform(fixed, moving, RigidTransform(), min_coverage=0.8)
    roi_result = evaluate_transform(fixed, moving, RigidTransform(), roi=roi, min_coverage=0.8, min_roi_ncc=0.6)
    assert global_only.status == "success"
    assert global_only.error["ncc"] > 0.85
    assert roi_result.status == "failed"
    assert roi_result.final_ncc > 0.85
    assert roi_result.transformed is None
    assert roi_result.roi_coverage is None and roi_result.roi_error is None
    # Moving was built as shifted(fixed); identity is the wrong placement and
    # the inverse shift recovers the correct physical alignment.
    aligned = evaluate_transform(fixed, moving, shifted.inverse(), roi=roi, min_coverage=0.8, min_roi_ncc=0.6)
    assert aligned.status == "success"
    assert aligned.roi_coverage["roi_coverage_fraction"] > 0.9
    assert aligned.roi_error["ncc"] > 0.9


def test_partial_overlap_and_orientation_handled_in_physical_space():
    from scipy.spatial.transform import Rotation

    from medreg.resample import resample as warp

    fixed_shape = (34, 30, 28)
    fixed_affine = make_affine(fixed_shape, spacing=(1.0, 1.2, 0.9))
    fixed_data = np.zeros(fixed_shape, dtype=np.float32)
    ii, jj, kk = np.indices(fixed_shape)
    grid = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)
    world = grid @ fixed_affine[:3, :3].T + fixed_affine[:3, 3]
    for center, scale in (([0.0, 0.0, 0.0], 4.0), ([7.0, -5.0, 3.0], 2.5)):
        distance = np.linalg.norm((world - np.asarray(center)) / scale, axis=1)
        fixed_data += np.exp(-(distance**2)).reshape(fixed_shape)
    fixed = MedicalImage(fixed_data.astype(np.float32), fixed_affine)

    moving_shape = (26, 34, 24)
    flipped_direction = Rotation.from_euler("xyz", (0.0, 0.0, 40.0), degrees=True).as_matrix()
    moving_affine = make_affine(moving_shape, spacing=(1.1, 0.9, 1.3), direction=flipped_direction)
    truth = RigidTransform(
        Rotation.from_euler("xyz", (6.0, -8.0, 5.0), degrees=True).as_matrix(),
        np.asarray([4.0, -3.0, 2.0]),
    )
    reference = MedicalImage(np.zeros(moving_shape, dtype=np.float32), moving_affine)
    warped, _ = warp(fixed, reference, truth, order=3)
    moving = MedicalImage(warped, moving_affine)

    result = register_rigid(
        fixed,
        moving,
        RigidTransform.from_parameters(np.r_[0.1, -0.15, 0.1, 4.0, -3.0, 2.0]),
        levels=(3, 2, 1),
        max_iterations=90,
        min_coverage=0.3,
    )
    assert result.status == "success"
    points = np.asarray([[0.0, 0.0, 0.0], [6.0, -4.0, 3.0]])
    assert np.allclose(result.transform.apply(points), truth.inverse().apply(points), atol=1.5)


def test_multi_initialization_retains_uncertain_candidates_without_fake_outputs():
    fixed, moving, roi, _ = _localized_pair(shift_mm=6.0)
    initializations = [
        # Identity placement matches the broad background (globally fine) but
        # misses the ROI; this candidate must be retained but rejected.
        InitializationSpec("start_identity", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lock_initial=True),
        InitializationSpec("start_far", [0.0, 0.0, 0.0, 40.0, 40.0, 40.0], lock_initial=True),
    ]
    multi = register_multi(
        fixed,
        moving,
        initializations,
        levels=(1,),
        max_iterations=1,
        min_coverage=0.5,
        roi=roi,
        min_roi_ncc=0.6,
    )
    assert len(multi.candidates) == 2
    for candidate in multi.candidates:
        assert not candidate.result.accepted
        assert candidate.result.transformed is None
    assert multi.selected_candidate() is None
    assert multi.status == "failed"


def _synthetic_ambiguous():
    shape = (48, 48, 48)
    affine = make_affine(shape, spacing=(1.0, 1.0, 1.0))
    grid = np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
    points = np.stack([axis.ravel() for axis in grid], axis=1)
    world = points @ affine[:3, :3].T + affine[:3, 3]
    data = np.zeros(shape, dtype=np.float32)
    for center in ([-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]):
        distance = np.linalg.norm((world - np.asarray(center)) / 2.6, axis=1)
        data += np.exp(-(distance**2)).reshape(shape)
    return MedicalImage(data.astype(np.float32), affine)


def test_similar_but_disagreeing_candidates_are_flagged_uncertain():
    fixed = _synthetic_ambiguous()
    moving = fixed
    initializations = [
        InitializationSpec("left_lock", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], lock_initial=True),
        InitializationSpec("right_lock", [0.0, 0.0, 0.0, 18.0, 0.0, 0.0], lock_initial=True),
    ]
    multi = register_multi(
        fixed,
        moving,
        initializations,
        levels=(1,),
        max_iterations=1,
        min_coverage=0.5,
        translation_tolerance_mm=2.0,
        score_tolerance=0.1,
    )
    assert multi.status == "uncertain"
    assert multi.uncertainty["ambiguous"] is True
    assert multi.uncertainty["max_translation_diff_mm"] > 2.0
    assert multi.selected_candidate_id is not None
    summary = multi.summary()
    assert all("parameters_rotvec_xyz_translation_xyz" in candidate for candidate in summary["candidates"])


def test_parameter_changes_and_manual_selection_create_separate_versions(tmp_path):
    fixed = _synthetic_ambiguous()
    moving = fixed
    item = BatchItem(
        "dup",
        levels=(1,),
        max_iterations=1,
        min_coverage=0.5,
        initializations=(
            {"candidate_id": "left_lock", "initial_parameters": [0, 0, 0, 0, 0, 0], "lock_initial": True},
            {"candidate_id": "right_lock", "initial_parameters": [0, 0, 0, 18, 0, 0], "lock_initial": True},
        ),
    )
    first = run_batch([(item, fixed, moving)], tmp_path)
    second = run_batch([(item, fixed, moving)], tmp_path)
    assert first["version"] == "v0001"
    assert second["version"] == "v0002"
    registry = json.loads((tmp_path / "versions" / "versions.json").read_text(encoding="utf-8"))
    assert [entry["trigger"] for entry in registry["versions"]] == ["initial", "rerun"]

    manual_item = BatchItem(
        "dup",
        levels=(1,),
        max_iterations=1,
        min_coverage=0.5,
        initializations=(
            {"candidate_id": "left_lock", "initial_parameters": [0, 0, 0, 0, 0, 0], "lock_initial": True},
            {"candidate_id": "right_lock", "initial_parameters": [0, 0, 0, 18, 0, 0], "lock_initial": True},
        ),
        manual_candidate_id="right_lock",
    )
    third = run_batch([(manual_item, fixed, moving)], tmp_path)
    item_report = third["items"][0]
    assert item_report["selected_candidate_id"] == "right_lock"
    assert item_report["result"]["status"] == "selected_manual"
    registry = json.loads((tmp_path / "versions" / "versions.json").read_text(encoding="utf-8"))
    assert registry["versions"][2]["trigger"] == "manual_selection"
    assert registry["versions"][2]["selection_mode"] == "manual"
    assert (tmp_path / "versions" / "v0001" / "dup" / "batch_report.json").exists() is False
    assert (tmp_path / "versions" / "v0001" / "dup" / "moving_original.nii.gz").exists()
    assert (tmp_path / "versions" / "v0003" / "dup" / "moving_original.nii.gz").exists()
    # Only the selected accepted candidate may produce a success image.
    assert (tmp_path / "versions" / "v0003" / "dup" / "moving_transformed.nii.gz").exists()


def test_manual_selection_of_failed_candidate_is_rejected():
    shape = (16, 16, 16)
    affine = make_affine(shape)
    blank = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    item = BatchItem(
        "blank",
        levels=(1,),
        max_iterations=1,
        initializations=(
            {"candidate_id": "a", "initial_parameters": [0, 0, 0, 0, 0, 0]},
            {"candidate_id": "b", "initial_parameters": [0, 0, 0, 5, 5, 5]},
        ),
        manual_candidate_id="a",
    )
    with pytest.raises(ValueError, match="failed acceptance"):
        run_batch([(item, blank, blank)], "/tmp/medreg-manual-fail")
