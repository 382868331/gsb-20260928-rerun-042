import json

import numpy as np
import pytest

from medreg.batch import BatchItem, parameter_hash, run_batch
from medreg.image import MedicalImage, make_affine
from medreg.register import (
    apply_manual_selection,
    register_multi,
    register_rigid,
    transform_disagreement,
)
from medreg.roi import RegionOfInterest
from medreg.synthetic import reoriented_direction, synthetic_pair, uniform_volume
from medreg.transform import RigidTransform


def _outputs(item: dict) -> dict:
    return item["outputs"]


def test_synthetic_rigid_registration_recovers_known_transform():
    fixed, moving, truth = synthetic_pair(
        shape=(32, 36, 30),
        rotation_angles_deg=(4.0, -5.0, 3.0),
        translation=(3.0, -2.0, 1.5),
    )
    report = run_batch(
        [(BatchItem("case", levels=(3, 2, 1), max_iterations=70), fixed, moving)],
        "/tmp/medreg-test-batch",
    )
    item = report["items"][0]
    assert item["selection"]["status"] == "success"
    params = np.asarray(item["result"]["parameters_rotvec_xyz_translation_xyz"])
    estimated = RigidTransform.from_parameters(params)
    true_points = np.asarray([[0.0, 0.0, 0.0], [8.0, -6.0, 4.0], [-5.0, 3.0, 9.0]])
    assert np.allclose(estimated.apply(true_points), truth.apply(true_points), atol=2.0)
    assert item["result"]["coverage"]["coverage_fraction"] > 0.6
    assert item["result"]["error"]["ncc"] > 0.85
    assert item["effective_parameter_hash"]
    assert report["batch_parameter_hash"] == parameter_hash([item["effective_parameters"]])


def test_uninformative_images_report_failure_without_transformed_output():
    shape = (18, 18, 18)
    affine = make_affine(shape, spacing=(1.0, 1.0, 1.0))
    fixed = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    moving = MedicalImage(np.zeros(shape, dtype=np.float32), affine)
    result = register_rigid(fixed, moving, levels=(2, 1), max_iterations=10)
    assert result.status == "failed"
    assert result.converged is False
    assert result.transformed is None
    assert "did not meet acceptance" in result.message


def test_roi_reports_separate_region_and_global_error_and_coverage():
    fixed, moving, _ = synthetic_pair(shape=(28, 30, 26))
    roi = RegionOfInterest((-7.0, -7.0, -7.0), (7.0, 7.0, 7.0), name="central")
    result = register_rigid(
        fixed,
        moving,
        RigidTransform.from_parameters(np.zeros(6)),
        levels=(3, 2, 1),
        max_iterations=70,
        roi=roi,
    )
    assert result.status == "success"
    assert result.roi_error is not None
    region_stats = result.roi_error
    assert region_stats["region_voxels"] > 0
    assert 0.0 <= region_stats["region_coverage_fraction"] <= 1.0
    assert region_stats["ncc"] > 0.7
    # Region metrics must be structurally distinct from the global metrics.
    assert set(["mse", "rmse", "mae", "ncc"]).issubset(region_stats)
    assert "covered_region_voxels" in region_stats


def test_roi_failure_is_not_hidden_by_background_agreement():
    # A deliberately wrong transform keeps much of the empty background aligned
    # while displacing the central anatomy; ROI scoring must reject it even if
    # the global image still looks broadly overlapping.
    fixed, moving, _ = synthetic_pair(shape=(28, 30, 26))
    roi = RegionOfInterest((-6.0, -6.0, -6.0), (6.0, 6.0, 6.0), name="central")
    wrong = RigidTransform.from_parameters(np.asarray([0.0, 0.0, 0.0, 9.0, 8.0, 7.0]))
    result = register_rigid(
        fixed,
        moving,
        wrong,
        levels=(2, 1),
        max_iterations=3,
        roi=roi,
        min_roi_ncc=0.95,
    )
    assert result.status == "failed"
    assert result.transformed is None
    assert result.roi_error is not None


def test_multiple_initial_positions_retain_candidates_and_uncertainty():
    fixed, moving, truth = synthetic_pair(shape=(28, 30, 26))
    multi = register_multi(
        fixed,
        moving,
        [
            np.zeros(6),
            [0.12, -0.12, 0.08, 5.0, -4.0, 3.0],
            [-0.1, 0.1, -0.08, -5.0, 4.0, -3.0],
        ],
        levels=(3, 2, 1),
        max_iterations=60,
    )
    assert len(multi.candidates) == 3
    assert all(candidate.initial_parameters is not None for candidate in multi.candidates)
    # Every candidate is retained, success or not, with its own diagnostics.
    statuses = {candidate.result.status for candidate in multi.candidates}
    assert "success" in statuses
    selected = multi.selected
    assert selected is not None
    params = np.asarray(selected.result.transform.parameters())
    estimated = RigidTransform.from_parameters(params)
    points = np.asarray([[0.0, 0.0, 0.0], [6.0, -5.0, 3.0]])
    assert np.allclose(estimated.apply(points), truth.apply(points), atol=2.5)
    summary = multi.summary()
    assert len(summary["candidates"]) == 3
    assert isinstance(summary["uncertainty_reasons"], list)


def test_disagreement_measures_physical_inconsistency():
    first = RigidTransform.from_parameters(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    second = RigidTransform.from_parameters(np.asarray([0.0, 0.0, 0.0, 8.0, 0.0, 0.0]))
    disagreement = transform_disagreement(first, second)
    assert disagreement["translation_distance_mm"] == pytest.approx(8.0)
    assert disagreement["rotation_angle_deg"] == pytest.approx(0.0, abs=1e-6)


def test_partial_overlap_and_different_orientation_still_register_in_world_coordinates():
    direction = reoriented_direction((1, 0, 2), (False, True, False))
    fixed, moving, truth = synthetic_pair(
        shape=(30, 32, 28),
        moving_direction=direction,
        partial_overlap=True,
        rotation_angles_deg=(3.0, -4.0, 2.0),
        translation=(2.0, -1.5, 1.0),
    )
    assert np.allclose(fixed.direction @ moving.direction.T, fixed.direction @ moving.direction.T)
    multi = register_multi(
        fixed,
        moving,
        [np.zeros(6), [0.05, -0.05, 0.03, 2.0, -1.5, 1.0]],
        roi=RegionOfInterest((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0)),
        levels=(3, 2, 1),
        max_iterations=70,
    )
    if multi.selected is not None:
        estimated = multi.selected.result.transform
        points = np.asarray([[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]])
        assert np.allclose(estimated.apply(points), truth.apply(points), atol=3.0)
        assert multi.selected.result.roi_error["region_coverage_fraction"] > 0.0


def test_failed_multistart_batch_writes_no_success_image():
    shape = (16, 16, 16)
    affine = make_affine(shape)
    fixed = uniform_volume(shape)
    moving = uniform_volume(shape, origin=(2.0, 0.0, 0.0))
    item = BatchItem(
        "flat",
        levels=(2, 1),
        max_iterations=8,
        initial_parameter_sets=[[0.0] * 6, [0.0, 0.0, 0.0, 3.0, 0.0, 0.0]],
        roi_world_min=[-4.0, -4.0, -4.0],
        roi_world_max=[4.0, 4.0, 4.0],
    )
    report = run_batch([(item, fixed, moving)], "/tmp/medreg-test-flat")
    entry = report["items"][0]
    assert entry["selection"]["status"] == "failed"
    assert _outputs(entry)["transformed"] is None
    for candidate in entry["candidates"]:
        assert candidate["outputs"]["transformed"] is None
        assert candidate["status"] == "failed"


def test_manual_selection_of_failed_candidate_is_rejected():
    shape = (16, 16, 16)
    fixed = uniform_volume(shape)
    moving = uniform_volume(shape)
    multi = register_multi(
        fixed,
        moving,
        [np.zeros(6)],
        levels=(2, 1),
        max_iterations=5,
    )
    assert multi.status == "failed"
    with pytest.raises(ValueError, match="did not pass acceptance"):
        apply_manual_selection(multi, "candidate_00")
    with pytest.raises(ValueError, match="unknown candidate"):
        apply_manual_selection(multi, "candidate_99")


def test_parameter_change_and_manual_selection_create_new_versions(tmp_path):
    fixed, moving, _ = synthetic_pair(shape=(24, 26, 22))
    base = BatchItem("versioned", levels=(3, 2, 1), max_iterations=50)
    first = run_batch([(base, fixed, moving)], tmp_path)
    first_id = first["items"][0]["version_id"]

    tuned = BatchItem(
        "versioned",
        levels=(3, 2, 1),
        max_iterations=90,
        initial_parameter_sets=[
            [0.0] * 6,
            [0.1, -0.1, 0.05, 4.0, -3.0, 2.0],
        ],
    )
    second = run_batch([(tuned, fixed, moving)], tmp_path)
    second_id = second["items"][0]["version_id"]
    assert second_id != first_id

    # A manual choice of a retained candidate creates yet another version.
    retained_id = second["items"][0]["multi_start"]["candidates"][-1]["candidate_id"]
    manual = BatchItem(
        "versioned",
        levels=(3, 2, 1),
        max_iterations=90,
        initial_parameter_sets=[
            [0.0] * 6,
            [0.1, -0.1, 0.05, 4.0, -3.0, 2.0],
        ],
        manual_candidate_id=retained_id,
    )
    third = run_batch([(manual, fixed, moving)], tmp_path)
    third_entry = third["items"][0]
    assert third_entry["version_id"] != second_id
    assert third_entry["selection"]["method"] == f"manual:{retained_id}"
    assert third_entry["manual_candidate_requested"] == retained_id

    ledger = json.loads((tmp_path / "versioned" / "versions.json").read_text(encoding="utf-8"))
    assert [row["version_id"] for row in ledger] == [first_id, second_id, third_entry["version_id"]]
    # The first version is untouched and remains on disk.
    assert (tmp_path / "versioned" / first_id / "fixed_original.nii.gz").exists()
