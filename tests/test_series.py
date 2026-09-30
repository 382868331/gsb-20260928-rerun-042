import json

import numpy as np
import pytest

from medreg.image import MedicalImage, make_affine
from medreg.roi import RegionOfInterest
from medreg.series import (
    apply_series_manual_selection,
    compute_jump_warnings,
    register_series,
)
from medreg.series_batch import SeriesItem, load_series_state, run_series_batch
from medreg.synthetic import synthetic_series, uniform_volume
from medreg.transform import RigidTransform


def test_series_registers_each_time_independently_and_recovers_truth():
    images, truths = synthetic_series(shape=(28, 30, 26))
    series = register_series(
        images,
        "t0",
        [[0.0] * 6, [0.05, -0.03, 0.03, 3.0, -2.0, 1.5]],
        levels=(3, 2, 1),
        max_iterations=70,
    )
    assert series.status == "success"
    points = np.asarray([[0.0, 0.0, 0.0], [5.0, -4.0, 3.0]])
    for point in series.time_points:
        if point.is_reference:
            assert point.status == "reference"
            continue
        assert point.status == "success"
        assert point.multi is not None
        # Each time point keeps its own candidates, coverage and errors.
        assert len(point.multi.candidates) == 2
        selected = point.multi.selected
        estimated = selected.result.transform
        assert np.allclose(estimated.apply(points), truths[point.time_id].apply(points), atol=2.5)
        assert selected.result.coverage["coverage_fraction"] > 0.5


def test_failed_time_point_does_not_drag_down_others():
    images, _ = synthetic_series(shape=(24, 26, 22), blank_time_id="t2")
    series = register_series(images, "t0", [[0.0] * 6], levels=(3, 2, 1), max_iterations=60)
    assert series.status == "partial"
    by_id = {point.time_id: point for point in series.time_points}
    assert by_id["t1"].status == "success"
    assert by_id["t3"].status == "success"
    failed = by_id["t2"]
    assert failed.status == "failed"
    assert failed.rejection_reasons
    # The failed time keeps its own (failed) candidates for review.
    assert failed.multi is not None
    assert all(candidate.result.status == "failed" for candidate in failed.multi.candidates)
    # The gap is reported, not interpolated over.
    gaps = [w for w in series.jump_warnings if w["kind"] == "missing_result"]
    assert any(w["to_time_id"] == "t2" for w in gaps)


def test_large_change_between_adjacent_times_is_flagged():
    images, _ = synthetic_series(shape=(24, 26, 22), jump_time_id="t2")
    series = register_series(
        images,
        "t0",
        [[0.0] * 6, [-0.05, 0.03, -0.03, -11.0, -4.0, 5.0]],
        levels=(3, 2, 1),
        max_iterations=80,
        jump_translation_mm=4.0,
    )
    large = [w for w in series.jump_warnings if w["kind"] == "large_change"]
    assert large, "expected a large-change warning around the jumped time point"
    assert any(w["from_time_id"] == "t1" and w["to_time_id"] == "t2" for w in large) or any(
        w["to_time_id"] == "t2" for w in large
    )


def test_manual_reselection_recomputes_series_review():
    images, _ = synthetic_series(shape=(24, 26, 22))
    series = register_series(
        images,
        "t0",
        [[0.0] * 6, [0.05, -0.03, 0.03, 3.0, -2.0, 1.5]],
        levels=(3, 2, 1),
        max_iterations=60,
    )
    point = series.time_point("t1")
    other = [c for c in point.multi.candidates if c is not point.multi.selected]
    if not other:
        pytest.skip("only one candidate retained")
    target = next((c for c in other if c.result.status == "success"), None)
    if target is None:
        with pytest.raises(ValueError, match="did not pass acceptance"):
            apply_series_manual_selection(series, "t1", other[0].candidate_id)
        return
    updated = apply_series_manual_selection(series, "t1", target.candidate_id)
    assert updated.time_point("t1").multi.selected.candidate_id == target.candidate_id
    assert updated.time_point("t1").multi.selection_method == f"manual:{target.candidate_id}"
    # Other time points are untouched and warnings are recomputed for the series.
    assert updated.time_point("t2") is series.time_point("t2")
    assert isinstance(updated.jump_warnings, list)


def test_uninformative_roi_rejects_conclusions_explicitly():
    images, _ = synthetic_series(shape=(24, 26, 22))
    flat = uniform_volume((24, 26, 22))
    images = dict(images)
    images["t0"] = flat
    roi = RegionOfInterest((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    series = register_series(images, "t0", [[0.0] * 6], roi=roi, levels=(2, 1), max_iterations=10)
    assert series.status == "failed"
    for point in series.time_points:
        if point.is_reference:
            continue
        assert point.status == "rejected"
        assert any("no image information" in reason for reason in point.rejection_reasons)


def test_roi_outside_grid_rejects_all_times():
    images, _ = synthetic_series(shape=(20, 22, 18))
    roi = RegionOfInterest((500.0, 500.0, 500.0), (510.0, 510.0, 510.0))
    series = register_series(images, "t0", [[0.0] * 6], roi=roi, levels=(2, 1), max_iterations=5)
    for point in series.time_points:
        if not point.is_reference:
            assert point.status == "rejected"
            assert any("no voxels" in reason for reason in point.rejection_reasons)


def test_series_batch_persists_and_reloads_one_to_one(tmp_path):
    images, _ = synthetic_series(shape=(24, 26, 22))
    item = SeriesItem(
        "case_series",
        reference_time_id="t0",
        time_order=("t0", "t1", "t2", "t3"),
        initial_parameter_sets=[[0.0] * 6, [0.05, -0.03, 0.03, 3.0, -2.0, 1.5]],
        levels=(3, 2, 1),
        max_iterations=60,
    )
    report = run_series_batch([(item, images)], tmp_path)
    entry = report["items"][0]
    assert entry["status"] == "success"
    version_dir = tmp_path / "case_series" / entry["version_id"]
    assert (version_dir / "series_state.json").exists()
    assert (version_dir / "series_report.json").exists()

    loaded = load_series_state(version_dir)
    assert loaded["reference_time_id"] == "t0"
    for time_id in ("t0", "t1", "t2", "t3"):
        state = loaded["times"][time_id]
        assert isinstance(state["image"], MedicalImage)
        if time_id == "t0":
            assert state["selected_candidate_id"] is None
            continue
        assert state["selected_candidate_id"] in state["candidate_ids"]
        assert isinstance(state["transform"], RigidTransform)
        # Original image, selection and transform correspond one-to-one.
        assert loaded["selections"][time_id] == state["selected_candidate_id"]

    # A manual re-selection creates a new version without touching the first.
    first_selected = loaded["selections"]["t1"]
    manual = SeriesItem(
        "case_series",
        reference_time_id="t0",
        time_order=("t0", "t1", "t2", "t3"),
        initial_parameter_sets=[[0.0] * 6, [0.05, -0.03, 0.03, 3.0, -2.0, 1.5]],
        levels=(3, 2, 1),
        max_iterations=60,
        manual_selections={"t1": first_selected},
    )
    second = run_series_batch([(manual, images)], tmp_path)
    second_entry = second["items"][0]
    assert second_entry["version_id"] != entry["version_id"]
    assert second_entry["manual_selections"] == {"t1": first_selected}
    reloaded = load_series_state(tmp_path / "case_series" / second_entry["version_id"])
    assert reloaded["selections"]["t1"] == first_selected
    ledger = json.loads((tmp_path / "case_series" / "versions.json").read_text(encoding="utf-8"))
    assert [row["version_id"] for row in ledger] == [entry["version_id"], second_entry["version_id"]]
    assert (tmp_path / "case_series" / entry["version_id"] / "series_state.json").exists()


def test_load_series_state_detects_missing_original(tmp_path):
    images, _ = synthetic_series(shape=(20, 22, 18))
    item = SeriesItem(
        "broken",
        reference_time_id="t0",
        initial_parameter_sets=[[0.0] * 6],
        levels=(2, 1),
        max_iterations=20,
    )
    report = run_series_batch([(item, images)], tmp_path)
    version_dir = tmp_path / "broken" / report["items"][0]["version_id"]
    state = json.loads((version_dir / "series_state.json").read_text(encoding="utf-8"))
    missing = state["times"]["t1"]["moving_original"]
    import os

    os.remove(missing)
    with pytest.raises(ValueError, match="missing"):
        load_series_state(version_dir)
