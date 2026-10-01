import json

import numpy as np
import pytest

from medreg.roi import RegionOfInterest
from medreg.series import SeriesConfig, register_series, select_series_candidate
from medreg.series_batch import SeriesItem, run_series_batch, verify_series_version
from medreg.synthetic import synthetic_series, uniform_volume


def _small_series(n=3, **kwargs):
    kwargs.setdefault("shape", (28, 30, 26))
    kwargs.setdefault("step_rotation_deg", (1.0, -1.5, 1.0))
    kwargs.setdefault("step_translation", (1.5, -1.0, 0.8))
    return synthetic_series(n_timepoints=n, **kwargs)


def _fast_config(reference_index=0, **kwargs):
    kwargs.setdefault("levels", (3, 2, 1))
    kwargs.setdefault("max_iterations", 60)
    return SeriesConfig(reference_index=reference_index, **kwargs)


def test_series_keeps_per_timepoint_candidates_and_metrics():
    images, _ = _small_series(3)
    result = register_series(images, config=_fast_config())
    assert result.status == "success"
    assert result.timepoints[0].status == "reference"
    for tp in result.timepoints[1:]:
        assert tp.status == "success"
        assert tp.multi is not None
        assert tp.multi.candidates, "each timepoint keeps its own candidates"
        assert tp.multi.selected.result.coverage is not None
        assert tp.multi.selected.result.error is not None
    # Candidates are per-timepoint objects, not shared across the series.
    assert result.timepoints[1].multi is not result.timepoints[2].multi


def test_failed_timepoint_does_not_drag_down_others():
    images, _ = _small_series(3, uninformative_index=2)
    result = register_series(images, config=_fast_config())
    assert result.timepoints[1].status == "success"
    assert result.timepoints[2].status == "failed"
    assert result.timepoints[2].rejection_reasons
    assert result.status == "partial"
    gaps = [entry for entry in result.adjacency if entry["type"] == "gap"]
    assert gaps, "a failed timepoint must break the trajectory explicitly"
    assert any("t02" in entry["between"] for entry in gaps)


def test_adjacent_jump_warning_and_manual_reselect():
    images, _ = _small_series(3)
    # Give the second timepoint a wildly wrong extra initial position that
    # still converges somewhere, then force a manual choice to review the
    # series again.
    per_index = {
        1: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]],
        2: [[0.0] * 6],
    }
    config = _fast_config(jump_translation_mm=4.0, jump_rotation_deg=6.0)
    result = register_series(images, config=config, initial_sets_per_timepoint=per_index)
    tp1 = result.timepoint("t01")
    assert tp1.multi is not None and len(tp1.multi.candidates) == 2

    auto = result.adjacency
    assert all(entry["type"] in ("comparison", "gap") for entry in auto)

    # Manual selection of the other candidate recomputes adjacency so the
    # whole trajectory can be reviewed again.
    other = next(
        c.candidate_id for c in tp1.multi.candidates if c is not tp1.multi.selected
    )
    updated = select_series_candidate(result, "t01", other)
    assert updated.timepoint("t01").multi.selected.candidate_id == other
    assert updated.timepoint("t01").multi.selection_method == f"manual:{other}"
    assert updated.timepoint("t02").multi is result.timepoint("t02").multi
    # Adjacency was recomputed and reflects the manual choice.
    assert updated.adjacency is not result.adjacency
    assert len(updated.adjacency) == len(result.adjacency)


def test_manual_selection_rejects_failed_candidate_and_reference():
    images, _ = _small_series(2, uninformative_index=1)
    result = register_series(images, config=_fast_config())
    failed = result.timepoint("t01").multi.candidates[0]
    assert failed.result.status == "failed"
    with pytest.raises(ValueError, match="did not pass acceptance"):
        select_series_candidate(result, "t01", failed.candidate_id)
    with pytest.raises(ValueError, match="reference"):
        select_series_candidate(result, "t00", "candidate_00")


def test_roi_without_information_rejects_whole_series():
    images, _ = _small_series(3)
    empty_roi = RegionOfInterest((500.0, 500.0, 500.0), (510.0, 510.0, 510.0))
    result = register_series(images, config=_fast_config(), roi=empty_roi)
    for tp in result.timepoints[1:]:
        assert tp.status == "rejected"
        assert any("roi_empty_on_reference" in reason for reason in tp.rejection_reasons)
    assert result.roi_summary["problems"]

    flat = uniform_volume((20, 20, 20))
    images2 = [flat, images[1], images[2]]
    roi = RegionOfInterest((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    result2 = register_series(images2, config=_fast_config(), roi=roi)
    for tp in result2.timepoints[1:]:
        assert tp.status == "rejected"
        assert any("no_image_information" in reason for reason in tp.rejection_reasons)


def test_series_batch_roundtrip_preserves_selections_and_transforms(tmp_path):
    images, _ = _small_series(3)
    item = SeriesItem(
        "followup",
        reference_index=0,
        levels=(3, 2, 1),
        max_iterations=60,
        initial_parameter_sets=[[0.0] * 6, [0.05, -0.05, 0.05, 1.0, -1.0, 0.5]],
        roi_world_min=[-6.0, -6.0, -6.0],
        roi_world_max=[6.0, 6.0, 6.0],
    )
    report = run_series_batch([(item, images)], tmp_path)
    entry = report["series"][0]
    version_dir = tmp_path / "followup" / entry["version_id"]
    verification = verify_series_version(version_dir)
    assert verification["verified"] is True
    assert verification["labels"] == entry["labels"]

    # A manual selection creates a new version without touching the old one.
    selected = entry["timepoints"][1]["selected_candidate_id"]
    other = next(
        c["candidate_id"]
        for c in entry["timepoints"][1]["candidates"]
        if c["candidate_id"] != selected and c["status"] == "success"
    )
    manual_item = SeriesItem(
        **{**item.__dict__, "manual_selections": {"t01": other}},
    )
    report2 = run_series_batch([(manual_item, images)], tmp_path)
    entry2 = report2["series"][0]
    assert entry2["version_id"] != entry["version_id"]
    assert (tmp_path / "followup" / entry["version_id"]).exists()
    verification2 = verify_series_version(tmp_path / "followup" / entry2["version_id"])
    assert verification2["manual_selections"] == {"t01": other}
    ledger = json.loads((tmp_path / "followup" / "versions.json").read_text())
    assert len(ledger) == 2

    # Tampering with the correspondence is detected on reload.
    record_path = tmp_path / "followup" / entry2["version_id"] / "series_report.json"
    record = json.loads(record_path.read_text())
    record["timepoints"][1]["selected_candidate_id"] = "candidate_99"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="verification failed"):
        verify_series_version(tmp_path / "followup" / entry2["version_id"])

