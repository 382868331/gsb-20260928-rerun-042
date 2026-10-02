import json

import numpy as np
import pytest

from medreg.propagation import PropagationConfig, PropagationSession
from medreg.propagation_store import load_propagation, save_propagation, verify_propagation
from medreg.series import SeriesConfig, register_series
from medreg.synthetic import synthetic_series


def _series(n=4, **kwargs):
    kwargs.setdefault("shape", (28, 30, 26))
    kwargs.setdefault("step_rotation_deg", (1.0, -1.5, 1.0))
    kwargs.setdefault("step_translation", (1.5, -1.0, 0.8))
    return synthetic_series(n_timepoints=n, **kwargs)


def _config(**kwargs):
    kwargs.setdefault("levels", (3, 2, 1))
    kwargs.setdefault("max_iterations", 60)
    return SeriesConfig(reference_index=0, **kwargs)


def _drawn_region(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[10:18, 12:20, 8:16] = True
    return mask


def _session(n=4, series_kwargs=None, config=None, per_index=None):
    images, _ = _series(n, **(series_kwargs or {}))
    result = register_series(
        images, config=config or _config(), initial_sets_per_timepoint=per_index
    )
    session = PropagationSession(images=images, series=result)
    session.draw_region(_drawn_region(images[0].shape))
    return session, images


def test_propagation_keeps_candidate_coverage_and_decision_per_timepoint():
    session, _ = _session(3)
    assert session.records["t00"].status == "source"
    assert session.records["t00"].decision == "original"
    for label in ("t01", "t02"):
        record = session.records[label]
        assert record.status == "propagated"
        assert record.decision == "auto"
        assert record.coverage_fraction >= session.config.min_region_coverage
        assert record.basis["candidate_id"] is not None
        assert record.basis["basis_label"] == "t00"
        assert record.basis["basis_kind"] == "original"
        assert record.mask is not None and record.mask.any()
    curve = {entry["label"]: entry for entry in session.change_curve()}
    assert curve["t00"]["origin"] == "original"
    assert curve["t01"]["origin"] == "auto"
    assert curve["t02"]["volume_mm3"] is not None
    edges = session.lineage()
    assert {edge["to_label"] for edge in edges} == {"t01", "t02"}
    assert all(edge["from_label"] == "t00" for edge in edges)


def test_failed_timepoint_stays_pending_without_fabricated_contour():
    session, _ = _session(3, series_kwargs={"uninformative_index": 2})
    record = session.records["t02"]
    assert record.status == "pending"
    assert record.mask is None
    assert any("t02" in reason for reason in record.reasons)
    curve = {entry["label"]: entry for entry in session.change_curve()}
    assert curve["t02"]["origin"] == "pending"
    assert curve["t02"]["volume_mm3"] is None
    # Neighbouring timepoints are unaffected by the gap.
    assert session.records["t01"].status == "propagated"


def test_insufficient_region_coverage_marks_pending_not_extrapolated():
    images, _ = _series(2)
    result = register_series(images, config=_config())
    session = PropagationSession(
        series=result, images=images, config=PropagationConfig(min_region_coverage=1.0)
    )
    # A region at the very edge of the volume loses voxels under any drift.
    mask = np.zeros(images[0].shape, dtype=bool)
    mask[0:3, 0:3, 0:3] = True
    session.draw_region(mask)
    record = session.records["t01"]
    assert record.status == "pending"
    assert record.coverage_fraction < 1.0
    assert any("region_coverage_insufficient" in reason for reason in record.reasons)
    # The partial mask is kept for inspection but not treated as a contour.
    assert record.mask is not None
    assert record.mask.sum() < mask.sum()


def test_correction_as_basis_repropagates_downstream_and_revert_restores():
    session, _ = _session(4)
    original_before = session.original_mask.copy()
    auto_t03_before = session.records["t03"].mask.copy()

    correction = _drawn_region(session.reference.shape)
    correction[20:24, 4:8, 18:22] = True  # user adds a second blob
    session.apply_correction("t02", correction, use_as_basis=True)

    assert session.records["t02"].status == "manual"
    assert session.records["t02"].decision == "manual"
    # Upstream of the correction nothing changes.
    assert session.records["t01"].basis["basis_kind"] == "original"
    # Downstream re-propagates from the correction.
    assert session.records["t03"].basis["basis_kind"] == "correction"
    assert session.records["t03"].basis["basis_label"] == "t02"
    assert not np.array_equal(session.records["t03"].mask, auto_t03_before)
    curve = {entry["label"]: entry for entry in session.change_curve()}
    assert curve["t02"]["origin"] == "manual"

    session.revert_correction("t02")
    assert session.records["t02"].status == "propagated"
    assert session.records["t03"].basis["basis_kind"] == "original"
    assert np.array_equal(session.records["t03"].mask, auto_t03_before)
    # The original drawing was never touched.
    assert np.array_equal(session.original_mask, original_before)
    with pytest.raises(ValueError, match="original drawing already exists"):
        session.draw_region(_drawn_region(session.reference.shape))


def test_preview_and_apply_candidate_change_keeps_manual_results():
    per_index = {
        1: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]],
        2: [[0.0] * 6],
    }
    session, _ = _session(3, per_index=per_index)
    correction = _drawn_region(session.reference.shape)
    correction[2:6, 2:6, 2:6] = True
    session.apply_correction("t02", correction)

    tp1 = session.series.timepoint("t01")
    other = next(
        c.candidate_id for c in tp1.multi.candidates if c is not tp1.multi.selected
    )
    preview = session.preview_candidate_change("t01", other)
    assert preview["auto_results_to_update"] == ["t01"]
    assert preview["manual_conflicts"] == []
    assert preview["manual_kept"] == ["t02"]

    session.apply_candidate_change("t01", other)
    assert session.series.timepoint("t01").multi.selected.candidate_id == other
    assert session.records["t01"].basis["candidate_id"] == other
    assert session.records["t01"].basis["selection_method"] == f"manual:{other}"
    # The manual correction at t02 is preserved untouched.
    assert session.records["t02"].status == "manual"
    assert np.array_equal(session.records["t02"].mask, correction)


def test_preview_flags_manual_conflict_at_changed_timepoint():
    per_index = {1: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    session, _ = _session(2, per_index=per_index)
    session.apply_correction("t01", _drawn_region(session.reference.shape))
    tp1 = session.series.timepoint("t01")
    other = next(
        c.candidate_id for c in tp1.multi.candidates if c is not tp1.multi.selected
    )
    preview = session.preview_candidate_change("t01", other)
    assert preview["manual_conflicts"] == ["t01"]
    assert preview["auto_results_to_update"] == []


def test_save_reopen_verify_roundtrip(tmp_path):
    per_index = {1: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    session, images = _session(3, per_index=per_index)
    correction = _drawn_region(session.reference.shape)
    correction[22:26, 22:26, 2:6] = True
    session.apply_correction("t01", correction, use_as_basis=True)

    report = save_propagation(session, tmp_path / "v1")
    assert report["schema"] == "medreg-propagation-v1"
    assert (tmp_path / "v1" / "propagation_report.json").exists()

    result = verify_propagation(tmp_path / "v1", session.series, images)
    assert result["verified"] is True
    assert result["corrections"] == ["t01"]

    reopened = load_propagation(tmp_path / "v1", session.series, images)
    for label in session.series.labels:
        before = session.records[label]
        after = reopened.records[label]
        assert before.status == after.status
        if before.mask is None:
            assert after.mask is None
        else:
            assert np.array_equal(before.mask, after.mask)
    curve = {entry["label"]: entry["origin"] for entry in reopened.change_curve()}
    assert curve == {"t00": "original", "t01": "manual", "t02": "auto"}
    # Traceability: every propagated record keeps candidate and transform.
    record = reopened.records["t02"]
    assert record.basis["candidate_id"] is not None
    assert len(record.basis["transform_parameters"]) == 6
    assert record.basis["basis_kind"] == "correction"


def test_verify_detects_tampered_report(tmp_path):
    session, images = _session(2)
    save_propagation(session, tmp_path / "v1")
    report_path = tmp_path / "v1" / "propagation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["records"][1]["coverage_fraction"] = 0.01
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="propagation verification failed"):
        verify_propagation(tmp_path / "v1", session.series, images)

