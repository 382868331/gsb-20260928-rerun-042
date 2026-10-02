import json

import numpy as np
import pytest

from medreg.propagation import PropagationSession
from medreg.propagation_batch import (
    load_propagation,
    save_propagation,
    verify_propagation_version,
)
from medreg.series import SeriesConfig, register_series
from medreg.synthetic import synthetic_series


def _series(n=4, **kwargs):
    kwargs.setdefault("shape", (28, 30, 26))
    kwargs.setdefault("step_rotation_deg", (1.0, -1.5, 1.0))
    kwargs.setdefault("step_translation", (1.5, -1.0, 0.8))
    return synthetic_series(n_timepoints=n, **kwargs)


def _config():
    return SeriesConfig(reference_index=0, levels=(3, 2, 1), max_iterations=60)


def _anchor_mask(image):
    mask = np.zeros(image.shape, dtype=bool)
    mask[10:16, 12:18, 9:15] = True
    return mask


def _session(n=4, **kwargs):
    images, _ = _series(n, **kwargs)
    series = register_series(images, config=_config())
    return images, series, PropagationSession(images, series, "t00", _anchor_mask(images[0]))


def test_propagation_records_candidate_coverage_and_region_per_timepoint():
    _, _, session = _session(3)
    assert session.region("t00").kind == "source"
    for label in ("t01", "t02"):
        region = session.region(label)
        assert region.kind == "auto"
        assert region.mask is not None and region.mask.any()
        assert region.candidate_id is not None
        assert region.coverage_fraction >= session.min_region_coverage
        assert region.transform_parameters is not None
        assert region.basis_label == "t00"


def test_failed_timepoint_stays_pending_without_fabricated_contour():
    images, series, session = _session(4, uninformative_index=2)
    assert series.timepoint("t02").status == "failed"
    pending = session.region("t02")
    assert pending.kind == "pending"
    assert pending.mask is None
    assert pending.reasons
    # Neighbours are unaffected and the curve carries an explicit gap.
    assert session.region("t03").kind == "auto"
    curve = {entry["label"]: entry for entry in session.change_curve()}
    assert curve["t02"]["volume_mm3"] is None
    assert curve["t03"]["volume_mm3"] is not None


def test_manual_correction_as_new_basis_rewires_visible_graph():
    images, _, session = _session(4)
    before = session.region("t03").mask.copy()
    correction = np.zeros(images[1].shape, dtype=bool)
    correction[8:14, 10:16, 8:14] = True
    session.apply_correction("t01", correction, use_as_basis=True)
    assert session.region("t01").kind == "manual"
    assert np.array_equal(session.region("t01").mask, correction)
    graph = {(edge["source"], edge["target"]): edge for edge in session.propagation_graph()}
    assert graph[("t01", "t02")]["basis_is_correction"]
    assert graph[("t01", "t03")]["basis_is_correction"]
    assert not np.array_equal(session.region("t03").mask, before)
    # Original drawing is untouched.
    assert np.array_equal(session.anchor_mask, _anchor_mask(images[0]))


def test_withdraw_correction_restores_auto_and_keeps_anchor():
    images, _, session = _session(3)
    auto_before = session.region("t01").mask.copy()
    correction = np.zeros(images[1].shape, dtype=bool)
    correction[5:9, 5:9, 5:9] = True
    session.apply_correction("t01", correction)
    assert session.region("t01").kind == "manual"
    session.withdraw_correction("t01")
    restored = session.region("t01")
    assert restored.kind == "auto"
    assert np.array_equal(restored.mask, auto_before)
    assert np.array_equal(session.anchor_mask, _anchor_mask(images[0]))
    with pytest.raises(KeyError):
        session.withdraw_correction("t01")


def test_preview_candidate_selection_lists_updates_and_conflicts():
    images, _ = _series(3)
    per_index = {1: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    series = register_series(images, config=_config(), initial_sets_per_timepoint=per_index)
    session = PropagationSession(images, series, "t00", _anchor_mask(images[0]))
    correction = np.zeros(images[1].shape, dtype=bool)
    correction[6:10, 6:10, 6:10] = True
    session.apply_correction("t01", correction)
    other_id = next(
        c.candidate_id
        for c in series.timepoint("t01").multi.candidates
        if c.candidate_id != series.timepoint("t01").multi.selected.candidate_id
    )
    preview = session.preview_candidate_selection("t01", other_id)
    assert "t01" not in preview["auto_to_update"]  # manual there, not auto
    assert "t01" in preview["manual_conflicts"]
    # Keeping manual results only recomputes automatic ones.
    impact = session.apply_candidate_selection("t01", other_id, keep_manual=True)
    assert impact == preview
    assert session.region("t01").kind == "manual"
    assert np.array_equal(session.region("t01").mask, correction)
    # Dropping conflicts removes the correction but keeps the anchor.
    session.apply_candidate_selection(
        "t01",
        series.timepoint("t01").multi.selected.candidate_id,
        keep_manual=False,
    )
    assert "t01" not in session.corrections
    assert np.array_equal(session.anchor_mask, _anchor_mask(images[0]))


def test_change_curve_distinguishes_auto_and_manual():
    images, _, session = _session(3)
    correction = np.zeros(images[2].shape, dtype=bool)
    correction[7:12, 7:12, 7:12] = True
    session.apply_correction("t02", correction)
    kinds = {entry["label"]: entry["kind"] for entry in session.change_curve()}
    assert kinds == {"t00": "source", "t01": "auto", "t02": "manual"}


def test_save_reload_roundtrip_and_verify(tmp_path):
    images, series, session = _session(4, uninformative_index=2)
    correction = np.zeros(images[1].shape, dtype=bool)
    correction[8:13, 9:14, 8:13] = True
    session.apply_correction("t01", correction, use_as_basis=True)
    save_propagation(session, tmp_path / "v1")

    reloaded = load_propagation(tmp_path / "v1", images, series)
    for label in session.labels:
        before, after = session.region(label), reloaded.region(label)
        assert before.kind == after.kind
        if before.mask is not None:
            assert np.array_equal(before.mask, after.mask)
    assert reloaded.basis_labels == {"t01"}
    outcome = verify_propagation_version(tmp_path / "v1", images, series)
    assert outcome["verified"]

    # Pending timepoints must not have a mask file on disk.
    report = json.loads((tmp_path / "v1" / "propagation_report.json").read_text())
    pending = [r for r in report["regions"] if r["kind"] == "pending"]
    assert pending and all(r["mask_path"] is None for r in pending)

    # Tampering with a saved region mask is detected.
    from medreg.image import MedicalImage

    mask_file = tmp_path / "v1" / "regions" / "t03_mask.nii.gz"
    tampered = MedicalImage.load_nifti(mask_file)
    tampered.data[0, 0, 0] = 1.0 - tampered.data[0, 0, 0]
    tampered.save_nifti(mask_file)
    with pytest.raises(ValueError, match="t03"):
        verify_propagation_version(tmp_path / "v1", images, series)


def test_reload_rejects_mismatched_series_decision(tmp_path):
    images, series, session = _session(3)
    save_propagation(session, tmp_path / "v1")
    per_index = {1: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    other_series = register_series(images, config=_config(), initial_sets_per_timepoint=per_index)
    other_id = next(
        c.candidate_id
        for c in other_series.timepoint("t01").multi.candidates
        if c.candidate_id != series.timepoint("t01").multi.selected.candidate_id
    )
    from medreg.series import select_series_candidate

    changed = select_series_candidate(other_series, "t01", other_id)
    with pytest.raises(ValueError, match="adopted candidate"):
        load_propagation(tmp_path / "v1", images, changed)
