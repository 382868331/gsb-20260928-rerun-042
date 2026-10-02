import json

import numpy as np
import pytest

from medreg.propagation import PropagationSession
from medreg.propagation import (
    MultiRegionPropagationSession,
    RegionRelation,
)
from medreg.propagation_batch import (
    load_propagation,
    load_multi_region_propagation,
    save_propagation,
    save_multi_region_propagation,
    verify_multi_region_propagation_version,
    verify_propagation_version,
)
from medreg.series import SeriesConfig, register_series
from medreg.image import MedicalImage, make_affine
from medreg.resample import resample
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


def test_multi_region_relationships_report_coverage_and_retain_conflict():
    images, series, _ = _session(3)
    outer = _anchor_mask(images[0])
    inner = np.zeros_like(outer)
    inner[12:14, 14:16, 11:13] = True
    far = np.zeros_like(outer)
    far[2:5, 2:5, 2:5] = True
    multi = MultiRegionPropagationSession(
        images,
        series,
        {
            "outer": ("t00", outer),
            "inner": ("t00", inner),
            "far": ("t00", far),
        },
        [
            RegionRelation("outer", "inner", "contains", "t00", "t02"),
            RegionRelation("outer", "far", "disjoint", "t00", "t02"),
        ],
    )
    violated_inner = np.zeros(images[1].shape, dtype=bool)
    violated_inner[17:21, 17:21, 14:18] = True
    multi.apply_correction("inner", "t01", violated_inner)

    report = multi.relationship_report()
    by_key = {
        (entry["label"], entry["region_a"], entry["region_b"]): entry
        for entry in report
    }
    assert by_key[("t00", "outer", "inner")]["status"] == "satisfied"
    assert by_key[("t00", "outer", "far")]["status"] == "satisfied"
    statuses = {(entry["label"], entry["kind"]): entry["status"] for entry in report}
    assert (
        statuses[("t01", "contains")] == "violated"
        or statuses[("t02", "contains")] == "violated"
    )
    violated = next(
        entry for entry in report if entry["kind"] == "contains" and entry["status"] == "violated"
    )
    assert violated["conflict_mask"] is not None
    assert violated["conflict_voxels"] == int(violated["conflict_mask"].sum())
    assert multi.region("inner", violated["label"]).mask.any()
    assert multi.region("outer", violated["label"]).mask.any()


def test_preview_correction_shows_associated_impacts_and_candidate_requires_choice():
    images, _ = _series(3)
    per_index = {1: [[0.0] * 6, [0.25, -0.2, 0.15, 8.0, 7.0, 5.0]]}
    series = register_series(images, config=_config(), initial_sets_per_timepoint=per_index)
    outer = _anchor_mask(images[0])
    inner = np.zeros_like(outer)
    inner[12:14, 14:16, 11:13] = True
    multi = MultiRegionPropagationSession(
        images,
        series,
        {"outer": ("t00", outer), "inner": ("t00", inner)},
        [RegionRelation("outer", "inner", "contains", "t00", "t02")],
    )
    correction = np.zeros(images[1].shape, dtype=bool)
    correction[10:18, 10:18, 8:16] = True
    preview = multi.preview_correction("outer", "t01", correction, use_as_basis=True)
    assert preview["label"] == "t01"
    assert {edge["target"] for edge in preview["downstream_propagation"]} == {"t02"}
    assert any(
        entry["label"] in {"t01", "t02"} for entry in preview["relationship_review"]
    )
    multi.apply_correction("outer", "t01", correction, use_as_basis=True)

    other_id = next(
        candidate.candidate_id
        for candidate in series.timepoint("t01").multi.candidates
        if candidate.candidate_id != series.timepoint("t01").multi.selected.candidate_id
    )
    candidate_preview = multi.preview_candidate_selection("t01", other_id)
    assert candidate_preview["manual_regions_at_label"] == ["outer"]
    with pytest.raises(ValueError, match="choose keep or replace"):
        multi.apply_candidate_selection("t01", other_id)
    impact = multi.apply_candidate_selection("t01", other_id, on_manual="keep")
    assert impact["on_manual"] == "keep"
    assert np.array_equal(multi.region("outer", "t01").mask, correction)
    assert multi.region("outer", "t01").kind == "manual"


def test_withdraw_one_region_does_not_change_other_region_or_anchor():
    images, series, _ = _session(3)
    outer = _anchor_mask(images[0])
    other = np.zeros_like(outer)
    other[4:8, 4:8, 4:8] = True
    multi = MultiRegionPropagationSession(
        images,
        series,
        {"outer": ("t00", outer), "other": ("t00", other)},
        [RegionRelation("outer", "other", "disjoint", "t00", "t02")],
    )
    outer_correction = np.zeros(images[1].shape, dtype=bool)
    outer_correction[9:15, 9:15, 8:14] = True
    other_before = multi.region("other", "t01").mask.copy()
    multi.apply_correction("outer", "t01", outer_correction)
    multi.withdraw_correction("outer", "t01")
    assert np.array_equal(multi.region("other", "t01").mask, other_before)
    assert np.array_equal(multi.region("outer", "t00").mask, outer)


def test_missing_image_relationship_stays_pending_without_crossing_gap():
    images, series, _ = _session(4, uninformative_index=2)
    outer = _anchor_mask(images[0])
    inner = np.zeros_like(outer)
    inner[12:14, 14:16, 11:13] = True
    multi = MultiRegionPropagationSession(
        images,
        series,
        {"outer": ("t00", outer), "inner": ("t00", inner)},
        [RegionRelation("outer", "inner", "contains", "t00", "t03")],
    )
    entry = next(entry for entry in multi.relationship_report("t02"))
    assert entry["status"] == "pending"
    assert entry["conflict_mask"] is None
    assert multi.region("outer", "t02").mask is None
    assert multi.region("inner", "t02").mask is None
    assert multi.region("outer", "t03").mask is not None


def test_multi_region_relationship_on_partial_overlap_field_of_view():
    images, _ = _series(1)
    fixed = images[0]
    partial_shape = (24, 26, 24)
    partial_affine = make_affine(partial_shape, (1.2, 1.0, 1.4))
    partial_data, overlap = resample(
        fixed,
        MedicalImage(np.zeros(partial_shape, dtype=np.float32), partial_affine),
        order=3,
    )
    partial_image = MedicalImage(
        np.where(overlap, partial_data, 0.0).astype(np.float32), partial_affine
    )
    partial_series = register_series([fixed, partial_image], config=_config())
    outer = np.zeros(fixed.shape, dtype=bool)
    outer[11:15, 12:17, 10:15] = True
    inner = np.zeros_like(outer)
    inner[12:14, 14:16, 11:13] = True
    multi = MultiRegionPropagationSession(
        [fixed, partial_image],
        partial_series,
        {"outer": ("t00", outer), "inner": ("t00", inner)},
        [RegionRelation("outer", "inner", "contains", "t00", "t01")],
    )
    assert multi.region("outer", "t01").coverage_fraction == 1.0
    assert multi.region("inner", "t01").coverage_fraction == 1.0
    entry = multi.relationship_report("t01")[0]
    assert entry["status"] == "satisfied"


def test_multi_region_save_reload_preserves_relationship_conflict_provenance(tmp_path):
    images, series, _ = _session(3)
    outer = _anchor_mask(images[0])
    inner = np.zeros_like(outer)
    inner[12:14, 14:16, 11:13] = True
    multi = MultiRegionPropagationSession(
        images,
        series,
        {"outer": ("t00", outer), "inner": ("t00", inner)},
        [RegionRelation("outer", "inner", "contains", "t00", "t02")],
    )
    correction = np.zeros(images[2].shape, dtype=bool)
    correction[10:16, 10:16, 8:14] = True
    multi.apply_correction("inner", "t02", correction)
    save_multi_region_propagation(multi, tmp_path / "multi")
    reloaded = load_multi_region_propagation(tmp_path / "multi", images, series)
    assert np.array_equal(reloaded.region("inner", "t02").mask, correction)
    outcome = verify_multi_region_propagation_version(tmp_path / "multi", images, series)
    assert outcome["verified"]
