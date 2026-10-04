import json

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from medreg.propagation import PropagationSession
from medreg.propagation_batch import (
    load_propagation,
    load_multi_region_propagation,
    save_propagation,
    save_multi_region_propagation,
    verify_multi_region_propagation_version,
    verify_propagation_version,
)
from medreg.multi_region import MultiRegionPropagationSession
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


def test_multi_region_relationships_manual_conflicts_and_roundtrip(tmp_path):
    images, _ = _series(2)
    per_index = {1: [[0.0] * 6, [0.05, -0.04, 0.03, 2.0, -1.5, 1.0]]}
    series = register_series(images, config=_config(), initial_sets_per_timepoint=per_index)
    multi = MultiRegionPropagationSession(images, series)

    large = np.zeros(images[0].shape, dtype=bool)
    large[8:18, 10:20, 8:18] = True
    detached = np.zeros(images[0].shape, dtype=bool)
    detached[22:25, 22:25, 20:23] = True
    contact_left = np.zeros(images[0].shape, dtype=bool)
    contact_right = np.zeros(images[0].shape, dtype=bool)
    contact_left[5:8, 5:8, 5:8] = True
    contact_right[8:10, 5:8, 5:8] = True

    multi.add_region("large", "t00", large)
    propagated_large = multi.region_session("large").region("t01").mask
    assert propagated_large is not None and propagated_large.any()
    small = binary_erosion(propagated_large)
    assert small.any()
    multi.add_region("small", "t01", small)
    multi.add_region("detached", "t00", detached)
    multi.add_region("contact_left", "t00", contact_left)
    multi.add_region("contact_right", "t00", contact_right)
    multi.define_relationship("large", "small", "contains", declared_at="t01")
    multi.define_relationship("large", "detached", "separate", declared_at="t00")
    multi.define_relationship("contact_left", "contact_right", "contact", declared_at="t00")

    reviews = {item["relation_id"]: item for item in multi.review_relationships()}
    by_label = {
        relation_id: {item["label"]: item for item in item["observations"]}
        for relation_id, item in reviews.items()
    }
    assert by_label["large--contains--small"]["t01"]["actual"] == "contains"
    assert by_label["large--separate--detached"]["t00"]["actual"] == "separate"
    assert by_label["contact_left--contact--contact_right"]["t00"]["actual"] == "contact"
    assert by_label["large--separate--detached"]["t00"]["regions"]["first"]["anchor_label"] == "t00"

    oversized = multi.region_session("small").region("t01").mask.copy()
    assert oversized is not None
    oversized[24:27, 26:29, 23:26] = True
    impact = multi.apply_region_correction("small", "t01", oversized)
    changed = next(item for item in impact["affected_relationships"] if "contains" in item["relation_id"])
    assert changed["after"]["status"] == "conflict"
    assert changed["after"]["metrics"]["second_outside_first_voxels"] > 0
    assert np.array_equal(multi.region_session("small").region("t01").mask, oversized)
    assert np.array_equal(multi.region_session("large").anchor_mask, large)

    withdrawn = multi.withdraw_region_correction("small", "t01")
    assert withdrawn["anchor_unchanged"]
    assert withdrawn["other_regions_unchanged"]["large"]
    assert multi.region_session("small").region("t01").kind == "source"
    assert np.array_equal(multi.region_session("small").anchor_mask, small)

    multi.apply_region_correction("small", "t01", oversized)
    other_id = next(
        candidate.candidate_id
        for candidate in series.timepoint("t01").multi.candidates
        if candidate.candidate_id != series.timepoint("t01").multi.selected.candidate_id
    )
    candidate_impact = multi.apply_candidate_selection(
        "t01", other_id, region_id="small"
    )
    reviewed_relations = {item["relation_id"] for item in candidate_impact["review"]}
    assert "large--contains--small" in reviewed_relations
    assert "large--separate--detached" not in reviewed_relations
    assert np.array_equal(multi.region_session("small").region("t01").mask, oversized)

    export = multi.export_relationships()
    relation = next(
        item for item in export["relationship_review"] if item["relation_id"] == "large--contains--small"
    )
    assert relation["source"] == "manual"
    assert all(
        observation["regions"]["second"]["candidate_id"] is not None
        or observation["regions"]["second"]["anchor_label"] == "t01"
        for observation in relation["observations"]
    )

    save_multi_region_propagation(multi, tmp_path / "multi")
    reloaded = load_multi_region_propagation(tmp_path / "multi", images, multi.series)
    assert np.array_equal(reloaded.region_session("small").region("t01").mask, oversized)
    reloaded_review = {item["relation_id"]: item for item in reloaded.review_relationships()}
    assert reloaded_review["large--contains--small"]["conflict_count"] >= 1
    outcome = verify_multi_region_propagation_version(tmp_path / "multi", images, multi.series)
    assert outcome["verified"]
    assert "large--contains--small" in outcome["conflicts"]
