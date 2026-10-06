
import json

import numpy as np
import pytest

from medreg.relations import MultiRegionSession, classify_pair
from medreg.relations_batch import (
    load_relations,
    save_relations,
    verify_relations_version,
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


def _large_mask(image):
    mask = np.zeros(image.shape, dtype=bool)
    mask[6:22, 6:24, 6:20] = True
    return mask


def _inner_box(mask, margin=3):
    coords = np.column_stack(np.where(mask))
    lo = coords.min(axis=0) + margin
    hi = coords.max(axis=0) - margin
    box = np.zeros(mask.shape, dtype=bool)
    box[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return box


def _multi_session(n=4, **kwargs):
    """Large region anchored at t00, small region anchored at t01."""
    images, _ = _series(n, **kwargs)
    series = register_series(images, config=_config())
    session = MultiRegionSession(images, series)
    session.add_region("large", "t00", _large_mask(images[0]))
    large_at_t01 = session.region_session("large").region("t01").mask
    session.add_region("small", "t01", _inner_box(large_at_t01))
    session.expect("large", "small", "contains")
    return images, series, session


def _evaluation(session, label):
    return next(
        record
        for record in session.evaluate_relations()
        if record["label"] == label and record["relation"] == "contains"
    )


def test_classify_pair_contains_contact_separate():
    large = np.zeros((12, 12, 12), dtype=bool)
    large[2:10, 2:10, 2:10] = True
    inside = np.zeros((12, 12, 12), dtype=bool)
    inside[4:6, 4:6, 4:6] = True
    assert classify_pair(large, inside) == ("contains", 0)
    assert classify_pair(inside, large)[0] == "contained_by"

    touching = np.zeros((12, 12, 12), dtype=bool)
    touching[10:12, 4:6, 4:6] = True  # face-adjacent to the large block
    assert classify_pair(large, touching)[0] == "contact"

    far = np.zeros((12, 12, 12), dtype=bool)
    far[0, 0, 0] = False
    distant = np.zeros((14, 14, 14), dtype=bool)
    big = np.zeros((14, 14, 14), dtype=bool)
    big[2:6, 2:6, 2:6] = True
    distant[10:12, 10:12, 10:12] = True
    assert classify_pair(big, distant)[0] == "separate"

    partial = np.zeros((12, 12, 12), dtype=bool)
    partial[8:11, 4:6, 4:6] = True
    actual, offending = classify_pair(large, partial)
    assert actual == "overlap"
    assert offending > 0


def test_contains_relation_satisfied_with_traceable_basis():
    _, _, session = _multi_session(3)
    evaluations = session.evaluate_relations()
    assert len(evaluations) == 3  # one expectation x three timepoints
    for record in evaluations:
        assert record["status"] == "satisfied", record
        assert record["actual"] == "contains"
        basis = record["basis"]
        assert basis["large"]["kind"] in {"source", "auto"}
        assert basis["small"]["kind"] in {"source", "auto"}
    # Every evaluation traces back to the adoption decision behind it.
    at_t02 = _evaluation(session, "t02")
    assert at_t02["basis"]["large"]["candidate_id"]
    assert at_t02["basis"]["large"]["transform_parameters"]
    assert at_t02["basis"]["small"]["candidate_id"]
    # Anchors sit at two different timepoints.
    assert _evaluation(session, "t00")["basis"]["large"]["kind"] == "source"
    assert _evaluation(session, "t01")["basis"]["small"]["kind"] == "source"
    assert session.conflicts() == []


def test_manual_block_outside_container_kept_as_conflict():
    images, _, session = _multi_session(3)
    small_session = session.region_session("small")
    correction = small_session.region("t02").mask.copy()
    block = np.zeros(images[2].shape, dtype=bool)
    block[0:3, 0:3, 0:3] = True  # outside the large container
    correction |= block
    impact = session.apply_correction("small", "t02", correction)
    # The manual block is stored verbatim, never clipped to fit the container.
    kept = small_session.region("t02").mask
    assert np.array_equal(kept, correction)
    assert kept[0:3, 0:3, 0:3].all()
    # The conflict is preserved for review with its provenance.
    conflicts = [c for c in impact["conflicts"] if c["label"] == "t02"]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["status"] == "conflict"
    assert conflict["manual_origin"]
    assert conflict["offending_voxels"] >= int(block.sum())
    assert conflict["basis"]["small"]["kind"] == "manual"
    # Impact comparison points at exactly this relation and timepoint.
    assert [
        (change["label"], change["after"]["status"])
        for change in impact["relation_changes"]
    ] == [("t02", "conflict")]
    # Other timepoints are untouched.
    assert _evaluation(session, "t01")["status"] == "satisfied"


def test_withdraw_small_correction_leaves_large_and_anchor_untouched():
    images, _, session = _multi_session(3)
    large_before = {
        label: session.region_session("large").region(label).mask.copy()
        for label in session.labels
    }
    auto_small_t02 = session.region_session("small").region("t02").mask.copy()
    correction = auto_small_t02.copy()
    correction[0:3, 0:3, 0:3] = True
    session.apply_correction("small", "t02", correction)
    assert session.conflicts()

    impact = session.withdraw_correction("small", "t02")
    # Small region falls back to the automatic result; conflict resolved.
    assert np.array_equal(
        session.region_session("small").region("t02").mask, auto_small_t02
    )
    assert session.conflicts() == []
    assert [
        (change["label"], change["after"]["status"])
        for change in impact["relation_changes"]
    ] == [("t02", "satisfied")]
    # The large region and both original anchors are unchanged.
    for label in session.labels:
        assert np.array_equal(
            session.region_session("large").region(label).mask, large_before[label]
        )
    assert np.array_equal(
        session.region_session("large").anchor_mask, _large_mask(images[0])
    )
    assert "t02" not in session.region_session("small").corrections


def test_candidate_switch_reviews_only_related_relations():
    images, _ = _series(4)
    # t02 is an anchor for neither region, so switching its candidate
    # only re-reviews the relations at t02 itself.
    per_index = {2: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    series = register_series(
        images, config=_config(), initial_sets_per_timepoint=per_index
    )
    session = MultiRegionSession(images, series)
    session.add_region("large", "t00", _large_mask(images[0]))
    large_at_t01 = session.region_session("large").region("t01").mask
    session.add_region("small", "t01", _inner_box(large_at_t01))
    session.expect("large", "small", "contains")
    other_id = next(
        c.candidate_id
        for c in series.timepoint("t02").multi.candidates
        if c.candidate_id != series.timepoint("t02").multi.selected.candidate_id
    )
    before = {record["label"]: record for record in session.evaluate_relations()}
    impact = session.apply_candidate_selection("t02", other_id, keep_manual=True)
    reviewed_labels = {key[3] for key in impact["relations_reviewed"]}
    assert reviewed_labels == {"t02"}
    # Only the affected timepoint is re-reviewed; the rest are re-asserted
    # unchanged rather than recomputed into the report.
    after = {record["label"]: record for record in session.evaluate_relations()}
    for label in ("t00", "t01", "t03"):
        assert after[label]["status"] == before[label]["status"]
        assert after[label]["actual"] == before[label]["actual"]
    assert all(change["label"] == "t02" for change in impact["relation_changes"])


def test_joint_selection_uses_actual_masks_and_confirms_only_after_recheck(tmp_path):
    images, _ = _series(3)
    per_index = {2: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    series = register_series(images, config=_config(), initial_sets_per_timepoint=per_index)
    session = MultiRegionSession(images, series)
    session.add_region("large", "t00", _large_mask(images[0]))
    propagated_large = session.region_session("large").region("t01").mask
    session.add_region("small", "t01", _inner_box(propagated_large))
    session.expect("large", "small", "contains")
    original_large = session.region_session("large").region("t02").mask.copy()
    original_small = session.region_session("small").region("t02").mask.copy()

    preview = session.preview_joint_selection("t02")
    # Previewing never changes adopted masks or the shared series.
    assert np.array_equal(session.region_session("large").region("t02").mask, original_large)
    assert np.array_equal(session.region_session("small").region("t02").mask, original_small)
    assert preview["compatible_solutions"]
    recommended = preview["recommended"]
    assert recommended["compatible"]
    assert recommended["satisfied_relations"] >= 1
    assert all(
        record["spatial_evidence"]["outside_voxels"] == 0
        for record in recommended["evaluations"]
        if record["status"] == "satisfied"
    )

    stale = dict(preview)
    stale["input_digest"] = "stale"
    with pytest.raises(ValueError, match="inputs changed"):
        session.confirm_joint_selection(stale)

    impact = session.confirm_joint_selection(preview)
    assert impact["confirmed"]
    final_record = impact["final_evaluations"][0]
    assert final_record["status"] == "satisfied"
    assert final_record["actual"] == "contains"
    assert final_record["spatial_evidence"]["outside_voxels"] == 0

    save_relations(session, tmp_path / "joint")
    reloaded = load_relations(tmp_path / "joint", images, series)
    chosen_candidate = next(iter(impact["selections"]["large"].values()))["candidate_id"]
    assert reloaded.region_session("large").region("t02").candidate_id == chosen_candidate


def test_joint_selection_keeps_independent_region_when_one_choice_is_blocked():
    images, _ = _series(3, uninformative_index=2)
    per_index = {2: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    series = register_series(images, config=_config(), initial_sets_per_timepoint=per_index)
    session = MultiRegionSession(images, series)
    far = np.zeros(images[0].shape, dtype=bool)
    far[1:4, 1:4, 1:4] = True
    session.add_region("large", "t00", _large_mask(images[0]))
    session.add_region("far", "t00", far)
    session.expect("large", "far", "separate")

    preview = session.preview_joint_selection("t02")
    assert preview["all_solutions"]
    blocked = [
        item
        for item in preview["blocked"]
        if any(
            detail["region"] == "far" for detail in item["blocked_ranges"]
        )
    ]
    assert blocked
    detail = next(
        detail
        for item in blocked
        for detail in item["blocked_ranges"]
        if detail["region"] == "far"
    )
    assert detail["range"] is not None
    # The large region is independent and still appears in the search plans.
    assert any(
        "large" in plan["selections"] for plan in preview["all_solutions"]
    )
    assert preview["recommended"] is None


def test_three_way_conflict_and_manual_outside_range_are_reported():
    images, _ = _series(3)
    series = register_series(images, config=_config())
    session = MultiRegionSession(images, series)
    large = np.zeros(images[0].shape, dtype=bool)
    large[8:22, 8:22, 8:20] = True
    far = np.zeros(images[0].shape, dtype=bool)
    far[0:3, 0:3, 0:3] = True
    session.add_region("large", "t00", large)
    small = _inner_box(session.region_session("large").region("t01").mask)
    session.add_region("small", "t01", small)
    session.add_region("far", "t00", far)
    session.expect("large", "small", "contains")
    session.expect("large", "far", "separate")

    propagated_small = session.region_session("small").region("t02").mask
    conflicted = propagated_small.copy()
    far_at_t02 = session.region_session("far").region("t02").mask
    far_coords = np.column_stack(np.where(far_at_t02))
    block = np.zeros(images[2].shape, dtype=bool)
    block[tuple(int(value) for value in far_coords[0])] = True
    conflicted |= block
    session.apply_correction("small", "t02", conflicted)
    pair = next(
        record
        for record in session.evaluate_relations()
        if record["label"] == "t02" and record["regions"] == ["large", "small"]
    )
    outside = pair["spatial_evidence"]["outside_voxel_range"]
    assert outside["voxels"] >= 1
    assert tuple(outside["min"]) <= tuple(far_coords[0]) <= tuple(outside["max"])
    triples = session.three_way_conflicts()
    assert any(
        item["rule"] == "contained_region_must_remain_separate"
        and item["label"] == "t02"
        and item["regions"] == ["large", "small", "far"]
        for item in triples
    )
    sources = next(
        item["sources"]
        for item in triples
        if item["label"] == "t02" and item["regions"] == ["large", "small", "far"]
    )
    assert sources["small"]["kind"] == "manual"


def test_save_reload_preserves_conflicts_and_provenance(tmp_path):
    images, series, session = _multi_session(4)
    correction = session.region_session("small").region("t03").mask.copy()
    correction[0:3, 0:3, 0:3] = True
    session.apply_correction("small", "t03", correction, use_as_basis=True)
    assert len(session.conflicts()) == 1

    save_relations(session, tmp_path / "v1")
    outcome = verify_relations_version(tmp_path / "v1", images, series)
    assert outcome["verified"]
    assert outcome["regions"] == ["large", "small"]
    assert len(outcome["conflicts"]) == 1

    reloaded = load_relations(tmp_path / "v1", images, series)
    # Conflict and the manual block behind it survive the round trip.
    conflicts = reloaded.conflicts()
    assert len(conflicts) == 1
    assert conflicts[0]["label"] == "t03"
    assert conflicts[0]["manual_origin"]
    kept = reloaded.region_session("small").region("t03").mask
    assert kept[0:3, 0:3, 0:3].all()
    assert reloaded.region_session("small").basis_labels == {"t03"}
    # Provenance of every evaluation is intact.
    record = next(r for r in reloaded.evaluate_relations() if r["label"] == "t03")
    assert record["basis"]["small"]["kind"] == "manual"
    assert record["basis"]["large"]["candidate_id"]

    report = json.loads(
        (tmp_path / "v1" / "multi_region_report.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "medreg-multiregion-v1"
    assert report["expectations"] == [
        {"region_a": "large", "region_b": "small", "relation": "contains"}
    ]
    assert len(report["conflicts"]) == 1


def test_pending_timepoint_yields_pending_relation_not_conflict():
    _, _, session = _multi_session(4, uninformative_index=2)
    record = _evaluation(session, "t02")
    assert record["status"] == "pending"
    assert record["reasons"]
    assert session.conflicts() == []
    assert _evaluation(session, "t03")["status"] == "satisfied"


def _map_by_id(entries):
    return {
        (
            item["relation_id"]["region_a"],
            item["relation_id"]["region_b"],
            item["relation_id"]["relation"],
            item["relation_id"]["label"],
        ): item
        for item in entries
    }


def test_relation_diff_covers_all_dispositions_and_origins(tmp_path):
    from medreg.relations_batch import save_relations
    from medreg.relations_diff import export_relation_diff, verify_relation_diff

    images, _ = _series(4)

    # Two frozen reviews on independent series runs, same images.
    per_index = {3: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    series_v1 = register_series(
        images, config=_config(), initial_sets_per_timepoint=per_index
    )
    series_v2 = register_series(
        images, config=_config(), initial_sets_per_timepoint=per_index
    )

    def build(series):
        session = MultiRegionSession(images, series)
        session.add_region("large", "t00", _large_mask(images[0]))
        large_at_t01 = session.region_session("large").region("t01").mask
        session.add_region("small", "t01", _inner_box(large_at_t01))
        far = np.zeros(images[0].shape, dtype=bool)
        far[0:3, 0:3, 0:3] = True
        session.add_region("far", "t00", far)
        session.expect("large", "small", "contains")
        session.expect("large", "far", "separate")
        return session

    v1 = build(series_v1)
    v2 = build(series_v2)

    # Frozen v1: an extra declaration that v2 withdraws.
    touch = np.zeros(images[0].shape, dtype=bool)
    touch[22:24, 6:24, 6:20] = True  # face-adjacent to the large block
    v1.add_region("touch", "t00", touch)
    v1.expect("large", "touch", "contact")

    # Frozen v2: human correction (out-of-container block) and a
    # propagation candidate switch at t03.
    corrected = v2.region_session("small").region("t02").mask.copy()
    corrected[0:3, 0:3, 0:3] = True
    v2.apply_correction("small", "t02", corrected)
    other_id = next(
        c.candidate_id
        for c in series_v2.timepoint("t03").multi.candidates
        if c.candidate_id != series_v2.timepoint("t03").multi.selected.candidate_id
    )
    v2.apply_candidate_selection("t03", other_id, keep_manual=True)
    series_v2 = v2.region_session("large").series

    save_relations(v1, tmp_path / "v1")
    save_relations(v2, tmp_path / "v2")
    first = export_relation_diff(tmp_path / "v1", tmp_path / "v2", tmp_path / "diff")
    second = export_relation_diff(tmp_path / "v1", tmp_path / "v2", tmp_path / "diff")
    assert first["content_digest"] == second["content_digest"]

    changed = _map_by_id(first["conclusion_changed"])
    manual_entry = changed[("large", "small", "contains", "t02")]
    assert manual_entry["origins"] == ["manual_correction"]
    assert manual_entry["conclusion_changed"]
    assert manual_entry["evidence_changed"]
    assert manual_entry["left"]["status"] == "satisfied"
    assert manual_entry["right"]["status"] == "conflict"
    assert manual_entry["right"]["basis"]["small"]["kind"] == "manual"
    assert manual_entry["right"]["offending_voxels"] >= 27

    withdrawn = _map_by_id(first["withdrawn"])
    contact_entry = withdrawn[("large", "touch", "contact", "t00")]
    assert contact_entry["origins"] == ["relation_declaration_update"]
    assert contact_entry["right"] is None
    assert contact_entry["left"]["actual"] == "contact"

    assert first["counts"]["added"] == 0
    assert first["counts"]["conclusion_changed"] >= 1

    # Both sides keep their own timepoint and source provenance.
    identity = next(
        item for item in first["unchanged"] if item["relation_id"]["label"] == "t01"
    )
    assert identity["left"]["basis"]["large"]["candidate_id"]
    assert identity["right"]["basis"]["large"]["candidate_id"]
    assert identity["left"]["label"] == identity["right"]["label"] == "t01"

    outcome = verify_relation_diff(
        tmp_path / "v1",
        tmp_path / "v2",
        tmp_path / "diff",
        images,
        series_v1,
        series_v2,
    )
    assert outcome["verified"]
    assert outcome["content_digest"] == first["content_digest"]


def test_relation_diff_detects_candidate_switch_origin(tmp_path):
    from medreg.relations_batch import save_relations
    from medreg.relations_diff import compare_relation_reviews

    images, _ = _series(4)
    per_index = {3: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}

    def freeze(switch):
        series = register_series(
            images, config=_config(), initial_sets_per_timepoint=per_index
        )
        session = MultiRegionSession(images, series)
        session.add_region("large", "t00", _large_mask(images[0]))
        large_at_t01 = session.region_session("large").region("t01").mask
        session.add_region("small", "t01", _inner_box(large_at_t01))
        session.expect("large", "small", "contains")
        if switch:
            other_id = next(
                c.candidate_id
                for c in series.timepoint("t03").multi.candidates
                if c.candidate_id != series.timepoint("t03").multi.selected.candidate_id
            )
            session.apply_candidate_selection("t03", other_id, keep_manual=True)
            series = session.region_session("large").series
        return series, session

    series_v1, v1 = freeze(False)
    series_v2, v2 = freeze(True)
    save_relations(v1, tmp_path / "a")
    save_relations(v2, tmp_path / "b")

    diff = compare_relation_reviews(tmp_path / "a", tmp_path / "b")
    t03 = next(
        item
        for bucket in (diff["conclusion_changed"], diff["unchanged"])
        for item in bucket
        if item["relation_id"]["label"] == "t03"
    )
    assert "candidate_switch" in t03["origins"]
    left_candidate = t03["left"]["basis"]["large"]["candidate_id"]
    right_candidate = t03["right"]["basis"]["large"]["candidate_id"]
    assert left_candidate != right_candidate
    assert series_v1.timepoint("t03").multi.selected.candidate_id == left_candidate
