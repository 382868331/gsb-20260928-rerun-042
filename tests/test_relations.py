
import hashlib
import json

import numpy as np
import pytest

from medreg.relations import MultiRegionSession, classify_pair
from medreg.relations_batch import (
    load_relations,
    save_relations,
    verify_relations_version,
)
from medreg.relations_compare import (
    compare_relation_reviews,
    export_relation_comparison,
    verify_relation_comparison,
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


def test_frozen_review_comparison_groups_and_manual_origin(tmp_path):
    images, series, session_v1 = _multi_session(4, uninformative_index=2)
    # A second declaration present only in v1 (withdrawn later) and one
    # present only in v2 (added later).
    session_v1.expect("large", "small", "contact")
    save_relations(session_v1, tmp_path / "v1")

    _, _, session_v2 = _multi_session(4, uninformative_index=2)
    session_v2.expect("large", "small", "separate")
    correction = session_v2.region_session("small").region("t03").mask.copy()
    block = np.zeros(images[3].shape, dtype=bool)
    block[0:3, 0:3, 0:3] = True  # manual block beyond the container
    correction |= block
    session_v2.apply_correction("small", "t03", correction, use_as_basis=True)
    save_relations(session_v2, tmp_path / "v2")

    v1_hash = hashlib.sha256(
        (tmp_path / "v1" / "multi_region_report.json").read_bytes()
    ).hexdigest()
    comparison = export_relation_comparison(
        tmp_path / "v1", tmp_path / "v2", tmp_path / "cmp"
    )

    # The frozen reviews themselves are never modified.
    assert hashlib.sha256(
        (tmp_path / "v1" / "multi_region_report.json").read_bytes()
    ).hexdigest() == v1_hash
    assert comparison["read_only"] is True
    assert comparison["counts"] == {
        "unchanged": 3,  # contains at t00/t01 and pending at t02
        "added": 4,      # separate declared only in v2 (one per label)
        "withdrawn": 4,  # contact declared only in v1 (one per label)
        "conclusion_changed": 1,  # contains at t03, manual block
    }

    changed = comparison["conclusion_changed"]
    assert len(changed) == 1
    entry = changed[0]
    assert entry["identity"] == ["large", "small", "contains", "t03"]
    assert entry["change_origins"] == ["manual_correction"]
    assert entry["before"]["status"] == "satisfied"
    assert entry["after"]["status"] == "conflict"
    # Both sides keep timepoint, sources and beyond-voxel evidence.
    assert entry["left"]["label"] == "t03"
    assert entry["right"]["label"] == "t03"
    assert entry["left"]["offending_voxels"] == 0
    assert entry["right"]["offending_voxels"] >= int(block.sum())
    assert entry["right"]["basis"]["small"]["kind"] == "manual"
    assert entry["right"]["sources"][1]["kind"] == "manual"
    assert entry["left"]["basis"]["large"]["candidate_id"]
    assert comparison["left"]["created_utc"]
    assert comparison["right"]["created_utc"]
    assert comparison["left"]["frozen_report"]["sha256"] == v1_hash

    # Pending evaluations stay paired and pending on both sides.
    pending = next(
        e for e in comparison["unchanged"] if e["label"] == "t02"
    )
    assert pending["left"]["status"] == "pending"
    assert pending["right"]["status"] == "pending"
    assert pending["left"]["actual"] is None

    # Declared-only relations are attributed to declaration updates and
    # cover separate / contact as well as contains.
    assert {
        (e["relation"], e["kind"]) for e in comparison["added"]
    } == {("separate", "added")}
    assert {
        (e["relation"], e["kind"]) for e in comparison["withdrawn"]
    } == {("contact", "withdrawn")}
    assert all(e["change_origins"] == ["declaration_update"]
               for e in comparison["added"] + comparison["withdrawn"])

    # Reopening reproduces the exact same comparison.
    again = compare_relation_reviews(tmp_path / "v1", tmp_path / "v2")
    for field in (
        "counts", "entries", "expectations_left", "expectations_right",
    ):
        assert again[field] == comparison[field]
    outcome = verify_relation_comparison(tmp_path / "cmp")
    assert outcome["verified"] is True
    assert outcome["counts"] == comparison["counts"]


def test_frozen_review_comparison_flags_candidate_switch(tmp_path):
    images, _ = _series(3)
    # Two far-apart candidates at t02; the organ anchor sits exactly at
    # t02, so switching t02's candidate moves only the propagated lesion.
    per_index = {2: [[0.0] * 6, [0.3, -0.3, 0.2, 12.0, 10.0, 8.0]]}
    series = register_series(
        images, config=_config(), initial_sets_per_timepoint=per_index
    )
    v1 = MultiRegionSession(images, series)
    v1.add_region("organ", "t02", _large_mask(images[2]))
    organ_at_t00 = v1.region_session("organ").region("t00").mask
    lesion_t00 = _inner_box(organ_at_t00, margin=3)
    v1.add_region("lesion", "t00", lesion_t00)
    v1.expect("organ", "lesion", "contains")
    assert _evaluation(v1, "t02")["status"] == "satisfied"
    save_relations(v1, tmp_path / "v1")

    other_id = next(
        c.candidate_id
        for c in series.timepoint("t02").multi.candidates
        if c.candidate_id != series.timepoint("t02").multi.selected.candidate_id
    )
    v1.apply_candidate_selection("t02", other_id, keep_manual=True)
    assert _evaluation(v1, "t02")["status"] == "conflict"
    save_relations(v1, tmp_path / "v2")

    comparison = export_relation_comparison(
        tmp_path / "v1", tmp_path / "v2", tmp_path / "cmp"
    )
    changed = comparison["conclusion_changed"]
    # The organ is anchored at t02, so its propagated masks at t00/t01
    # and the lesion propagation all trace through t02's new candidate.
    assert [e["label"] for e in changed] == ["t00", "t01", "t02"]
    assert all(e["change_origins"] == ["candidate_switch"] for e in changed)
    # The organ source at t02 never moves; the lesion follows a new candidate.
    at_t02 = next(e for e in changed if e["label"] == "t02")
    assert at_t02["left"]["basis"]["organ"]["kind"] == "source"
    assert at_t02["right"]["basis"]["organ"]["kind"] == "source"
    assert (
        at_t02["left"]["basis"]["lesion"]["candidate_id"]
        != at_t02["right"]["basis"]["lesion"]["candidate_id"]
    )
    assert comparison["counts"]["unchanged"] == 0
    assert verify_relation_comparison(tmp_path / "cmp")["verified"] is True
