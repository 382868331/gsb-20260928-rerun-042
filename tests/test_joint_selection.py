
import numpy as np
import pytest

from medreg.joint_selection import JointSelectionPlanner
from medreg.propagation import PropagationSession
from medreg.relations import MultiRegionSession
from medreg.series import SeriesConfig, register_series
from medreg.synthetic import synthetic_series
from medreg.transform import RigidTransform


def _series(n=3, **kwargs):
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


def _multi_session(n=3, **kwargs):
    images, _ = _series(n, **kwargs)
    series = register_series(images, config=_config())
    session = MultiRegionSession(images, series)
    session.add_region("large", "t00", _large_mask(images[0]))
    large_at_t01 = session.region_session("large").region("t01").mask
    session.add_region("small", "t01", _inner_box(large_at_t01))
    session.expect("large", "small", "contains")
    return images, series, session


def _plan_by_choice(preview, region, option_id):
    for plan in preview.plans:
        if plan["choices"].get(region) == option_id:
            return plan
    raise AssertionError(f"no plan with {region}={option_id}")


def test_joint_search_finds_compatible_manual_block():
    images, _, session = _multi_session(3)
    small = session.region_session("small")
    # Make the current state a conflict: a manual block outside the container.
    conflicting = small.region("t02").mask.copy()
    block = np.zeros(images[2].shape, dtype=bool)
    block[0:3, 0:3, 0:3] = True
    conflicting |= block
    session.apply_correction("small", "t02", conflicting)
    assert session.conflicts(), "current state should violate contains"

    compatible = _inner_box(session.region_session("large").region("t02").mask)
    planner = JointSelectionPlanner(session)
    before = small.region("t02").mask.copy()
    preview = planner.plan("t02", manual_options={"small": [compatible]})

    # Planning is read-only: nothing is written to any mask.
    assert np.array_equal(small.region("t02").mask, before)

    # The joint evaluation uses actual voxels, not merged pass labels:
    # keeping the conflicting block fails, the compatible block passes.
    keep = _plan_by_choice(preview, "small", "current")
    assert not keep["all_satisfied"]
    conflict_record = next(
        r for r in keep["relations"] if r["relation"] == "contains"
    )
    assert conflict_record["status"] == "conflict"
    assert conflict_record["offending_voxels"] > 0
    # The concrete blocking range of the offending voxels is preserved.
    assert conflict_record["offending_bounds"]["index_min"] == [0, 0, 0]
    assert conflict_record["offending_bounds"]["index_max"] == [2, 2, 2]

    fix = _plan_by_choice(preview, "small", "manual:0")
    assert fix["all_satisfied"]
    assert fix["changed_voxels"] > 0

    # The two objectives rank differently: fewest changes keeps the
    # conflict, prioritising relations picks the compatible block.
    report = preview.report()
    top_min = report["best_by_objective"]["min_changed_voxels"]
    top_rel = report["best_by_objective"]["prioritize_relations"]
    assert preview.plans[top_min]["choices"]["small"] == "current"
    # Prioritising relations picks a satisfying combination over the
    # zero-change but conflicting current state.
    assert preview.plans[top_rel]["all_satisfied"]
    assert preview.plans[top_rel]["choices"]["small"] != "current"

    # Confirm the plan that applies the compatible manual block.
    outcome = planner.confirm(preview, fix["plan_id"])
    assert outcome["applied"]["small"] == "manual:0"
    assert outcome["applied_manual_regions"] == ["small"]
    assert outcome["conflicts"] == []
    satisfied = [r for r in outcome["relations"] if r["status"] == "satisfied"]
    assert len(satisfied) == len(outcome["relations"])
    # The adopted source is traceable: small is now a manual correction.
    assert session.region_session("small").region("t02").kind == "manual"


def test_confirm_refuses_stale_preview_after_input_change():
    images, _, session = _multi_session(3)
    compatible = _inner_box(session.region_session("large").region("t02").mask)
    planner = JointSelectionPlanner(session)
    preview = planner.plan("t02", manual_options={"small": [compatible]})
    plan_id = preview.report()["best_by_objective"]["prioritize_relations"]

    # Any input change invalidates the preview: a new correction elsewhere.
    other = session.region_session("large").region("t02").mask.copy()
    session.apply_correction("large", "t02", other)
    with pytest.raises(ValueError, match="inputs changed"):
        planner.confirm(preview, plan_id)

    # A fresh preview of the updated inputs confirms fine.
    fresh = planner.plan("t02", manual_options={"small": [compatible]})
    outcome = planner.confirm(
        fresh, fresh.report()["best_by_objective"]["prioritize_relations"]
    )
    assert outcome["label"] == "t02"


def test_shared_boundary_contact_verified_in_joint_plan():
    images, _ = _series(2)
    series = register_series(images, config=_config())
    session = MultiRegionSession(images, series)
    core = np.zeros(images[0].shape, dtype=bool)
    core[8:14, 8:14, 8:14] = True
    session.add_region("core", "t00", core)
    shell = np.zeros(images[0].shape, dtype=bool)
    shell[14:17, 8:14, 8:14] = True  # face-adjacent to core
    session.add_region("shell", "t00", shell)
    session.expect("core", "shell", "contact")

    far = np.zeros(images[0].shape, dtype=bool)
    far[20:23, 20:23, 20:23] = True
    planner = JointSelectionPlanner(session)
    preview = planner.plan("t00", manual_options={"shell": [far]})

    touching = _plan_by_choice(preview, "shell", "current")
    record = touching["relations"][0]
    assert touching["all_satisfied"]
    assert record["actual"] == "contact"
    # The shared boundary is quantified for the final review.
    assert record["shared_boundary_voxels"] == 6 * 6

    separated = _plan_by_choice(preview, "shell", "manual:0")
    assert not separated["all_satisfied"]
    assert separated["relations"][0]["actual"] == "separate"


def test_three_way_relation_conflict_keeps_best_effort_plan():
    images, _ = _series(2)
    series = register_series(images, config=_config())
    session = MultiRegionSession(images, series)
    region_a = np.zeros(images[0].shape, dtype=bool)
    region_a[2:10, 2:10, 2:10] = True
    session.add_region("a", "t00", region_a)
    region_b = np.zeros(images[0].shape, dtype=bool)
    region_b[8:10, 4:6, 4:6] = True  # inside a, touching its boundary
    session.add_region("b", "t00", region_b)
    placeholder = np.zeros(images[0].shape, dtype=bool)
    placeholder[20:22, 20:22, 20:22] = True
    session.add_region("c", "t00", placeholder)
    session.expect("a", "b", "contains")
    session.expect("a", "c", "contact")
    session.expect("b", "c", "separate")

    # c1 touches both a and b: contact(a,c) holds, separate(b,c) fails.
    c1 = np.zeros(images[0].shape, dtype=bool)
    c1[10:12, 4:6, 4:6] = True
    # c2 is far away: separate(b,c) holds, contact(a,c) fails.
    c2 = np.zeros(images[0].shape, dtype=bool)
    c2[16:18, 16:18, 16:18] = True
    planner = JointSelectionPlanner(session)
    preview = planner.plan("t00", manual_options={"c": [c1, c2]})

    # No combination satisfies all three relations simultaneously.
    assert all(not plan["all_satisfied"] for plan in preview.plans)
    best = preview.report()["best_by_objective"]["prioritize_relations"]
    best_plan = preview.plans[best]
    assert best_plan["satisfied_relations"] == 2
    assert best_plan["total_relations"] == 3
    conflicts = [r for r in best_plan["relations"] if r["status"] == "conflict"]
    assert len(conflicts) == 1
    # The conflicting option still reports its shared boundary with b.
    if best_plan["choices"]["c"] == "manual:0":
        assert conflicts[0]["relation"] == "separate"
        assert conflicts[0]["actual"] == "contact"
        assert conflicts[0]["shared_boundary_voxels"] > 0
    else:
        assert conflicts[0]["relation"] == "contact"
        assert conflicts[0]["actual"] == "separate"


def test_blocked_options_keep_ranges_and_other_regions_still_search():
    images, _, session = _multi_session(3, uninformative_index=2)
    assert session.series.timepoint("t02").status == "failed"
    planner = JointSelectionPlanner(session)
    block = np.zeros(images[2].shape, dtype=bool)
    block[8:14, 10:16, 8:14] = True
    preview = planner.plan("t02", manual_options={"small": [block]})

    # large has no usable option at the failed timepoint: current is
    # pending and every candidate failed acceptance. Its blocking details
    # are kept, but small still takes part in the search.
    assert "large" in preview.excluded_regions
    large_options = {opt.option_id: opt for opt in preview.options["large"]}
    assert large_options["current"].blocked
    assert any(
        opt.blocked and opt.kind == "candidate"
        for opt in preview.options["large"]
    )
    assert preview.plans, "independent regions must still be searchable"
    assert all("small" in plan["choices"] for plan in preview.plans)
    # Relations involving the excluded region stay explicitly pending.
    pending = [
        r
        for r in preview.plans[0]["relations"]
        if r["status"] == "pending"
    ]
    assert pending and "large" in pending[0]["regions"]

    # A manual block that does not lie on the image grid is blocked too.
    bad_shape = np.zeros((4, 4, 4), dtype=bool)
    bad_shape[:] = True
    preview2 = planner.plan("t02", manual_options={"small": [bad_shape]})
    manual_opt = next(
        opt for opt in preview2.options["small"] if opt.option_id == "manual:0"
    )
    assert manual_opt.blocked
    assert "shape" in manual_opt.blocking["reason"]


def test_out_of_bounds_candidate_keeps_concrete_blocking_range():
    images, series, _ = _multi_session(3)
    session = PropagationSession(
        images, series, "t00", _large_mask(images[0])
    )
    # A transform pushing the region far outside the target field of view.
    far_away = RigidTransform(np.eye(3), np.asarray([500.0, 0.0, 0.0]))
    region = session._propagate_with(
        "t00", "t01", RigidTransform(), far_away, "candidate_x", "test"
    )
    assert region.kind == "pending"
    assert region.mask is None
    assert any("region_coverage_insufficient" in r for r in region.reasons)
    assert region.out_of_fov is not None
    assert region.out_of_fov["voxels"] == int(_large_mask(images[0]).sum())
    # The blocking range is a concrete index-space bounding box.
    assert (
        region.out_of_fov["index_max"][0] < 0
        or region.out_of_fov["index_min"][0] >= images[1].shape[0]
    )
    assert region.out_of_fov["index_max"][0] > region.out_of_fov["index_min"][0]


def test_confirm_adopts_candidate_choice_with_provenance():
    images, _ = _series(
        2,
    )
    per_index = {1: [[0.0] * 6, [0.05, -0.04, 0.03, 2.0, -1.5, 1.0]]}
    series = register_series(
        images, config=_config(), initial_sets_per_timepoint=per_index
    )
    session = MultiRegionSession(images, series)
    session.add_region("large", "t00", _large_mask(images[0]))
    large_at_t01 = session.region_session("large").region("t01").mask
    session.add_region("small", "t01", _inner_box(large_at_t01))
    session.expect("large", "small", "contains")

    planner = JointSelectionPlanner(session)
    preview = planner.plan("t01")
    candidate_options = [
        opt for opt in preview.options["large"] if opt.kind == "candidate"
    ]
    assert candidate_options, "candidate options must be offered"
    chosen = next(opt for opt in candidate_options if not opt.blocked)
    plan = _plan_by_choice(preview, "large", chosen.option_id)

    outcome = planner.confirm(preview, plan["plan_id"])
    assert outcome["adopted_candidates"]["large"] == chosen.candidate_id
    # The adoption decision is visible in the series provenance.
    multi = session.series.timepoint("t01").multi
    assert multi.selected.candidate_id == chosen.candidate_id
    assert multi.selection_method == f"manual:{chosen.candidate_id}"
    # All region sessions share the same final series state.
    for name in session.region_names:
        assert session.region_session(name).series is session.series
