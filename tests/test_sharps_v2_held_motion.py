"""The sharps-disposal v2 fold into the held-object motion bundles, CPU-only.

``planning-held-object-motion`` gained ``carry_then_turn``, an opt-in unchecked
symmetry fallback and opt-in staged/plausibility-checked engagement;
``executing-held-object-motion`` gained the plan-driven turn search, the
``hold_position`` forward and an opt-in arrival tolerance -- all from
RoboSimStudio ``sharps_disposal/gap_perception_v2``.

Two halves, as in ``test_promotion_is_behaviour_preserving.py``:

- **Parity.** Each script as it was at ``752dfbd`` and as it is now, driven over
  a small input grid with default parameters through a recording
  ``FakeContext``: the full tool-call sequence (names and keyword values), the
  return value and any raised error must match. The one declared difference is
  ``plan_linear_engagement``'s five additive, call-free plan keys, which are
  stripped from the new result (and required to be there) before comparing.
- **Behaviour.** What each new parameter or plan key does when asked for.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE = "752dfbd"
CLEARANCE = "skills/planning-held-object-motion/scripts/plan_clearance_motion.py"
LINEAR = "skills/planning-held-object-motion/scripts/plan_linear_engagement.py"
REORIENT = "skills/executing-held-object-motion/scripts/execute_reorientation.py"
ADDITIVE_PLAN_KEYS = {"aim_source", "leading_end_in_hand", "aperture", "fixture_axis", "rim_z"}


def _load_source(source: str, name: str):
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _before(rel: str, name: str):
    done = subprocess.run(["git", "show", f"{BASE}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip(f"{BASE} is not in this checkout: {done.stderr.strip()}")
    return _load_source(done.stdout, name)


def _drive(module, responses, *args, **kwargs):
    from gap.testing.fakes import FakeContext

    ctx = FakeContext(responses)
    try:
        result, error = module.run(ctx, *args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 -- the error is part of the comparison
        result, error = None, f"{type(exc).__name__}: {exc}"
    calls = [(r.tool, json.dumps(r.kwargs, sort_keys=True, default=str)) for r in ctx.calls]
    return {"calls": calls, "result": result, "error": error}


def pose(x=0.0, y=0.0, z=0.0, rotation: Rotation | None = None):
    q = (rotation or Rotation.identity()).as_quat()
    return {"position": {"x": x, "y": y, "z": z},
            "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}}


def _ok(rows=((0.0,), (1.0,))):
    return {"planned": True, "trajectory": {"waypoints": [{"positions": list(r)} for r in rows]}}


# --- parity: plan_clearance_motion --------------------------------------------------------------

CLEARANCE_GRID = list(itertools.product(
    ["clearance_first", "direct", "direct_cartesian"],
    ["shaft_into_aperture", "loop_over_shaft"],
    ["all", "none", "rolled_only"],
    [0.0, 0.03],
    [None, 1],
))


def _clearance_responses(check):
    def plan_joint(pose, **_):
        w = abs(float(pose["rotation"]["w"]))
        planned = check == "all" or (check == "rolled_only" and w < 0.99)
        return {"planned": planned, "position_error_m": 0.0, "rotation_error_rad": 0.0}

    return {"robot.get_ee_pose": {"pose": pose(0.2, 0.1, 0.5)}, "motion.plan_joint": plan_joint}


@pytest.mark.parametrize("strategy,relation,check,outward,arm", CLEARANCE_GRID)
def test_plan_clearance_motion_defaults_match_752dfbd(strategy, relation, check, outward, arm):
    before, after = _before(CLEARANCE, "clearance_before"), _load(CLEARANCE, "clearance_after")
    attachment = {"frame": "tcp", "spheres": [{"center": [0.0, 0.0, 0.08], "radius": 0.01}]}
    if arm is not None:
        attachment["arm_id"] = arm
    args = (pose(z=0.1), {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": -1.0}}, relation,
            {"meshes": []}, attachment)
    kwargs = {"strategy": strategy, "stage_outward_m": outward,
              "approach_pose": pose(0.4, 0.0, 0.45) if strategy != "clearance_first" else None}
    got_before = _drive(before, _clearance_responses(check), *args, **kwargs)
    got_after = _drive(after, _clearance_responses(check), *args, **kwargs)
    assert got_before == got_after


# --- parity: plan_linear_engagement -------------------------------------------------------------


def _camera(name):
    return {"name": name, "rgb": np.zeros((8, 8, 3), np.uint8), "depth": np.ones((8, 8), np.float32),
            "intrinsics": [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]], "pose": pose()}


def _linear_responses(marker_visible):
    n = 100
    body = np.column_stack([1e-4 * np.cos(np.arange(n)), 1e-4 * np.sin(np.arange(n)), np.linspace(0.0, 0.12, n)])
    marker = np.column_stack([np.zeros(8), np.zeros(8), np.linspace(0.11, 0.12, 8)])

    def segment(query, **_):
        if query == "syringe":
            return {"masks": [np.ones((8, 8), np.uint8)], "scores": [0.8]}
        return {"masks": [2 * np.ones((8, 8), np.uint8)] if marker_visible else [], "scores": [0.7]}

    def to_points(mask, **_):
        return {"points": {"points": body if int(np.max(mask)) == 1 else marker}}

    return {"robot.get_ee_pose": {"pose": pose(0.0, 0.0, 0.05)}, "sam3.segment_text": segment,
            "geometry.mask_to_world_points": to_points}


LINEAR_GRID = list(itertools.product(
    ["shaft_into_aperture", "feature_to_fixture", "not_a_relation"],
    [None, "wrist", "wrist_no_marker", "overhead_only"],
    [(0.02, 0.035), (0.01, 0.05)],
))


@pytest.mark.parametrize("relation,observed,depths", LINEAR_GRID)
def test_plan_linear_engagement_defaults_match_752dfbd(relation, observed, depths):
    before, after = _before(LINEAR, "linear_before"), _load(LINEAR, "linear_after")
    observation = None if observed is None else {
        "cameras": [_camera("overhead")] + ([] if observed == "overhead_only" else [_camera("eye_in_hand_0")])}
    args = (pose(z=0.1), {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": -1.0}}, relation,
            {"meshes": []}, {"frame": "tcp", "spheres": []})
    kwargs = {"precontact_clearance_m": depths[0], "engagement_depth_m": depths[1], "observation": observation,
              "object_description": "syringe", "direction_marker_description": "cap"}
    got_before = _drive(before, _linear_responses(observed != "wrist_no_marker"), *args, **kwargs)
    got_after = _drive(after, _linear_responses(observed != "wrist_no_marker"), *args, **kwargs)
    if got_after["result"] is not None:
        plan = got_after["result"]["placement_plan"]
        assert ADDITIVE_PLAN_KEYS <= set(plan)
        got_after["result"] = {"placement_plan": {k: v for k, v in plan.items() if k not in ADDITIVE_PLAN_KEYS}}
    assert got_before == got_after


# --- parity: execute_reorientation --------------------------------------------------------------

REORIENT_PLANS = {
    "three_modes": [{"pose": pose(z=0.55), "mode": "contact_transition", "cartesian": True},
                    {"pose": pose(0.3, 0.0, 0.6), "mode": "planned_joint", "cartesian": False},
                    {"pose": pose(0.4, 0.0, 0.6), "mode": "planned_linear", "allow_goal_contact": True,
                     "contact_margin": 0.003}],
    "retry_fallback": [{"pose": pose(0.3, 0.0, 0.6), "mode": "planned_joint", "max_attempts": 3,
                        "cartesian_fallback": True, "speed_scale": 0.4, "use_world": False}],
    "legacy_flags": [{"pose": pose(0.3, 0.0, 0.6), "cartesian": False, "use_attachment": False},
                     {"pose": pose(0.3, 0.1, 0.6), "cartesian": True}],
    "bad_mode": [{"pose": pose(), "mode": "cartesian_cross"}],
}

REORIENT_GRID = list(itertools.product(
    sorted(REORIENT_PLANS), ["ok", "refuse"], [None, {"speed_scale": 0.5, "cartesian_fallback": True}],
    [False, True], [None, 1], [1.0, 2.5],
))


@pytest.mark.parametrize("plan_name,planner,profile,measure,arm,time_scale", REORIENT_GRID)
def test_execute_reorientation_defaults_match_752dfbd(plan_name, planner, profile, measure, arm, time_scale):
    before, after = _before(REORIENT, "reorient_before"), _load(REORIENT, "reorient_after")
    attachment = {"frame": "tcp", "spheres": [], "translation_uncertainty_m": 0.004}
    if arm is not None:
        attachment["arm_id"] = arm
    plan = {"time_scale": time_scale, "waypoints": REORIENT_PLANS[plan_name], "world_config": {"meshes": []},
            "attached_object": attachment}

    def responses():
        answer = (lambda **_: _ok(((0.0, 0.1), (0.5, 0.2), (1.0, 0.3)))) if planner == "ok" else {"planned": False}
        return {"motion.plan_to_pose": answer, "motion.plan_linear": answer, "robot.execute_trajectory": {},
                "robot.go_to_pose_cartesian": {}, "robot.get_ee_pose": {"pose": pose(0.3, 0.0, 0.6)}}

    kwargs = {"execution_profile": profile, "measure_errors": measure}
    assert _drive(before, responses(), plan, **kwargs) == _drive(after, responses(), plan, **kwargs)


# --- behaviour: planning -------------------------------------------------------------------------


def _plan_joint_ok(**_):
    return {"planned": True, "position_error_m": 0.0, "rotation_error_rad": 0.0}


def test_carry_then_turn_carries_in_pick_orientation_then_turns_in_place_with_a_search_block():
    from gap.testing.fakes import FakeContext

    module = _load(CLEARANCE, "clearance_carry_turn")
    tilted = Rotation.from_euler("x", 90, degrees=True)
    ctx = FakeContext({"robot.get_ee_pose": {"pose": pose(0.2, 0.1, 0.5, tilted)},
                       "motion.plan_joint": _plan_joint_ok})
    fixture = {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": -1.0}}
    feature = pose(z=0.1)
    out = module.run(ctx, feature, fixture, "shaft_into_aperture", {"meshes": []},
                     {"frame": "tcp", "spheres": [{"center": [0, 0, 0], "radius": 0.01}]},
                     strategy="carry_then_turn")
    plan = out["reorientation_plan"]
    escape, carry, turn = plan["waypoints"]
    assert [w["mode"] for w in plan["waypoints"]] == ["contact_transition", "planned_joint", "planned_joint"]
    assert carry["pose"]["rotation"] == pose(rotation=tilted)["rotation"]
    assert carry["pose"]["position"] == turn["pose"]["position"]
    assert turn["hold_position"] is True and "hold_position" not in carry
    search = turn["turn_search"]
    assert set(search) == {"fixture_center", "axis", "feature_offset", "feature_rotation", "transit_clearance_m",
                           "transit_target", "insert_depths_m"}
    assert search["insert_depths_m"] == [-0.020, -0.010, 0.0]
    # An aperture axis points into the opening: the feature stages against it.
    assert search["transit_target"] == pytest.approx([0.4, 0.0, 0.38])
    assert plan["time_scale"] == 2.0 and "arm_id" not in plan
    assert [r.tool for r in ctx.calls].count("motion.plan_joint") == 8


def test_carry_then_turn_keeps_the_library_staging_sign_for_a_loop():
    from gap.testing.fakes import FakeContext

    module = _load(CLEARANCE, "clearance_loop")
    ctx = FakeContext({"robot.get_ee_pose": {"pose": pose(0.2, 0.1, 0.5)}, "motion.plan_joint": _plan_joint_ok})
    out = module.run(ctx, pose(z=0.1), {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}},
                     "loop_over_shaft", {}, {"frame": "tcp", "spheres": []}, strategy="carry_then_turn",
                     turn_search_depths_m=[0.03, 0.0])
    search = out["reorientation_plan"]["waypoints"][2]["turn_search"]
    assert search["transit_target"] == pytest.approx([0.4, 0.0, 0.38])  # +axis: a shaft's free end
    assert search["insert_depths_m"] == [0.03, 0.0]
    bare = module.run(FakeContext({"robot.get_ee_pose": {"pose": pose(z=0.5)}, "motion.plan_joint": _plan_joint_ok}),
                      pose(z=0.1), {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": 1.0}},
                      "loop_over_shaft", {}, {"frame": "tcp", "spheres": []}, strategy="carry_then_turn",
                      turn_search=False)
    assert "turn_search" not in bare["reorientation_plan"]["waypoints"][2]


def test_unchecked_symmetry_fallback_is_opt_in():
    from gap.testing.fakes import FakeContext

    module = _load(CLEARANCE, "clearance_unchecked")
    args = (pose(z=0.1), {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": -1.0}},
            "shaft_into_aperture", {}, {"frame": "tcp", "spheres": []})

    def ctx():
        return FakeContext({"robot.get_ee_pose": {"pose": pose(z=0.5)}, "motion.plan_joint": {"planned": False}})

    with pytest.raises(RuntimeError, match="no feasible orientation"):
        module.run(ctx(), *args, strategy="carry_then_turn")
    out = module.run(ctx(), *args, strategy="carry_then_turn", accept_unchecked_symmetry=True)
    turn = out["reorientation_plan"]["waypoints"][2]
    # The feature (+z) must face the aperture axis (-z): the smallest unchecked turn is a half turn.
    assert abs(turn["pose"]["rotation"]["w"]) == pytest.approx(0.0, abs=1e-9)
    assert "turn_search" in turn


# --- behaviour: engagement ----------------------------------------------------------------------


def _engagement_ctx(handlers=None):
    from gap.testing.fakes import FakeContext

    return FakeContext({"robot.get_ee_pose": {"pose": pose(0.4, 0.0, 0.40)}, **(handlers or {})})


APERTURE = {"pose": pose(0.4, 0.0, 0.3), "axis": {"x": 0.0, "y": 0.0, "z": -1.0}}


def test_staged_strokes_replace_the_two_legs():
    module = _load(LINEAR, "linear_strokes")
    out = module.run(_engagement_ctx(), pose(z=-0.06), APERTURE, "shaft_into_aperture", {}, {},
                     stroke_depths_m=[-0.020, -0.010, 0.0])
    plan = out["placement_plan"]
    assert [w["mode"] for w in plan["waypoints"]] == ["planned_linear"] * 3
    assert [w["pose"]["position"]["z"] for w in plan["waypoints"]] == pytest.approx([0.38, 0.37, 0.36])
    assert "allow_goal_contact" not in plan["waypoints"][0]
    for waypoint in plan["waypoints"][1:]:
        assert waypoint["allow_start_contact"] and waypoint["allow_goal_contact"]
        assert waypoint["contact_margin"] == 0.008
    assert plan["aim_source"] == "carried" and plan["rim_z"] == pytest.approx(0.3)
    joint_first = module.run(_engagement_ctx(), pose(z=-0.06), APERTURE, "shaft_into_aperture", {}, {},
                             stroke_depths_m=[-0.02, 0.0], first_stroke_mode="planned_joint")
    assert joint_first["placement_plan"]["waypoints"][0]["cartesian"] is False


def test_mirroring_aims_the_end_that_leads_along_the_fixture_axis():
    module = _load(LINEAR, "linear_mirror")
    # The carried feature points up (+z) from the TCP; the aperture axis is -z.
    out = module.run(_engagement_ctx(), pose(z=0.06), APERTURE, "shaft_into_aperture", {}, {},
                     mirror_leading_end=True)
    plan = out["placement_plan"]
    assert plan["aim_source"] == "carried_mirrored"
    assert plan["leading_end_in_hand"]["z"] == pytest.approx(-0.06)


def _wrist_tip_handlers(tip_z):
    n = 100
    # On the hand's axis (the hand is at x = 0.4), hanging from it down to ``tip_z``; the marker at the tip.
    body = np.column_stack([0.4 + 1e-4 * np.cos(np.arange(n)), 1e-4 * np.sin(np.arange(n)),
                            np.linspace(tip_z, 0.40, n)])
    marker = np.column_stack([np.full(8, 0.4), np.zeros(8), np.linspace(tip_z, tip_z + 0.01, 8)])
    return {
        "sam3.segment_text": lambda query, **_: (
            {"masks": [np.ones((8, 8), np.uint8)], "scores": [0.8]} if query == "syringe"
            else {"masks": [2 * np.ones((8, 8), np.uint8)], "scores": [0.7]}),
        "geometry.mask_to_world_points": lambda mask, **_: {
            "points": {"points": body if int(np.max(mask)) == 1 else marker}},
    }


def test_plausibility_band_lets_perception_replace_only_an_implausible_carried_end():
    module = _load(LINEAR, "linear_band")
    kwargs = {"observation": {"cameras": [_camera("eye_in_hand_0")]}, "object_description": "syringe",
              "direction_marker_description": "cap", "leading_end_band_m": [0.045, 0.11]}
    # A wrist tip 70 mm below the TCP (x offset 0.4 world -> hand frame via identity rotation).
    handlers = _wrist_tip_handlers(0.33)

    ctx = _engagement_ctx(handlers)
    kept = module.run(ctx, pose(0.0, 0.0, -0.06), APERTURE, "shaft_into_aperture", {}, {},
                      tip_source="carried_unless_implausible", **kwargs)["placement_plan"]
    assert kept["aim_source"] == "carried" and not ctx.calls_to("sam3.segment_text")

    ctx = _engagement_ctx(handlers)
    replaced = module.run(ctx, pose(0.0, 0.0, -0.20), APERTURE, "shaft_into_aperture", {}, {},
                          tip_source="carried_unless_implausible", **kwargs)["placement_plan"]
    assert replaced["aim_source"] == "wrist_tip"
    assert replaced["leading_end_in_hand"]["z"] == pytest.approx(-0.07, abs=0.003)

    # Default mode: perception wins, but a perceived end outside the band is ignored.
    ctx = _engagement_ctx(_wrist_tip_handlers(0.20))  # 200 mm below the TCP
    ignored = module.run(ctx, pose(0.0, 0.0, -0.06), APERTURE, "shaft_into_aperture", {}, {}, **kwargs)
    assert ignored["placement_plan"]["aim_source"] == "carried" and ctx.calls_to("sam3.segment_text")
    with pytest.raises(ValueError, match="needs leading_end_band_m"):
        module.run(_engagement_ctx(), pose(), APERTURE, "shaft_into_aperture", {}, {},
                   tip_source="carried_unless_implausible")


def test_side_view_reads_the_leading_end_along_the_fixture_axis():
    module = _load(LINEAR, "linear_side")
    hand = np.array([0.4, 0.0, 0.40])
    side = {"name": "agentview", "rgb": np.zeros((100, 100, 3), np.uint8), "depth": np.ones((100, 100), np.float32),
            "intrinsics": [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            "pose": pose(0.4, -0.0, -0.10)}  # looking along +z at the hand, 0.5 m away
    n = 200
    shaft = np.column_stack([hand[0] + 2e-3 * np.cos(np.arange(n)), 2e-3 * np.sin(np.arange(n)),
                             np.linspace(0.40 + 0.02, 0.40 - 0.08, n)])
    handlers = {
        "sam3.segment_text": lambda image, **_: {"masks": [np.ones(np.asarray(image).shape[:2], np.uint8)],
                                                 "scores": [0.9]},
        "geometry.mask_to_world_points": lambda **_: {"points": {"points": shaft}},
    }
    ctx = _engagement_ctx(handlers)
    plan = module.run(ctx, pose(0.0, 0.0, -0.20), APERTURE, "shaft_into_aperture", {}, {},
                      observation={"cameras": [side]}, object_description="syringe",
                      side_camera_names=["agentview"], leading_end_band_m=[0.045, 0.11],
                      tip_source="carried_unless_implausible")["placement_plan"]
    assert plan["aim_source"] == "side_view"
    assert plan["leading_end_in_hand"]["z"] == pytest.approx(-0.075, abs=0.005)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["max_results"] == 0


# --- behaviour: execution -----------------------------------------------------------------------

LIMITS = {"joint_limits": {"lower": [-1.0, -1.0], "upper": [1.0, 1.0]}}
SEARCH = {"fixture_center": [0.4, 0.0, 0.3], "axis": [0.0, 0.0, -1.0], "feature_offset": [0.0, 0.0, 0.1],
          "feature_rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}, "transit_clearance_m": 0.08,
          "insert_depths_m": [-0.02, -0.01, 0.0]}


def _search_plan(**turn_search):
    flipped = Rotation.from_euler("x", 180, degrees=True)  # feature axis already along the aperture axis
    return {"time_scale": 2.0, "world_config": {"meshes": []}, "attached_object": {"frame": "tcp", "spheres": []},
            "waypoints": [
                {"pose": pose(0.2, 0.0, 0.55, flipped), "mode": "contact_transition", "cartesian": True},
                {"pose": pose(0.4, 0.0, 0.48, flipped), "mode": "planned_joint", "cartesian": False},
                {"pose": pose(0.4, 0.0, 0.48, flipped), "mode": "planned_joint", "cartesian": False,
                 "hold_position": True, "turn_search": {**SEARCH, **turn_search}},
            ]}


def _search_ctx(turn_values, live=(0.1, 0.1)):
    """Carry plans end at joints (0.1, 0.1); the i-th turn plan sits at joint value v (margin 1 - |v|)."""
    from gap.testing.fakes import FakeContext

    values = list(turn_values)

    def plan_to_pose(**kwargs):
        if "hold_position" not in kwargs:
            return _ok(((0.0, 0.0), (0.1, 0.1)))
        v = values.pop(0)
        return _ok(((0.1, 0.1), (v, v)))

    return FakeContext({
        "robot.describe_arm": LIMITS,
        "robot.get_ee_pose": {"pose": pose(0.2, 0.0, 0.55, Rotation.from_euler("x", 180, degrees=True))},
        "robot.go_to_pose_cartesian": {},
        "robot.execute_trajectory": {},
        "robot.forward_kinematics": {"joint_config": list(live)},
        "motion.plan_to_pose": plan_to_pose,
        "motion.plan_linear": _ok(((0.0, 0.0),)),
    })


def _yaw_index(yaw):
    return int((yaw + 180) // 30)


def test_turn_search_keeps_the_in_place_yaw_with_the_most_joint_margin():
    module = _load(REORIENT, "reorient_search")
    values = [0.98] * 12
    values[_yaw_index(30)] = 0.75  # margin 0.25 >= margin_good: the search stops at in-place turns
    ctx = _search_ctx(values)
    out = module.run(ctx, _search_plan())
    chosen = out["waypoint_reports"][-1]["turn_search"]
    assert (chosen["yaw_deg"], chosen["strategy"], chosen["replanned"]) == (30.0, "turn in place", False)
    assert chosen["margin_rad"] == pytest.approx(0.25)
    turns = [r.kwargs for r in ctx.calls_to("motion.plan_to_pose") if "hold_position" in r.kwargs]
    assert len(turns) == 12 and all(k["hold_position"] and k["seed_joints"] == [0.1, 0.1] for k in turns)
    linear = [r.kwargs for r in ctx.calls_to("motion.plan_linear")]
    assert len(linear) == 12 * 4 and all("start" in k and "seed_joints" in k for k in linear)
    retract = linear[3]
    assert "attached_object" not in retract
    assert retract["end"]["position"]["z"] - retract["start"]["position"]["z"] == pytest.approx(0.16)
    executed = [r.kwargs["trajectory"]["waypoints"] for r in ctx.calls_to("robot.execute_trajectory")]
    assert len(executed) == 2 and len(executed[1]) == 5  # the turn plays at 4x
    assert executed[1][-1]["positions"] == pytest.approx([0.75, 0.75])
    assert out["final_pose"]["rotation"] != _search_plan()["waypoints"][2]["pose"]["rotation"]
    assert [r["index"] for r in out["waypoint_reports"]] == [0, 1, 2]


def test_turn_search_caps_margin_then_prefers_the_smaller_turn():
    module = _load(REORIENT, "reorient_cap")
    values = [0.98] * 12
    values[_yaw_index(0)], values[_yaw_index(90)] = 0.65, 0.5  # margins 0.35 and 0.5, both past 0.30
    out = module.run(_search_ctx(values), _search_plan())
    assert out["waypoint_reports"][-1]["turn_search"]["yaw_deg"] == 0.0


def test_turn_search_carries_upright_when_no_in_place_turn_has_margin():
    module = _load(REORIENT, "reorient_upright")
    upright = [0.98] * 12
    upright[_yaw_index(-60)] = 0.8
    ctx = _search_ctx([0.98] * 12 + upright)
    out = module.run(ctx, _search_plan(retract_probe_axis=[1.0, 0.0, 0.0], retract_probe_m=0.1))
    chosen = out["waypoint_reports"][-1]["turn_search"]
    assert (chosen["yaw_deg"], chosen["strategy"]) == (-60.0, "carried upright")
    upright_calls = [r.kwargs for r in ctx.calls_to("motion.plan_to_pose")
                     if r.kwargs.get("hold_position") is False]
    assert len(upright_calls) == 12 and all("seed_joints" not in k for k in upright_calls)
    executed = ctx.calls_to("robot.execute_trajectory")
    assert len(executed) == 1 and len(executed[0].kwargs["trajectory"]["waypoints"]) == 7  # 6x, no carry
    assert not ctx.calls_to("robot.forward_kinematics")
    retract = ctx.calls_to("motion.plan_linear")[3].kwargs
    assert retract["end"]["position"]["x"] - retract["start"]["position"]["x"] == pytest.approx(0.1)


def test_turn_is_replanned_from_live_joints_when_the_carry_drifts():
    module = _load(REORIENT, "reorient_drift")
    values = [0.98] * 12 + [0.4]  # the 13th turn plan is the re-plan
    values[_yaw_index(30)] = 0.75
    ctx = _search_ctx(values, live=(0.3, 0.1))
    out = module.run(ctx, _search_plan())
    assert out["waypoint_reports"][-1]["turn_search"]["replanned"] is True
    replan = ctx.calls_to("motion.plan_to_pose")[-1].kwargs
    assert replan["hold_position"] is True and "seed_joints" not in replan
    assert ctx.calls_to("robot.execute_trajectory")[-1].kwargs["trajectory"]["waypoints"][-1]["positions"] == \
        pytest.approx([0.4, 0.4])

    still = _search_ctx([0.98] * 7 + [0.75] + [0.98] * 4, live=(0.1 + 0.04, 0.1))
    assert module.run(still, _search_plan())["waypoint_reports"][-1]["turn_search"]["replanned"] is False


def test_turn_search_with_no_feasible_candidate_blocks():
    from gap.testing.fakes import FakeContext

    module = _load(REORIENT, "reorient_none")
    ctx = FakeContext({"robot.describe_arm": LIMITS, "robot.get_ee_pose": {"pose": pose(z=0.5)},
                       "robot.go_to_pose_cartesian": {}, "motion.plan_to_pose": {"planned": False},
                       "motion.plan_linear": {"planned": False}})
    with pytest.raises(RuntimeError, match="no carry and wrist yaw"):
        module.run(ctx, _search_plan())
    assert not ctx.calls_to("robot.execute_trajectory")


def test_hold_position_is_forwarded_only_when_a_waypoint_carries_it():
    from gap.testing.fakes import FakeContext

    module = _load(REORIENT, "reorient_hold")
    ctx = FakeContext({"motion.plan_to_pose": _ok(), "robot.execute_trajectory": {}})
    module.run(ctx, {"waypoints": [{"pose": pose(z=0.5), "mode": "planned_joint", "hold_position": True},
                                   {"pose": pose(z=0.6), "mode": "planned_joint"}]})
    first, second = (r.kwargs for r in ctx.calls_to("motion.plan_to_pose"))
    assert first["hold_position"] is True and "hold_position" not in second


def test_arrival_tolerance_raises_to_blocked_and_is_off_by_default():
    from gap.testing.fakes import FakeContext

    module = _load(REORIENT, "reorient_arrival")
    plan = {"waypoints": [{"pose": pose(z=0.50), "mode": "contact_transition"}]}

    def ctx():
        return FakeContext({"robot.go_to_pose_cartesian": {}, "robot.get_ee_pose": {"pose": pose(z=0.51)}})

    plain = ctx()
    module.run(plain, plan)
    assert not plain.calls_to("robot.get_ee_pose")
    with pytest.raises(RuntimeError, match="waypoint 0 ended 10.0 mm"):
        module.run(ctx(), plan, arrival_tolerance_m=0.003)
    report = module.run(ctx(), plan, arrival_tolerance_m=0.02)["waypoint_reports"][0]
    assert report["position_error_m"] == pytest.approx(0.01)
