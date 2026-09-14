"""The sharps_disposal v2 fold into two bundles: defaults unchanged, opt-ins do what they say.

RoboSimStudio ``sharps_disposal/gap_perception_v2`` (sweep s8) carried three
changes to scripts these bundles own: a GPU TSDF collision world
(``reconstructing-collision-worlds``), a planned retract ladder and a checked,
time-scaled insertion (``executing-feature-mating``). Each was folded in behind
a parameter whose default is the script as it was.

The first half of this file is the proof of that, in the manner of
``test_promotion_is_behaviour_preserving``: each script is driven as it stood at
``752dfbd`` and as it is now, over a grid of inputs, through a recording
:class:`gap.testing.FakeContext`, and the full tool-call sequence (names and
keyword values), the return value and any raised error must agree. The second
half exercises the opt-ins, including every fallback.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[1]
BASE = "752dfbd"
WORLD = "skills/reconstructing-collision-worlds/scripts/build_collision_world.py"
RELEASE = "skills/executing-feature-mating/scripts/release_and_retract.py"
EXECUTE = "skills/executing-feature-mating/scripts/execute_placement_plan.py"

IDENTITY = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


# ---------------------------------------------------------------------------
# Loading and recording
# ---------------------------------------------------------------------------


def _load(source: str, name: str):
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _at_ref(rel: str, ref: str) -> str:
    done = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        pytest.skip(f"{ref} is not in this checkout: {done.stderr.strip()}")
    return done.stdout


def _pair(rel: str):
    stem = Path(rel).stem
    return _load(_at_ref(rel, BASE), f"{stem}_before"), _load((ROOT / rel).read_text(), f"{stem}_after")


@pytest.fixture(scope="module")
def modules():
    return {rel: _pair(rel) for rel in (WORLD, RELEASE, EXECUTE)}


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, default=_jsonable)


def _drive(module, responses: dict, *args, **kwargs) -> dict:
    ctx = FakeContext(responses)
    try:
        result, error = module.run(ctx, *args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 -- the error is part of the comparison
        result, error = None, f"{type(exc).__name__}: {exc}"
    return {
        "calls": [(r.tool, _canon(r.kwargs)) for r in ctx.calls],
        "result": _canon(result),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CUBE_FACES = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 7, 5], [4, 6, 7], [0, 5, 1], [0, 4, 5],
     [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 7, 3], [1, 5, 7]],
    dtype=np.int32,
)


def _box(name: str, center, size) -> dict:
    cx, cy, cz = center
    sx, sy, sz = size
    vertices = np.array(
        [[cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2] for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)],
        dtype=np.float32,
    )
    return {"name": name, "vertices": vertices, "faces": _CUBE_FACES.copy(), "pose": IDENTITY}


def _sliver() -> dict:
    vertices = np.array(
        [[0.3, 0.0, 0.9], [0.35, 0.0, 0.9], [0.3, 0.002, 0.9], [0.35, 0.002, 0.902]], dtype=np.float32
    )
    return {"name": "sliver", "vertices": vertices, "faces": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            "pose": IDENTITY}


def _scene(cube=(0.5, 0.0, 0.945)) -> list[dict]:
    return [_sliver(), _box("cube", cube, (0.02, 0.02, 0.03)), _box("slab", (0.5, 0.0, 0.89), (1.0, 1.0, 0.02))]


def _frame(name: str, fill: int = 0) -> dict:
    return {
        "name": name,
        "rgb": np.full((8, 8, 3), fill, dtype=np.uint8),
        "depth": np.ones((8, 8), dtype=np.float64),
        "intrinsics": np.eye(3),
        "pose": IDENTITY,
    }


def _segment(image, query, max_results):
    if query == "syringe":
        return {"masks": [np.ones((8, 8), dtype=np.uint8)], "scores": [0.5]}
    if query == "robot gripper":
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:2, :2] = 1
        return {"masks": [mask], "scores": [0.4]}
    return {"masks": [], "scores": []}


def _world_responses(meshes: list[dict], spheres: str = "absent", **extra) -> dict:
    responses = {
        "sam3.segment_text": _segment,
        # A fresh config per call: the script writes the carved meshes back into it.
        "geometry.build_world_config": lambda **_kw: {
            "config": {"meshes": [dict(m) for m in meshes]},
            "mesh_names": [m["name"] for m in meshes],
        },
    }
    if spheres == "present":
        responses["motion.get_robot_collision_spheres"] = {
            "spheres": [{"center": {"x": 0.1, "y": 0.0, "z": 1.0}, "radius": 0.05}]
        }
    elif spheres == "raises":
        def refuse(**_kw):
            raise RuntimeError("scripted: no planner model")

        responses["motion.get_robot_collision_spheres"] = refuse
    responses.update(extra)
    return responses


def _pose(z, x=0.4, rotation=None):
    return {"position": {"x": x, "y": 0.0, "z": z}, "rotation": rotation or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


_TRAJECTORY = {"planned": True, "trajectory": {"waypoints": [{"positions": [0.0, 1.0]}, {"positions": [0.5, 1.5]}]}}


# ---------------------------------------------------------------------------
# Parity: build_collision_world
# ---------------------------------------------------------------------------

_KEEP_OUT_PROFILE = {
    "source_kind": "syringe", "strategy": "rgbd_mesh",
    "excluded_masks": {"target": True, "cross_view_target": True, "robot": True},
    "fixture_keep_out": {"boxes": [{"name": "wall", "center": {"x": 0.5, "y": 0.0, "z": 0.5},
                                    "size": {"x": 0.1, "y": 0.1, "z": 0.1}}]},
}

WORLD_CASES = {
    "default": ({}, _scene(), False),
    "corridor": (dict(corridor_center={"x": 0.5, "y": 0.0, "z": 0.95}, corridor_radius=0.03,
                      corridor_rim_z=0.95), _scene(), False),
    "pack": (dict(pack_meshes=True, voxel_size=0.016), _scene(), False),
    "tube_out": (dict(fixture_tip={"x": 0.5, "y": 0.0, "z": 0.95}, fixture_axis={"x": -1.0, "y": 0.0, "z": 0.0},
                      outward_sign=1.0), _scene((0.45, 0.0, 0.95)), False),
    "tube_in": (dict(fixture_tip={"x": 0.5, "y": 0.0, "z": 0.95}, fixture_axis={"x": -1.0, "y": 0.0, "z": 0.0},
                     outward_sign=-1.0), _scene((0.45, 0.0, 0.95)), False),
    "two_views": (dict(target_description="syringe", fixture_mask=np.zeros((8, 8), dtype=np.uint8)), _scene(), True),
    "select_view": (dict(target_description="syringe", camera_names="overhead"), _scene(), True),
    "no_camera": (dict(camera_names="wrist"), _scene(), False),
    "nothing_survives": ({}, [_sliver()], False),
    "profile": (dict(target_kind="syringe", collision_profiles=[_KEEP_OUT_PROFILE]), _scene(), True),
    "disabled": (dict(target_kind="syringe", collision_profiles=[{"source_kind": "syringe", "strategy": "disabled"}]),
                 _scene(), False),
    "given_spheres": (dict(robot_spheres=[]), _scene(), False),
}


@pytest.mark.parametrize(
    "case,spheres", list(itertools.product(sorted(WORLD_CASES), ["absent", "present", "raises"]))
)
def test_world_defaults_are_the_script_before_the_fold(modules, case, spheres):
    before, after = modules[WORLD]
    kwargs, meshes, two_views = WORLD_CASES[case]

    def observation():
        cameras = [_frame("overhead")] + ([_frame("agentview", 1)] if two_views else [])
        return {"cameras": cameras}

    target = np.zeros((8, 8), dtype=np.uint8)
    got_before = _drive(before, _world_responses(meshes, spheres), observation(), target, **kwargs)
    got_after = _drive(after, _world_responses(meshes, spheres), observation(), target, **kwargs)
    assert got_before == got_after
    assert not any(name == "motion.build_world_tsdf" for name, _ in got_after["calls"])


# ---------------------------------------------------------------------------
# Parity: release_and_retract
# ---------------------------------------------------------------------------

_SMALL = {"frame": "tcp", "spheres": [{"center": {"x": 0.0, "y": 0.0, "z": 0.0}, "radius": 0.01}]}
_LARGE = {"frame": "tcp", "arm_id": 1,
          "spheres": [{"center": {"x": 0.12, "y": 0.0, "z": -0.05}, "radius": 0.02}]}

RELEASE_GRID = list(
    itertools.product(
        [_pose(0.5), _pose(0.62, x=0.55)],
        [None, {"x": 0.0, "y": 2.0, "z": 0.0}],
        [None, _SMALL, _LARGE],
        ["loop_over_shaft", "shaft_into_aperture"],
        [0.0, 0.2],
        [False, True],
    )
)


def _release_responses(servo_fails: bool) -> dict:
    def servo(**_kw):
        if servo_fails:
            raise RuntimeError("scripted: servo stalled")
        return {}

    return {"robot.open_gripper": None, "robot.go_to_pose_cartesian": servo, "robot.wait_steps": None,
            "motion.plan_linear": _TRAJECTORY, "robot.execute_trajectory": None}


@pytest.mark.parametrize("final_pose,axis,attached,relation,retract_m,servo_fails", RELEASE_GRID)
def test_release_defaults_are_the_script_before_the_fold(
    modules, final_pose, axis, attached, relation, retract_m, servo_fails
):
    before, after = modules[RELEASE]
    kwargs = dict(retract_m=retract_m, retreat_axis=axis, attached_object=attached, relation=relation)
    got_before = _drive(before, _release_responses(servo_fails), final_pose, **kwargs)
    got_after = _drive(after, _release_responses(servo_fails), final_pose, **kwargs)
    assert got_before == got_after


def test_release_profile_defaults_are_the_script_before_the_fold(modules):
    before, after = modules[RELEASE]
    profile = {"release_settle_steps": 100, "post_release_wait_steps": 200, "retract_m": 0.16,
               "vertical_retreat_fraction": 0.25}
    kwargs = dict(retreat_axis={"x": 1.0, "y": 0.0, "z": 0.0}, contact_profile=profile, arm_id=0)
    assert _drive(before, _release_responses(False), _pose(0.5), **kwargs) == _drive(
        after, _release_responses(False), _pose(0.5), **kwargs
    )


# ---------------------------------------------------------------------------
# Parity: execute_placement_plan
# ---------------------------------------------------------------------------


def _plan(*waypoints, arm=None):
    attached = {"frame": "tcp", "spheres": []}
    if arm is not None:
        attached["arm_id"] = arm
    return {"waypoints": list(waypoints), "world_config": {"meshes": []}, "attached_object": attached}


PLANS = {
    "joint": [{"pose": _pose(0.5), "mode": "planned_joint"}],
    "servo_then_skip": [{"pose": _pose(0.5), "mode": "planned_joint"}, {"pose": _pose(0.4), "mode": "planned_linear"}],
    "linear_contact": [{"pose": _pose(0.5), "mode": "planned_linear", "allow_goal_contact": True,
                        "contact_margin": 0.002}, {"pose": _pose(0.45), "cartesian": True}],
    "cross_seat": [{"pose": _pose(0.45), "mode": "cartesian_cross"}, {"pose": _pose(0.40), "mode": "contact_seat"}],
    "unknown": [{"pose": _pose(0.5), "mode": "contact_transition"}],
    "empty": [],
}
PLANNERS = {
    "plans": lambda: _TRAJECTORY,
    "refuses": lambda: {"planned": False},
    "second_try": lambda: [{"planned": False}, {"planned": True, "trajectory": {"waypoints": []}}, _TRAJECTORY],
}

EXECUTE_GRID = list(itertools.product(sorted(PLANS), sorted(PLANNERS), [0.8, 0.505, 0.4005], [None, 1]))


def _execute_responses(planner: str, ee_z: float) -> dict:
    return {
        "robot.get_ee_pose": {"pose": _pose(ee_z)},
        "motion.plan_to_pose": PLANNERS[planner](),
        "motion.plan_linear": PLANNERS[planner](),
        "robot.execute_trajectory": None,
        "robot.go_to_pose_cartesian": None,
        "robot.move_cartesian_until_contact": {"status": "target"},
        "robot.wait_steps": None,
        "robot.describe_arm": {"solver": {"honours_roll": True}},
    }


@pytest.mark.parametrize("plan,planner,ee_z,arm", EXECUTE_GRID)
def test_execute_defaults_are_the_script_before_the_fold(modules, plan, planner, ee_z, arm):
    before, after = modules[EXECUTE]
    placement = _plan(*PLANS[plan], arm=arm)
    got_before = _drive(before, _execute_responses(planner, ee_z), placement)
    got_after = _drive(after, _execute_responses(planner, ee_z), placement)
    assert got_before == got_after


def test_execute_existing_switches_are_the_script_before_the_fold(modules):
    before, after = modules[EXECUTE]
    placement = _plan(*PLANS["linear_contact"], *PLANS["cross_seat"])
    kwargs = dict(verify_cartesian=True, measure_errors=True, contact_profile={"max_steps_per_waypoint": 180})
    assert _drive(before, _execute_responses("plans", 0.8), placement, **kwargs) == _drive(
        after, _execute_responses("plans", 0.8), placement, **kwargs
    )


# ---------------------------------------------------------------------------
# Behaviour: TSDF world
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world(modules):
    return modules[WORLD][1]


_CORRIDOR = dict(corridor_center={"x": 0.5, "y": 0.0, "z": 0.95}, corridor_rim_z=0.95)


def test_tsdf_mode_asks_the_backend_and_clears_the_descent(world):
    ctx = FakeContext(_world_responses(_scene(), "present", **{"motion.build_world_tsdf": {"integrated": 1}}))
    target = np.zeros((8, 8), dtype=np.uint8)
    out = world.run(ctx, {"cameras": [_frame("overhead")]}, target, tsdf=True, corridor_radius=0.03, **_CORRIDOR)
    # Masks are still segmented; no sphere query and no mesh reconstruction.
    assert [call.tool for call in ctx.calls] == ["sam3.segment_text", "sam3.segment_text", "motion.build_world_tsdf"]
    request = ctx.calls[-1].kwargs
    assert request["voxel_size"] == pytest.approx(0.01)
    assert len(request["exclude_masks"]) == 2  # the grasp target and the robot gripper
    (cylinder,) = request["clear_cylinders"]
    assert cylinder["x"] == pytest.approx(0.5) and cylinder["y"] == pytest.approx(0.0)
    assert cylinder["radius"] == pytest.approx(0.05)  # 1.1 * 30 mm is under the 50 mm floor
    assert cylinder["z_lo"] == pytest.approx(0.75) and cylinder["z_hi"] == pytest.approx(1.20)
    assert out == {"world_config": {"use_tsdf": True, "meshes": [], "tsdf": {"integrated": 1}},
                   "mesh_names": [], "removed_mesh_names": [], "strategy": "rgbd_tsdf"}

    ctx = FakeContext(_world_responses(_scene(), **{"motion.build_world_tsdf": {}}))
    world.run(ctx, {"cameras": [_frame("overhead")]}, target, tsdf=True, corridor_radius=0.06,
              tsdf_voxel_size=0.02, tsdf_clear_below_rim_m=0.1, **_CORRIDOR)
    request = ctx.calls_to("motion.build_world_tsdf")[0].kwargs
    assert request["clear_cylinders"][0]["radius"] == pytest.approx(0.066)
    assert request["clear_cylinders"][0]["z_lo"] == pytest.approx(0.85)
    assert request["voxel_size"] == pytest.approx(0.02)
    # No corridor asked for: nothing cleared.
    ctx = FakeContext(_world_responses(_scene(), **{"motion.build_world_tsdf": {}}))
    world.run(ctx, {"cameras": [_frame("overhead")]}, target, tsdf=True)
    assert ctx.calls_to("motion.build_world_tsdf")[0].kwargs["clear_cylinders"] == []


def _mesh_run(world, **kwargs):
    ctx = FakeContext(_world_responses(_scene(), "present"))
    out = world.run(ctx, {"cameras": [_frame("overhead")]}, np.zeros((8, 8), dtype=np.uint8), **kwargs)
    return ctx, out


@pytest.mark.parametrize("tool", ["raises", "absent"])
def test_tsdf_failure_falls_back_loudly_to_the_same_mesh_world(world, capsys, tool):
    extra = {}
    if tool == "raises":
        def refuse(**_kw):
            raise RuntimeError("scripted: mapper unavailable")

        extra["motion.build_world_tsdf"] = refuse
    ctx = FakeContext(_world_responses(_scene(), "present", **extra))
    out = world.run(ctx, {"cameras": [_frame("overhead")]}, np.zeros((8, 8), dtype=np.uint8), tsdf=True,
                    corridor_radius=0.03, **_CORRIDOR)
    assert "falling back to mesh reconstruction" in capsys.readouterr().out
    mesh_ctx, mesh_out = _mesh_run(world, corridor_radius=0.03, **_CORRIDOR)
    assert _canon(out) == _canon(mesh_out)
    assert out["strategy"] == "rgbd_mesh" and "use_tsdf" not in out["world_config"]
    calls = [(c.tool, _canon(c.kwargs)) for c in ctx.calls if c.tool != "motion.build_world_tsdf"]
    assert calls == [(c.tool, _canon(c.kwargs)) for c in mesh_ctx.calls]
    assert ctx.call_count("motion.build_world_tsdf") == 1


def test_tsdf_refuses_what_a_mask_list_and_a_cylinder_cannot_express(world, capsys):
    tsdf = {"motion.build_world_tsdf": {}}
    # Two views: the tool would apply one view's masks to the other.
    ctx = FakeContext(_world_responses(_scene(), **tsdf))
    out = world.run(ctx, {"cameras": [_frame("overhead"), _frame("agentview", 1)]},
                    np.zeros((8, 8), dtype=np.uint8), tsdf=True)
    assert ctx.call_count("motion.build_world_tsdf") == 0
    assert ctx.call_count("geometry.build_world_config") == 1 and out["strategy"] == "rgbd_mesh"
    assert "exactly one camera" in capsys.readouterr().out
    # An approach tube along an arbitrary axis.
    ctx = FakeContext(_world_responses(_scene((0.45, 0.0, 0.95)), **tsdf))
    out = world.run(ctx, {"cameras": [_frame("overhead")]}, np.zeros((8, 8), dtype=np.uint8), tsdf=True,
                    fixture_tip={"x": 0.5, "y": 0.0, "z": 0.95}, fixture_axis={"x": -1.0, "y": 0.0, "z": 0.0})
    assert ctx.call_count("motion.build_world_tsdf") == 0
    assert [m["name"] for m in out["world_config"]["meshes"]] == ["slab"]  # the tube was carved
    assert "approach tube" in capsys.readouterr().out


def test_tsdf_profile_strategy_carries_keep_out_boxes_and_tunables(world):
    profile = dict(_KEEP_OUT_PROFILE, strategy="rgbd_tsdf", tsdf={"voxel_size_m": 0.012, "clear_radius_min_m": 0.0})
    ctx = FakeContext(_world_responses(_scene(), **{"motion.build_world_tsdf": {"ok": True}}))
    out = world.run(ctx, {"cameras": [_frame("overhead")]}, np.zeros((8, 8), dtype=np.uint8),
                    target_kind="syringe", collision_profiles=[profile], corridor_radius=0.03, **_CORRIDOR)
    request = ctx.calls_to("motion.build_world_tsdf")[0].kwargs
    assert request["voxel_size"] == pytest.approx(0.012)
    assert request["clear_cylinders"][0]["radius"] == pytest.approx(0.033)
    assert out["strategy"] == "rgbd_tsdf" and out["world_config"]["use_tsdf"] is True
    assert [m["name"] for m in out["world_config"]["meshes"]] == ["wall"] and out["mesh_names"] == ["wall"]


# ---------------------------------------------------------------------------
# Behaviour: planned retract ladder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release(modules):
    return modules[RELEASE][1]


def _release_ctx(plan_linear) -> FakeContext:
    return FakeContext({"robot.open_gripper": None, "robot.go_to_pose_cartesian": None, "robot.wait_steps": None,
                        "motion.plan_linear": plan_linear, "robot.execute_trajectory": None})


def test_retract_ladder_executes_the_first_rung_that_plans(release):
    ctx = _release_ctx([{"planned": False}, _TRAJECTORY])
    out = release.run(ctx, _pose(0.5), retract_planner=True)
    assert [call.tool for call in ctx.calls] == [
        "robot.open_gripper", "motion.plan_linear", "motion.plan_linear", "robot.execute_trajectory", "robot.wait_steps",
    ]
    first, second = ctx.calls_to("motion.plan_linear")
    assert first.kwargs == {"end": _pose(0.5 + 0.16), "orientation": "lock", "allow_start_contact": True,
                            "contact_margin": 0.008}
    assert second.kwargs["end"]["position"]["z"] == pytest.approx(0.58)
    assert ctx.calls_to("robot.execute_trajectory")[0].kwargs == {"trajectory": _TRAJECTORY["trajectory"]}
    assert out["retreat_pose"]["position"]["z"] == pytest.approx(0.58)
    assert out["release_report"]["retreat_mode"] == "planned_linear"
    assert out["release_report"]["retreat_distance_m"] == pytest.approx(0.08)


@pytest.mark.parametrize("refusal", ["refused", "empty", "raises"])
def test_retract_ladder_falls_back_to_the_servo_retreat(release, refusal):
    if refusal == "raises":
        def plan(**_kw):
            raise RuntimeError("scripted: planner down")
    elif refusal == "empty":
        plan = {"planned": True, "trajectory": {"waypoints": []}}
    else:
        plan = {"planned": False}
    ctx = _release_ctx(plan)
    out = release.run(ctx, _pose(0.5), retract_m=0.12, retract_planner=True, retract_ladder_m=[0.3, 0.2, 0.1])
    assert [call.tool for call in ctx.calls] == [
        "robot.open_gripper", "motion.plan_linear", "motion.plan_linear", "motion.plan_linear",
        "robot.go_to_pose_cartesian", "robot.wait_steps",
    ]
    # The servo goes where it always went: retract_m, not a rung.
    assert ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs == {"pose": _pose(0.5 + 0.12)}
    assert out["retreat_pose"] == _pose(0.5 + 0.12)
    assert out["release_report"]["retreat_mode"] == "cartesian"
    assert out["release_report"]["retreat_distance_m"] == pytest.approx(0.12)


def test_retract_rungs_keep_the_extent_floor_axis_reversal_and_arm(release):
    ctx = _release_ctx({"planned": False})
    attached = {"frame": "tcp", "arm_id": 1,
                "spheres": [{"center": {"x": 0.0, "y": 0.0, "z": -0.18}, "radius": 0.02}]}  # extent 0.20
    release.run(ctx, _pose(0.5), retreat_axis={"x": 1.0, "y": 0.0, "z": 0.0}, attached_object=attached,
                relation="shaft_into_aperture", retract_planner=True)
    # Both rungs floor to 0.21 m: one line is planned, not the same one twice.
    (plan,) = ctx.calls_to("motion.plan_linear")
    assert plan.kwargs["arm_id"] == 1
    assert plan.kwargs["end"]["position"]["x"] == pytest.approx(0.4 - 0.21)
    assert plan.kwargs["end"]["position"]["z"] == pytest.approx(0.5 + 0.5 * 0.21)
    assert ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"] == plan.kwargs["end"]
    assert ctx.calls_to("robot.open_gripper")[0].kwargs == {"settle_steps": 80, "arm_id": 1}


# ---------------------------------------------------------------------------
# Behaviour: time-scaled, arrival-checked insertion
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def execute(modules):
    return modules[EXECUTE][1]


def _execute_ctx(ee_poses, planned=None) -> FakeContext:
    return FakeContext({
        "robot.get_ee_pose": [{"pose": p} for p in ee_poses],
        "motion.plan_to_pose": planned or _TRAJECTORY,
        "motion.plan_linear": planned or _TRAJECTORY,
        "robot.execute_trajectory": None,
        "robot.wait_steps": None,
    })


def test_time_scale_resamples_the_planned_trajectory(execute):
    planned = {"planned": True, "trajectory": {"joint_names": ["a", "b"], "waypoints": [
        {"positions": [0.0, 0.0]}, {"positions": [1.0, 2.0]}, {"positions": [2.0, 2.0]}]}}
    ctx = _execute_ctx([_pose(0.8)], planned)
    execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_joint"}), time_scale=3.0)
    sent = ctx.calls_to("robot.execute_trajectory")[0].kwargs["trajectory"]
    assert sent["joint_names"] == ["a", "b"]
    rows = [w["positions"] for w in sent["waypoints"]]
    assert len(rows) == 7
    assert rows[0] == [0.0, 0.0] and rows[3] == pytest.approx([1.0, 2.0]) and rows[-1] == pytest.approx([2.0, 2.0])
    assert rows[1] == pytest.approx([1 / 3, 2 / 3])
    assert planned["trajectory"]["waypoints"][1] == {"positions": [1.0, 2.0]}  # the plan itself is untouched


def test_verify_arrival_settles_and_reads_the_tcp(execute):
    ctx = _execute_ctx([_pose(0.8), _pose(0.503)])  # 3 mm short: inside 4 mm
    out = execute.run(ctx, _plan({"pose": _pose(0.5), "mode": "planned_linear"}), verify_arrival=True)
    assert [(c.tool, c.kwargs.get("steps")) for c in ctx.calls] == [
        ("robot.get_ee_pose", None), ("motion.plan_linear", None), ("robot.execute_trajectory", None),
        ("robot.wait_steps", 40), ("robot.get_ee_pose", None),
    ]
    report = out["waypoint_reports"][0]
    assert report["arrival_position_error_m"] == pytest.approx(0.003)
    assert report["arrival_rotation_error_deg"] == pytest.approx(0.0, abs=1e-6)


def test_verify_arrival_prohibits_release_short_of_the_target(execute):
    plan = _plan({"pose": _pose(0.5), "mode": "planned_joint"}, {"pose": _pose(0.4), "mode": "planned_linear"})
    ctx = _execute_ctx([_pose(0.8), _pose(0.566)])  # the recorded 66 mm high release
    with pytest.raises(RuntimeError, match="release prohibited"):
        execute.run(ctx, plan, verify_arrival=True)
    assert ctx.call_count("robot.execute_trajectory") == 1
    assert ctx.call_count("motion.plan_linear") == 0

    half = math.radians(3.0) / 2  # a 3 degree twist at the right position
    twisted = {"w": math.cos(half), "x": 0.0, "y": 0.0, "z": math.sin(half)}
    ctx = _execute_ctx([_pose(0.8), _pose(0.5, rotation=twisted)])
    with pytest.raises(RuntimeError, match="3.0 deg; release prohibited"):
        execute.run(ctx, plan, verify_arrival=True)
    # A profile can widen the gate.
    ctx = _execute_ctx([_pose(0.8), _pose(0.5, rotation=twisted), _pose(0.4)])
    execute.run(ctx, plan, verify_arrival=True, contact_profile={"arrival_rotation_tolerance_deg": 5.0})
