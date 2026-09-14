"""The sharps-disposal v2 grasp fold: opt-in, proven against the scripts before it.

RoboSimStudio's ``sharps_disposal/gap_perception_v2`` (sweep s8) grew a
packed-tray grasp and a look-alike-proof grasp check as graph-local scripts.
The fold put them into ``grasping-direct-ik`` (``refine_top_down_grasp``,
``execute_grasp_align``) and ``verifying-grasps`` (``verify_grasp``) behind
inputs that default to the old behaviour.

Two halves, as in ``test_promotion_is_behaviour_preserving.py``:

- **Parity.** Each script as it was at ``752dfbd`` and as it is now, driven
  through a recording :class:`gap.testing.FakeContext` over an input grid with
  default parameters: the full sequence of tool calls (names and keyword
  values), the outputs the old script returned, and any raised error must
  match exactly. The outputs may only have grown by the declared additive keys.
- **Behaviour.** What each new input does when a caller asks for it.
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
from gap.testing import FakeContext
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE_REF = "752dfbd"

REFINE = "skills/grasping-direct-ik/scripts/refine_top_down_grasp.py"
EXECUTE = "skills/grasping-direct-ik/scripts/execute_grasp_align.py"
VERIFY = "skills/verifying-grasps/scripts/verify_grasp.py"

ADDITIVE = {
    REFINE: {"object_width_m", "preclose_width_m", "grasp_station"},
    EXECUTE: {"commanded_pose", "commanded_grasp_pose", "grasp_final_pose", "half_turn_used", "bias_corrections",
              "descent_planned"},
    VERIFY: set(),
}


def _load(source: str, name: str):
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _at_ref(rel: str) -> str:
    done = subprocess.run(
        ["git", "show", f"{BASE_REF}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        pytest.skip(f"{BASE_REF} is not in this checkout: {done.stderr.strip()}")
    return done.stdout


@pytest.fixture(scope="module")
def pair():
    cache: dict[str, tuple] = {}

    def get(rel: str):
        if rel not in cache:
            stem = Path(rel).stem
            cache[rel] = (_load(_at_ref(rel), f"{stem}_before"), _load((ROOT / rel).read_text(), f"{stem}_after"))
        return cache[rel]

    return get


def _plain(value):
    if isinstance(value, np.ndarray):
        return {"ndarray": list(value.shape), "dtype": str(value.dtype), "data": value.tolist()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _drive(module, responses, *args, **kwargs):
    ctx = FakeContext(responses)
    try:
        result, error = module.run(ctx, *args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 -- the error is part of the comparison
        result, error = None, f"{type(exc).__name__}: {exc}"
    calls = [(r.tool, json.dumps(_plain(r.kwargs), sort_keys=True)) for r in ctx.calls]
    return {"calls": calls, "result": _plain(result), "error": error}


def _assert_parity(rel, before, after):
    assert after["calls"] == before["calls"]
    assert after["error"] == before["error"]
    if before["result"] is None:
        assert after["result"] is None
        return
    assert set(after["result"]) - set(before["result"]) <= ADDITIVE[rel]
    assert {k: after["result"][k] for k in before["result"]} == before["result"]


def _pose(x=0.0, y=0.0, z=0.0, rotation=None):
    return {
        "position": {"x": float(x), "y": float(y), "z": float(z)},
        "rotation": dict(rotation or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
    }


def _quat(r: Rotation) -> dict[str, float]:
    q = r.as_quat()
    return {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])}


def _obb(center, extent, yaw_deg=0.0):
    return {
        "center": dict(zip("xyz", map(float, center), strict=True)),
        "extent": dict(zip("xyz", map(float, extent), strict=True)),
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        if yaw_deg == 0.0
        else {k: v for k, v in _quat(Rotation.from_euler("z", yaw_deg, degrees=True)).items()},
    }


# ---------------------------------------------------------------------------
# refine_top_down_grasp
# ---------------------------------------------------------------------------


def _gripper(stated=True, reach=0.045):
    return {
        "finger": {"stated": stated, "reach_m": reach, "clearance_m": 0.004},
        "approach_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
        "close_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
        "span_m": 0.085,
        "open_footprint_m": 0.104,
    }


def _refine_responses(gripper):
    def frame(approach, close_heading_deg):
        return {"rotation": _quat(Rotation.from_euler("zx", [close_heading_deg, 180.0], degrees=True))}

    return {"robot.describe_gripper": gripper, "robot.grasp_frame": frame}


REFINE_GRID = list(
    itertools.product(
        [0.0, 30.0, 90.0, 137.0],
        [(0.10, 0.02, 0.01), (0.02, 0.06, 0.03)],
        [0.01, 0.10],
        [0.01, 0.20],
        [True, False],
    )
)


@pytest.mark.parametrize("yaw,extent,grasp_z,center_z,stated", REFINE_GRID)
def test_refine_defaults_are_the_script_before_the_fold(pair, yaw, extent, grasp_z, center_z, stated):
    before, after = pair(REFINE)
    obb = _obb((0.4, 0.1, center_z), extent, yaw)
    grasp = _pose(0.41, 0.12, grasp_z, {"w": 0.0, "x": 0.96, "y": 0.28, "z": 0.0})
    got = [_drive(m, _refine_responses(_gripper(stated)), grasp, obb) for m in (before, after)]
    _assert_parity(REFINE, *got)


BAR = _obb((0.4, 0.1, 0.01), (0.10, 0.02, 0.01))  # along x, support at z=0
FINGERTIP_2F85 = [
    [0.0131, 0.0179], [0.0176, 0.0178], [0.0223, 0.0176], [0.0268, 0.0173], [0.0309, 0.0170],
    [0.0355, 0.0165], [0.0436, 0.0155], [0.0502, 0.0144], [0.0576, 0.0129], [0.0648, 0.0112],
    [0.0707, 0.0095], [0.0766, 0.0077], [0.0818, 0.0059],
]


@pytest.fixture(scope="module")
def refine(pair):
    return pair(REFINE)[1]


def _run_refine(refine, **kwargs):
    ctx = FakeContext(_refine_responses(_gripper()))
    out = refine.run(ctx, _pose(0.4, 0.1, 0.01), BAR, **kwargs)
    # Every option is pure math: the calls are the two the script always made.
    assert [c.tool for c in ctx.calls] == ["robot.describe_gripper", "robot.grasp_frame"]
    return out


def test_refine_default_outputs_say_open_descent_at_the_obb_width(refine):
    out = _run_refine(refine)
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.049)
    assert out["preclose_width_m"] == 0.0
    assert out["object_width_m"] == pytest.approx(0.04)
    assert out["grasp_station"] == "candidate"


def test_refine_support_z_replaces_the_obb_bottom_in_the_floor(refine):
    out = _run_refine(refine, support_z=0.03)
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.03 + 0.045 + 0.004)


@pytest.mark.parametrize(
    "clearance,band,expected",
    [
        (None, None, 0.030),  # nothing measured: the widest descent jaw
        (0.020, None, 2.0 * (0.020 - 0.0095 - 0.002)),  # band from the hand: (0.104 - 0.085) / 2
        (0.020, 0.0085, 2.0 * (0.020 - 0.0085 - 0.002)),  # band pinned
        (0.012, None, 0.0107 + 0.006),  # tight: clipped up to object + margin
        (0.100, None, 0.030),  # roomy: clipped down to the widest
    ],
)
def test_refine_preclose_width_from_the_measured_clearance(refine, clearance, band, expected):
    out = _run_refine(
        refine, object_width_m=0.0107, preclose_max_width_m=0.030, jaw_clearance_m=clearance,
        finger_band_past_jaw_m=band,
    )
    assert out["preclose_width_m"] == pytest.approx(expected)
    # No fingertip table: the height is left alone.
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.049)


def test_refine_fingertip_drop_raises_the_tcp_only_for_a_preclosed_descent(refine):
    out = _run_refine(
        refine, object_width_m=0.0107, preclose_max_width_m=0.030, fingertip_beyond_tcp_m=FINGERTIP_2F85
    )
    gaps, beyond = zip(*FINGERTIP_2F85, strict=True)
    drop = float(np.interp(0.030, gaps, beyond)) - 0.0059
    assert drop == pytest.approx(0.0112, abs=2e-4)  # +17.1 mm at 30 mm vs +5.9 mm open
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.049 + drop)
    # The same table without a pre-close is not a drop.
    out = _run_refine(refine, fingertip_beyond_tcp_m=FINGERTIP_2F85)
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.049)


def _barrel_cloud(radius=0.007, n=120, flange=20, center=(0.4, 0.1, 0.01)):
    # Evenly spaced round the ring, so the cloud median sits on the axis and the
    # test measures the percentile rather than sampling noise in the origin.
    along = np.r_[np.linspace(-0.06, 0.06, n), np.linspace(0.04, 0.05, flange)]
    angle = np.r_[np.linspace(0.0, 2.0 * np.pi, n, endpoint=False), np.linspace(0.0, 2.0 * np.pi, flange, endpoint=False)]
    r = np.r_[np.full(n, radius), np.full(flange, 0.012)]
    return {
        "points": np.column_stack(
            [center[0] + along, center[1] + r * np.cos(angle), center[2] + r * np.sin(angle)]
        )
    }


def test_refine_object_width_from_the_cloud_transverse_percentile(refine):
    cloud = _barrel_cloud()
    out = _run_refine(refine, target_cloud=cloud, width_percentile=35, width_range_m=[0.006, 0.025])
    # The 20 flange points (14 %) sit above the 35th percentile.
    assert out["object_width_m"] == pytest.approx(0.014, abs=2e-4)
    out = _run_refine(
        refine, target_cloud=cloud, width_percentile=35, width_range_m=[0.006, 0.010], object_width_m=0.0107
    )
    assert out["object_width_m"] == pytest.approx(0.0107)
    # A cloud alone, without the percentile, is not an estimate.
    assert _run_refine(refine, target_cloud=cloud)["object_width_m"] == pytest.approx(0.04)


@pytest.mark.parametrize("feature_x,expected_x", [(0.33, 0.33 + 0.043), (0.47, 0.47 - 0.043)])
def test_refine_station_is_offset_from_the_end_feature_into_the_body(refine, feature_x, expected_x):
    out = _run_refine(refine, end_feature_center={"x": feature_x, "y": 0.103, "z": 0.01}, station_offset_m=0.043)
    assert out["grasp_station"] == "end_feature_offset"
    assert out["grasp_pose"]["position"]["x"] == pytest.approx(expected_x)
    # The feature is projected onto the axis through the OBB centre.
    assert out["grasp_pose"]["position"]["y"] == pytest.approx(0.1)
    assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.049)


def test_refine_station_axis_runs_through_the_cloud_median_when_dense(refine):
    cloud = _barrel_cloud(center=(0.4, 0.105, 0.01))
    feature = {"x": 0.33, "y": 0.1, "z": 0.01}
    out = _run_refine(refine, end_feature_center=feature, station_offset_m=0.043, target_cloud=cloud)
    assert out["grasp_pose"]["position"]["y"] == pytest.approx(0.105, abs=1e-3)
    sparse = {"points": cloud["points"][:10]}
    out = _run_refine(refine, end_feature_center=feature, station_offset_m=0.043, target_cloud=sparse)
    assert out["grasp_pose"]["position"]["y"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# execute_grasp_align
# ---------------------------------------------------------------------------


def _yawed(pose, deg):
    out = _pose(**pose["position"])
    out["rotation"] = _quat(Rotation.from_euler("z", deg, degrees=True))
    return out


TARGET = _pose(0.0, 0.0, 0.19)
EXECUTE_REACHES = {
    "arrives": [TARGET],
    "rotated": [_yawed(TARGET, 21.0), TARGET],
    "displaced": [_yawed(_pose(0.08, 0.0, 0.19), 21.0), _yawed(TARGET, 21.0), TARGET],
    "never": [_pose(0.08, 0.0, 0.19)] * 3,
    "short": [_pose(0.0, 0.0, 0.21)],
}
EXECUTE_PROFILES = {
    "none": {},
    "loose": {"grasp_profile": {"position_tolerance_m": 0.01, "angular_tolerance_deg": 20.0}},
    "no_recovery": {"grasp_profile": {"cartesian_recovery_on_verification_failure": False}},
    "keyed": {"target_kind": "rod", "grasp_profiles": [{"source_kind": "rod", "position_tolerance_m": 0.03}]},
    "missing_key": {"target_kind": "cup", "grasp_profiles": [{"source_kind": "rod"}]},
}


@pytest.mark.parametrize(
    "reach,profile,arm_id", list(itertools.product(EXECUTE_REACHES, EXECUTE_PROFILES, [None, 1]))
)
def test_execute_defaults_are_the_script_before_the_fold(pair, reach, profile, arm_id):
    before, after = pair(EXECUTE)

    def responses():
        return {
            "robot.go_to_pose": {},
            "robot.go_to_pose_cartesian": {},
            "robot.get_ee_pose": [{"pose": p} for p in EXECUTE_REACHES[reach]],
        }

    kwargs = dict(EXECUTE_PROFILES[profile])
    if arm_id is not None:
        kwargs["arm_id"] = arm_id
    got = [_drive(m, responses(), TARGET, **kwargs) for m in (before, after)]
    _assert_parity(EXECUTE, *got)


@pytest.fixture(scope="module")
def execute(pair):
    return pair(EXECUTE)[1]


TRAJECTORY = {"waypoints": [{"positions": [0.0] * 7}, {"positions": [1.0] * 7}]}
GRASP = _pose(0.0, 0.0, 0.05)


def _plan(error, planned=True):
    return {"planned": planned, "trajectory": TRAJECTORY if planned else None, "rotation_error_rad": error}


def test_execute_half_turn_takes_the_better_yaw_and_turns_the_grasp_with_it(execute):
    turned = _yawed(TARGET, 180.0)
    ctx = FakeContext({
        "motion.plan_joint": [_plan(0.80), _plan(0.01)],
        "robot.execute_trajectory": {},
        "robot.wait_steps": {},
        "robot.get_ee_pose": [{"pose": turned}],
    })
    out = execute.run(ctx, TARGET, half_turn_tolerance_rad=0.05)
    assert [c.tool for c in ctx.calls] == [
        "motion.plan_joint", "motion.plan_joint", "robot.execute_trajectory", "robot.wait_steps", "robot.get_ee_pose",
    ]
    plans = ctx.calls_to("motion.plan_joint")
    assert plans[0].kwargs == {"pose": TARGET, "orientation": "lock"}
    assert Rotation.from_quat([plans[1].kwargs["pose"]["rotation"][k] for k in "xyzw"]).magnitude() == (
        pytest.approx(np.pi)
    )
    sent = ctx.calls_to("robot.execute_trajectory")[0].kwargs
    assert len(sent["trajectory"]["waypoints"]) == 3  # slowed 2x
    assert sent["tolerance"] == 0.002 and sent["max_steps_per_waypoint"] == 180
    assert out["half_turn_used"] is True
    assert out["commanded_pose"] == plans[1].kwargs["pose"]
    assert out["position_error_m"] == 0.0


def test_execute_half_turn_keeps_the_requested_yaw_when_the_turn_is_no_better(execute):
    for alternative in (_plan(0.90), _plan(0.01, planned=False)):
        ctx = FakeContext({
            "motion.plan_joint": [_plan(0.30), alternative],
            "robot.execute_trajectory": {},
            "robot.wait_steps": {},
            "robot.get_ee_pose": {"pose": TARGET},
        })
        out = execute.run(ctx, TARGET, half_turn_tolerance_rad=0.05)
        assert out["half_turn_used"] is False and out["commanded_pose"] == TARGET
    ctx = FakeContext({"motion.plan_joint": _plan(0.01), "robot.execute_trajectory": {}, "robot.wait_steps": {},
                       "robot.get_ee_pose": {"pose": TARGET}})
    execute.run(ctx, TARGET, half_turn_tolerance_rad=0.05)
    assert ctx.call_count("motion.plan_joint") == 1


def test_execute_half_turn_refuses_an_unplanned_approach(execute):
    ctx = FakeContext({"motion.plan_joint": _plan(0.0, planned=False)})
    with pytest.raises(RuntimeError, match="pregrasp approach unavailable"):
        execute.run(ctx, TARGET, half_turn_tolerance_rad=0.05)


def test_execute_half_turn_turns_the_descent_target_too(execute):
    ctx = FakeContext({
        "motion.plan_joint": [_plan(0.80), _plan(0.01)],
        "robot.execute_trajectory": {},
        "robot.wait_steps": {},
        "robot.go_to_pose_cartesian": {},
        "robot.get_ee_pose": {"pose": TARGET},
    })
    out = execute.run(ctx, TARGET, half_turn_tolerance_rad=0.05, grasp_pose=GRASP)
    descended = ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]
    assert descended == out["commanded_grasp_pose"]
    assert descended["position"] == GRASP["position"]
    assert descended["rotation"] == out["commanded_pose"]["rotation"]


ENVELOPE = {"across_m": 0.0045, "along_m": 0.010, "vertical_m": 0.005, "rotation_rad": 0.04}


def _descend_ctx(readings):
    return FakeContext({
        "robot.go_to_pose": {},
        "robot.go_to_pose_cartesian": {},
        "robot.wait_steps": {},
        "robot.set_grip": {},
        "robot.get_ee_pose": [{"pose": TARGET}] + [{"pose": p} for p in readings],
    })


def test_execute_preclose_descend_correct_bias_then_close(execute):
    ctx = _descend_ctx([_pose(0.006, 0.0, 0.05), GRASP])
    out = execute.run(
        ctx, TARGET, grasp_pose=GRASP, preclose_width_m=0.02, object_width_m=0.0107,
        pad_envelope={**ENVELOPE, "max_corrections": 2},
    )
    assert [c.tool for c in ctx.calls] == [
        "robot.go_to_pose", "robot.get_ee_pose",
        "robot.set_grip", "robot.go_to_pose_cartesian", "robot.wait_steps", "robot.get_ee_pose",
        "robot.go_to_pose_cartesian", "robot.wait_steps", "robot.get_ee_pose",
        "robot.set_grip",
    ]
    grips = ctx.calls_to("robot.set_grip")
    assert grips[0].kwargs == {"width_m": 0.02, "ramp_steps": 40, "settle_steps": 40}
    assert grips[1].kwargs == {"object_width_m": 0.0107, "squeeze_m": 0.002, "ramp_steps": 100, "settle_steps": 120}
    corrected = ctx.calls_to("robot.go_to_pose_cartesian")[1].kwargs["pose"]
    assert corrected["position"]["x"] == pytest.approx(-0.006)
    assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 30}
    assert out["bias_corrections"] == 1 and out["grasp_final_pose"] == GRASP


@pytest.mark.parametrize(
    "offset,across_axis,passes",
    [
        ((0.0, 0.008, 0.0), "x", True),  # along the object: room
        ((0.008, 0.0, 0.0), "x", False),  # across the jaws: none
        ((0.0, 0.008, 0.0), "y", False),
        ((0.008, 0.0, 0.0), "y", True),
        ((0.0, 0.0, 0.006), "x", False),  # above the pad depth
    ],
)
def test_execute_pad_envelope_is_anisotropic_in_the_tool_frame(execute, offset, across_axis, passes):
    ctx = _descend_ctx([_pose(offset[0], offset[1], 0.05 + offset[2])])
    kwargs = dict(grasp_pose=GRASP, object_width_m=0.0107, pad_envelope={**ENVELOPE, "across_axis": across_axis})
    if passes:
        execute.run(ctx, TARGET, **kwargs)
        assert ctx.call_count("robot.set_grip") == 1
    else:
        with pytest.raises(RuntimeError, match="outside pad envelope"):
            execute.run(ctx, TARGET, **kwargs)
        assert ctx.call_count("robot.set_grip") == 0


def test_execute_bias_correction_is_gated_by_the_largest_correction(execute):
    ctx = _descend_ctx([_pose(0.020, 0.0, 0.05)])
    with pytest.raises(RuntimeError, match="outside pad envelope"):
        execute.run(ctx, TARGET, grasp_pose=GRASP, pad_envelope={**ENVELOPE, "max_corrections": 2})
    assert ctx.call_count("robot.go_to_pose_cartesian") == 1  # the descent, no correction


def test_execute_descent_options_need_a_grasp_pose(execute):
    with pytest.raises(ValueError, match="give grasp_pose"):
        execute.run(FakeContext({}), TARGET, object_width_m=0.01)
    with pytest.raises(ValueError, match="give grasp_pose"):
        execute.run(FakeContext({}), TARGET, descent_planner=True)


def _planned_ctx(plan_linear, readings):
    return FakeContext({
        "robot.go_to_pose": {},
        "motion.plan_linear": plan_linear,
        "robot.execute_trajectory": {},
        "robot.go_to_pose_cartesian": {},
        "robot.wait_steps": {},
        "robot.get_ee_pose": [{"pose": TARGET}] + [{"pose": p} for p in readings],
    })


def test_execute_descent_planner_flies_the_planned_line_before_the_cartesian_finish(execute):
    ctx = _planned_ctx({"planned": True, "trajectory": TRAJECTORY}, [_pose(0.0, 0.0, 0.056), GRASP])
    out = execute.run(ctx, TARGET, grasp_pose=GRASP, descent_planner=True)
    assert [c.tool for c in ctx.calls] == [
        "robot.go_to_pose", "robot.get_ee_pose",
        "motion.plan_linear", "robot.execute_trajectory", "robot.wait_steps", "robot.get_ee_pose",
        "robot.go_to_pose_cartesian", "robot.wait_steps", "robot.get_ee_pose",
    ]
    # The graph's planned_linear, kwarg for kwarg.
    assert ctx.calls_to("motion.plan_linear")[0].kwargs == {
        "end": GRASP, "orientation": "lock", "world_config": None, "attached_object": None,
        "allow_goal_contact": False, "allow_start_contact": False,
    }
    sent = ctx.calls_to("robot.execute_trajectory")[0].kwargs
    assert len(sent["trajectory"]["waypoints"]) == 4  # slowed 3x
    assert sent["tolerance"] == 0.002 and sent["max_steps_per_waypoint"] == 180
    assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 40}
    assert out["descent_planned"] is True and out["grasp_final_pose"] == GRASP


def _raises(**_kwargs):
    raise RuntimeError("scripted: no IK backend")


@pytest.mark.parametrize(
    "plan_linear",
    [
        {"planned": False, "trajectory": None, "reason": "collision"},
        {"planned": True, "trajectory": {"waypoints": []}},
        _raises,
    ],
    ids=["refused", "empty", "raises"],
)
def test_execute_descent_planner_falls_back_to_the_cartesian_descent(execute, plan_linear):
    ctx = _planned_ctx(plan_linear, [GRASP])
    out = execute.run(ctx, TARGET, grasp_pose=GRASP, descent_planner=True, arm_id=1)
    assert [c.tool for c in ctx.calls] == [
        "robot.go_to_pose", "robot.get_ee_pose", "motion.plan_linear",
        "robot.go_to_pose_cartesian", "robot.wait_steps", "robot.get_ee_pose",
    ]
    assert ctx.calls_to("motion.plan_linear")[0].kwargs["arm_id"] == 1
    assert out["descent_planned"] is False and out["grasp_final_pose"] == GRASP


def test_execute_descent_planner_refuses_a_planned_line_that_was_not_tracked(execute):
    ctx = _planned_ctx({"planned": True, "trajectory": TRAJECTORY}, [_pose(0.0, 0.0, 0.07)])
    with pytest.raises(RuntimeError, match="planned descent tracking failed"):
        execute.run(ctx, TARGET, grasp_pose=GRASP, descent_planner=True)
    assert ctx.call_count("robot.go_to_pose_cartesian") == 0


@pytest.mark.parametrize(
    "arrival,kwargs,passes",
    [
        (_pose(0.005, 0.0, 0.19), {}, True),  # inside the 15 mm profile default
        (_pose(0.005, 0.0, 0.19), {"approach_check_tolerance_m": 0.003}, False),
        (_pose(0.002, 0.0, 0.19), {"approach_check_tolerance_m": 0.003}, True),
        (_yawed(TARGET, 3.0), {}, True),  # inside 12 deg
        (_yawed(TARGET, 3.0), {"approach_check_tolerance_rad": 0.035}, False),
        (_yawed(TARGET, 1.5), {"approach_check_tolerance_m": 0.003, "approach_check_tolerance_rad": 0.035}, True),
    ],
)
def test_execute_approach_check_holds_the_arrival_to_a_tighter_bar(execute, arrival, kwargs, passes):
    ctx = FakeContext({"robot.go_to_pose": {}, "robot.get_ee_pose": {"pose": arrival}})
    if passes:
        execute.run(ctx, TARGET, **kwargs)
    else:
        with pytest.raises(RuntimeError, match="outside the approach check"):
            execute.run(ctx, TARGET, **kwargs)
    # The check reads nothing new.
    assert [c.tool for c in ctx.calls] == ["robot.go_to_pose", "robot.get_ee_pose"]


# ---------------------------------------------------------------------------
# verify_grasp
# ---------------------------------------------------------------------------


def _blob(center, n=40, spread=0.004, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.uniform(-spread, spread, size=(n, 3))


def _frame(name, fill):
    return {
        "name": name,
        "rgb": np.full((8, 8, 3), fill, dtype=np.uint8),
        "depth": np.ones((8, 8)),
        "intrinsics": np.eye(3),
        "pose": _pose(),
    }


def _verify_responses(wrist, overhead=(), *, layout="both", surface_z=0.0, ee_z=0.30):
    """``wrist``/``overhead``: ``[(score, points), ...]`` in the order SAM returns them."""
    clouds: list[np.ndarray] = []
    by_image: dict[int, list[tuple[float, int]]] = {}
    cameras = []
    for name, fill, masks in (("eye_in_hand_0", 0, wrist), ("overhead", 1, overhead)):
        if (name == "overhead" and layout == "no_overhead") or (name.startswith("eye") and layout == "no_wrist"):
            continue
        entries = []
        for score, points in masks:
            clouds.append(np.asarray(points, dtype=np.float32).reshape(-1, 3))
            entries.append((score, len(clouds) - 1))
        by_image[fill] = entries
        cameras.append(_frame(name, fill))

    def segment(image, query, max_results):
        entries = by_image[int(image.flat[0])]
        if max_results:
            entries = entries[:max_results]
        return {
            "masks": [np.full((8, 8), index, dtype=np.uint8) for _, index in entries],
            "scores": [score for score, _ in entries],
        }

    def backproject(mask, depth, intrinsics, camera_pose):
        return {"points": {"points": clouds[int(mask.flat[0])]}}

    return {
        "robot.get_ee_pose": {"pose": _pose(0.5, 0.0, ee_z)},
        "robot.go_to_pose_cartesian": {},
        "robot.get_observation": {
            "cameras": {c["name"]: c for c in cameras} if layout == "dict" else cameras
        },
        "sam3.segment_text": segment,
        "geometry.mask_to_world_points": backproject,
        "robot.describe_workspace": {"surface_z": surface_z},
    }


HELD = (0.5, 0.0, 0.30)
TRAY = [(0.052, _blob((0.70, 0.20, 0.02), seed=1)), (0.050, _blob((0.30, 0.25, 0.02), seed=2)),
        (0.041, _blob((0.65, -0.20, 0.02), seed=3))]
VERIFY_SCENES = {
    "held": ([(0.02, _blob(HELD))], [(0.02, _blob(HELD))]),
    "on_table": ([(0.02, _blob((0.5, 0.0, 0.02)))], [(0.02, _blob((0.5, 0.0, 0.02)))]),
    "far": ([(0.02, _blob((0.5, 0.0, 0.09)))], []),
    "sparse": ([(0.02, _blob(HELD, n=5))], [(0.02, _blob(HELD))]),
    "wrist_below_floor": ([(0.001, _blob(HELD))], [(0.02, _blob(HELD))]),
    "unseen": ([(0.001, _blob(HELD))], [(0.001, _blob(HELD))]),
    "no_masks": ([], []),
    "look_alikes": ([*TRAY, (0.040, _blob(HELD, seed=4))], []),
}


@pytest.mark.parametrize(
    "scene,layout,marker,lift",
    list(itertools.product(VERIFY_SCENES, ["both", "dict", "no_overhead", "no_wrist"], ["", "red cap"], [0.04, 0.0])),
)
def test_verify_defaults_are_the_script_before_the_fold(pair, scene, layout, marker, lift):
    before, after = pair(VERIFY)
    wrist, overhead = VERIFY_SCENES[scene]
    got = [
        _drive(m, _verify_responses(wrist, overhead, layout=layout), "syringe", marker_description=marker, lift_m=lift)
        for m in (before, after)
    ]
    _assert_parity(VERIFY, *got)


@pytest.fixture(scope="module")
def verify(pair):
    return pair(VERIFY)[1]


def test_verify_nearest_to_hand_finds_the_held_one_behind_brighter_look_alikes(verify):
    wrist = VERIFY_SCENES["look_alikes"][0]
    # Today's check: the top mask is a tray syringe on the table.
    assert verify.run(FakeContext(_verify_responses(wrist)), "syringe", lift_m=0.0)["route"] == "not_held"
    # All masks considered, but only three returned: the held one is fourth.
    out = verify.run(FakeContext(_verify_responses(wrist)), "syringe", lift_m=0.0, nearest_to_hand=True)
    assert out["route"] == "not_held" and "nearest candidate" in out["reason"]
    ctx = FakeContext(_verify_responses(wrist))
    out = verify.run(ctx, "syringe", lift_m=0.0, nearest_to_hand=True, max_masks_per_query=0)
    assert out["route"] == "verified" and out["camera"] == "eye_in_hand_0"
    assert out["observed_center"]["z"] == pytest.approx(0.30, abs=0.005)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["max_results"] == 0
    assert ctx.call_count("geometry.mask_to_world_points") == 4
    assert ctx.call_count("robot.describe_workspace") == 1


def test_verify_nearest_to_hand_keeps_the_query_and_camera_order(verify):
    ctx = FakeContext(_verify_responses([], [(0.02, _blob(HELD))]))
    out = verify.run(ctx, "syringe", marker_description="red cap", nearest_to_hand=True, max_masks_per_query=0)
    assert out["route"] == "verified" and out["camera"] == "overhead"
    assert [c.kwargs["query"] for c in ctx.calls_to("sam3.segment_text")] == ["syringe", "red cap", "syringe"]


def test_verify_nearest_to_hand_routes_not_held_with_the_same_reasons(verify):
    ctx = FakeContext(_verify_responses(*VERIFY_SCENES["unseen"]))
    out = verify.run(ctx, "syringe", nearest_to_hand=True)
    assert out["route"] == "not_held" and "no camera sees" in out["reason"]
    assert ctx.call_count("robot.describe_workspace") == 0
    ctx = FakeContext(_verify_responses([(0.02, _blob(HELD, n=5))]))
    out = verify.run(ctx, "syringe", nearest_to_hand=True)
    assert "too few valid depth points" in out["reason"] and out["point_count"] == 5
    assert ctx.call_count("robot.describe_workspace") == 0


def _arm_cloud():
    rng = np.random.default_rng(7)
    return np.column_stack([rng.uniform(0.30, 0.70, 200), rng.uniform(-0.01, 0.01, 200), np.full(200, 0.34)])


def test_verify_size_gate_keeps_the_arm_from_passing_as_the_held_object(verify):
    wrist = [(0.9, _arm_cloud()), (0.04, _blob((0.5, 0.0, 0.33), seed=5))]
    ungated = verify.run(FakeContext(_verify_responses(wrist)), "syringe", nearest_to_hand=True)
    assert ungated["route"] == "verified" and ungated["point_count"] == 200  # the arm
    gated = verify.run(
        FakeContext(_verify_responses(wrist)), "syringe", nearest_to_hand=True, max_span_m=0.25, max_width_m=0.06
    )
    assert gated["route"] == "verified" and gated["point_count"] == 40
    top1 = verify.run(FakeContext(_verify_responses(wrist)), "syringe", max_span_m=0.25)
    assert top1["route"] == "not_held" and "larger than one object" in top1["reason"]
    only_arm = verify.run(
        FakeContext(_verify_responses([(0.9, _arm_cloud())])), "syringe", nearest_to_hand=True, max_span_m=0.25
    )
    assert only_arm["route"] == "not_held" and "larger than one object" in only_arm["reason"]


def test_verify_declares_the_new_inputs_with_behaviour_preserving_defaults(skills_registry):
    inputs = skills_registry.get("verifying-grasps").canonical_scripts["verify_grasp"].schema.inputs
    assert inputs["max_masks_per_query"].default == 3
    assert inputs["nearest_to_hand"].default is False
    assert inputs["max_span_m"].default is None and inputs["max_width_m"].default is None
    direct = skills_registry.get("grasping-direct-ik")
    assert {"robot.set_grip", "robot.wait_steps", "motion.plan_joint", "motion.plan_linear"} <= set(
        direct.meta.requires.connector
    )
    assert {
        "robot.set_grip", "robot.wait_steps", "motion.plan_joint", "motion.plan_linear", "robot.execute_trajectory"
    } <= set(direct.meta.allowed_tools)
    refine_inputs = direct.canonical_scripts["refine_top_down_grasp"].schema.inputs
    assert refine_inputs["fingertip_beyond_tcp_m"].default is None
    assert refine_inputs["support_z"].default is None
