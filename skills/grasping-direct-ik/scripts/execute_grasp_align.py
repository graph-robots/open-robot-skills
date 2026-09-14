"""Reach a top-down pre-grasp with a planned joint trajectory and verify it.

``robot.go_to_pose`` may silently retain the previous joint solution when a
seed starts from an awkward elbow configuration. The final pre-grasp segment
is a short, obstacle-free top-down move, so a miss is recovered with the
calibrated Cartesian controller before the grasp is declared unreachable --
and only a miss: an arrival inside tolerance never enters the recovery path.

The verification tolerances come from the selected ``grasp_profile`` (or a
``grasp_profiles`` table keyed by ``target_kind``); without one they are the
defaults below. ``arm_id`` names the hand on a bimanual cell and is threaded
as an ABSENT keyword when unset, so a single-arm graph's recorded calls carry
no ``arm_id``.

**Opt-in: a checked descend and close.** Everything below is off by default
and every new recorded call sits behind its input, so a graph that binds only
``pose`` gets exactly the calls it always got. They came from the sharps
disposal graph's ``grasp_checked.py`` (sweep s8: 16/30 cluttered syringe trays
fully solved), where a thin barrel is taken between close neighbours:

- ``half_turn_tolerance_rad``: plan the pre-grasp with ``motion.plan_joint``
  (``orientation="lock"``) and execute that trajectory instead of
  ``robot.go_to_pose``. When the plan's ``rotation_error_rad`` exceeds the
  tolerance, plan the pose turned by pi about world z as well and keep the
  better of the two: a parallel jaw grips the same way at yaw and yaw + pi.
  Sweep s6 ``t20``: the pre-grasp planned with 0.80 rad of rotation error,
  executed 46 deg off, and the grasp was refused. ``grasp_pose`` turns with it.
  The graph used 0.05 rad.
- ``approach_check_tolerance_m`` / ``approach_check_tolerance_rad``: after the
  profile verification (and any recovery) passes, hold the same TCP reading to
  a tighter bar -- position error and full rotation error -- and raise outside
  it, before anything descends toward contact. No extra call. The graph checked
  every executed approach at 3 mm and 0.035 rad (its ``checked_execute``); the
  profile default is 15 mm and 12 deg.
- ``grasp_pose``: after the pre-grasp is verified, descend to it with
  ``robot.go_to_pose_cartesian``, settle ``settle_steps``, and read the TCP
  back. The sub-options below apply to this stage and need it:

  - ``descent_planner``: first fly an orientation-locked
    ``motion.plan_linear`` line to the grasp (slowed 3x through
    ``robot.execute_trajectory``, then ``robot.wait_steps(40)``), and require
    the TCP within 10 mm and 0.035 rad of it, raising otherwise, as the graph
    did: the global plan gets the hand to the contact neighbourhood and the
    Cartesian contact controller finishes the last millimetres. When the
    planner refuses, returns no waypoints, or raises, the Cartesian descent
    runs alone. ``descent_planned`` says which happened.

  - ``preclose_width_m``: ``robot.set_grip(width_m=...)`` before the descent,
    so the fingertip band does not land on a neighbour (fully open, a 2F-85's
    band spans +/-49 mm; ``tray_clutter_t16``/``t17`` have neighbours 37-49 mm
    away and the hand stopped 9 mm high). ``refine_top_down_grasp`` computes it
    and raises the TCP by the fingertip drop it costs. ``0`` or absent: descend
    open.
  - ``pad_envelope``: the tolerance, in the commanded grasp's tool frame, that
    the TCP must sit inside before the jaws close -- ``across_m`` along the
    closing axis (``across_axis``, ``"x"`` or ``"y"`` of the commanded TCP),
    ``along_m`` along the other horizontal tool axis, ``vertical_m`` along tool
    z, ``rotation_rad`` overall. The jaw gap is sensitive across the closing
    axis and has room along the object and within the pad depth, so a safe
    4 mm station offset along the barrel is not treated as an empty grasp; the
    graph used 4.5 / 10 / 5 mm and 0.04 rad. Outside it, up to
    ``max_corrections`` Cartesian bias corrections are made from the measured
    TCP error (the servo can hold a small bias with the wrist near the table),
    each only while the world error is within ``max_correction_m`` (0.012) and
    ``max_correction_rad`` (0.06); still outside, it raises rather than closing
    on a one-sided pinch. The graph made two.
  - ``object_width_m``: close with ``robot.set_grip(object_width_m=...,
    squeeze_m=...)``, the hand-portable "grip what I measured" -- the pads
    press past the surface by ``squeeze_m`` so the grip is loaded, not swept.

Additive outputs: ``commanded_pose`` (the pre-grasp actually sent, possibly
half-turned), ``commanded_grasp_pose`` / ``grasp_final_pose`` (the descent
target and the TCP read back after it; empty without ``grasp_pose``),
``half_turn_used``, ``bias_corrections`` and ``descent_planned``.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    final_pose: dict[str, Any]
    position_error_m: float
    approach_alignment: float
    closing_alignment: float
    recovery_used: bool
    commanded_pose: dict[str, Any]
    commanded_grasp_pose: dict[str, Any]
    grasp_final_pose: dict[str, Any]
    half_turn_used: bool
    bias_corrections: int
    descent_planned: bool


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _vec(position: dict[str, Any]) -> np.ndarray:
    return np.array([float(position[k]) for k in ("x", "y", "z")], dtype=np.float64)


def _tracking_error(actual: dict[str, Any], wanted: dict[str, Any]) -> tuple[float, float, float]:
    """Return position, approach-axis, and closing-axis agreement."""
    position_error = np.linalg.norm(
        np.array([actual["position"][k] for k in ("x", "y", "z")])
        - np.array([wanted["position"][k] for k in ("x", "y", "z")])
    )
    wanted_rotation = _rotation(wanted)
    actual_rotation = _rotation(actual)
    approach_alignment = float(np.dot(
        wanted_rotation.apply([0.0, 0.0, 1.0]),
        actual_rotation.apply([0.0, 0.0, 1.0]),
    ))
    # A parallel-jaw grasp is invariant to reversing the closing axis.
    closing_alignment = abs(float(np.dot(
        wanted_rotation.apply([1.0, 0.0, 0.0]),
        actual_rotation.apply([1.0, 0.0, 0.0]),
    )))
    return float(position_error), approach_alignment, closing_alignment


def _half_turn(pose: dict[str, Any]) -> dict[str, Any]:
    """The same pose turned by pi about world z: the jaw's other equivalent yaw."""
    q = (Rotation.from_euler("z", np.pi) * _rotation(pose)).as_quat()
    return {
        "position": dict(pose["position"]),
        "rotation": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
    }


def _resample(trajectory: dict[str, Any], scale: float = 2.0) -> dict[str, Any]:
    """Stretch a joint trajectory to ``scale`` times as many waypoints (slower)."""
    rows = np.array([w["positions"] for w in trajectory["waypoints"]], dtype=float)
    src = np.linspace(0, 1, len(rows))
    dst = np.linspace(0, 1, max(2, int((len(rows) - 1) * scale) + 1))
    out = np.column_stack([np.interp(dst, src, rows[:, i]) for i in range(rows.shape[1])])
    return {"waypoints": [{"positions": row.tolist()} for row in out]}


def _profile(
    grasp_profile: dict[str, Any] | None, target_kind: str, grasp_profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """Resolve the selected profile; keyed lists support existing graphs."""
    if grasp_profile is not None:
        return dict(grasp_profile)
    if not target_kind and not grasp_profiles:
        return {}
    profiles = {str(item["source_kind"]): item for item in grasp_profiles}
    if target_kind not in profiles:
        raise ValueError(f"no grasp verification profile declared for profile key {target_kind!r}")
    return dict(profiles[target_kind])


def run(
    ctx: NodeContext,
    pose: dict[str, Any],
    arm_id: int | None = None,
    grasp_profile: dict[str, Any] | None = None,
    target_kind: str = "",
    grasp_profiles: list[dict[str, Any]] | None = None,
    half_turn_tolerance_rad: float | None = None,
    grasp_pose: dict[str, Any] | None = None,
    preclose_width_m: float | None = None,
    pad_envelope: dict[str, Any] | None = None,
    object_width_m: float | None = None,
    squeeze_m: float = 0.002,
    settle_steps: int = 30,
    descent_planner: bool = False,
    approach_check_tolerance_m: float | None = None,
    approach_check_tolerance_rad: float | None = None,
) -> Output:
    profile = _profile(grasp_profile, target_kind, grasp_profiles or [])
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}
    if grasp_pose is None and (
        preclose_width_m is not None or pad_envelope is not None or object_width_m is not None or descent_planner
    ):
        raise ValueError(
            "preclose_width_m, pad_envelope, object_width_m and descent_planner act on the descent: give grasp_pose"
        )

    def where() -> tuple[dict[str, Any], float, float, float]:
        actual = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
        return (actual, *_tracking_error(actual, pose))

    half_turn_used = False
    if half_turn_tolerance_rad is None:
        # The simulator connector has calibrated IK/control for both arms. The
        # optional cuRobo backend exposes trajectories only for its primary chain,
        # so using it here would reject every second-arm pre-grasp even when the
        # same-side robot IK can reach it exactly.
        ctx.tool("robot.go_to_pose", pose=pose, **on_arm)
    else:
        plan = ctx.tool("motion.plan_joint", pose=pose, orientation="lock", **on_arm)
        if float(plan.get("rotation_error_rad", 0.0)) > float(half_turn_tolerance_rad):
            turned = _half_turn(pose)
            alternative = ctx.tool("motion.plan_joint", pose=turned, orientation="lock", **on_arm)
            if alternative.get("planned") and float(alternative.get("rotation_error_rad", 9.0)) < float(
                plan.get("rotation_error_rad", 9.0)
            ):
                plan, pose, half_turn_used = alternative, turned, True
                if grasp_pose is not None:
                    grasp_pose = _half_turn(grasp_pose)
        if not plan.get("planned") or not plan.get("trajectory"):
            raise RuntimeError("pregrasp approach unavailable: motion.plan_joint produced no path")
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=_resample(plan["trajectory"], 2.0),
            tolerance=0.002,
            max_steps_per_waypoint=180,
        )
        ctx.tool("robot.wait_steps", steps=40)
    actual, position_error, approach_alignment, closing_alignment = where()
    tolerance_deg = float(profile.get("angular_tolerance_deg", 12.0))
    angular_tolerance = float(np.cos(np.deg2rad(tolerance_deg)))
    position_tolerance = float(profile.get("position_tolerance_m", 0.015))

    def failed() -> bool:
        return (
            position_error > position_tolerance
            or approach_alignment < angular_tolerance
            or closing_alignment < angular_tolerance
        )

    recovery_used = False
    # Recovery is confined to a pose the plain path would reject: a small
    # residual translation or wrist yaw gets one calibrated Cartesian
    # correction before failure is declared; arrivals stay on their old path.
    recovery_threshold = float(profile.get("cartesian_recovery_threshold_m", 0.030))
    recover = bool(profile.get("cartesian_recovery_on_verification_failure", True))
    if position_error > recovery_threshold and recover:
        # A combined translation + wrist rotation can be locally infeasible
        # even though both components are reachable, especially near the
        # symmetry plane of a dual-arm workspace: translate first while
        # keeping the controller's current wrist solution, then ask for the
        # grasp frame.
        ctx.tool(
            "robot.go_to_pose_cartesian",
            pose={"position": dict(pose["position"]), "rotation": dict(actual["rotation"])},
            **on_arm,
        )
        actual, position_error, approach_alignment, closing_alignment = where()
        recovery_used = True
    if failed() and recover:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm)
        actual, position_error, approach_alignment, closing_alignment = where()
        recovery_used = True
    # Tracking quantisation and the mirrored wrist solution can contribute a
    # few millimetres/degrees without changing the grasp line; twelve degrees
    # stays conservative for a top-down parallel-jaw grasp. The selected grasp
    # region declares how much yaw residual it tolerates, which also admits a
    # mirrored arm's equivalent solution without any object-class assumption.
    if failed():
        raise RuntimeError(
            "pre-grasp tracking error too large: "
            f"{position_error * 1000:.1f} mm, approach_dot={approach_alignment:.3f}, "
            f"closing_dot={closing_alignment:.3f}"
        )
    if approach_check_tolerance_m is not None or approach_check_tolerance_rad is not None:
        # The same reading, held to a tighter bar before anything descends
        # toward contact; the full rotation counts, not just the jaw line.
        rotation_error = float((_rotation(actual).inv() * _rotation(pose)).magnitude())
        if (approach_check_tolerance_m is not None and position_error > float(approach_check_tolerance_m)) or (
            approach_check_tolerance_rad is not None and rotation_error > float(approach_check_tolerance_rad)
        ):
            raise RuntimeError(
                "pre-grasp arrival outside the approach check: "
                f"{position_error * 1000:.1f} mm, {np.degrees(rotation_error):.1f} deg"
            )

    grasp_final: dict[str, Any] = {}
    corrections = 0
    descent_planned = False
    if grasp_pose is not None:
        grasp_final, corrections, descent_planned = _descend_and_close(
            ctx, grasp_pose, on_arm, preclose_width_m, pad_envelope, object_width_m, squeeze_m, settle_steps,
            descent_planner,
        )
    return {
        "final_pose": actual,
        "position_error_m": float(position_error),
        "approach_alignment": approach_alignment,
        "closing_alignment": closing_alignment,
        "recovery_used": recovery_used,
        "commanded_pose": pose,
        "commanded_grasp_pose": grasp_pose or {},
        "grasp_final_pose": grasp_final,
        "half_turn_used": half_turn_used,
        "bias_corrections": corrections,
        "descent_planned": descent_planned,
    }


def _planned_descent(ctx: NodeContext, grasp_pose: dict[str, Any], on_arm: dict[str, Any]) -> bool:
    """An orientation-locked planned line to the grasp; False when there is none to fly."""
    try:
        plan = ctx.tool(
            "motion.plan_linear",
            end=grasp_pose,
            orientation="lock",
            world_config=None,
            attached_object=None,
            allow_goal_contact=False,
            allow_start_contact=False,
            **on_arm,
        )
    except Exception:  # noqa: BLE001 -- a planner that cannot plan leaves the Cartesian descent
        return False
    plan = plan or {}
    trajectory = plan.get("trajectory") or {}
    if not plan.get("planned") or not trajectory.get("waypoints"):
        return False
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=_resample(trajectory, 3.0),
        tolerance=0.002,
        max_steps_per_waypoint=180,
    )
    ctx.tool("robot.wait_steps", steps=40)
    reached = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    position_error = float(np.linalg.norm(_vec(reached["position"]) - _vec(grasp_pose["position"])))
    angle = float((_rotation(reached).inv() * _rotation(grasp_pose)).magnitude())
    if position_error > 0.010 or angle > 0.035:
        raise RuntimeError(
            f"planned descent tracking failed: {position_error * 1000:.1f} mm, {np.degrees(angle):.1f} deg"
        )
    return True


def _descend_and_close(
    ctx: NodeContext,
    grasp_pose: dict[str, Any],
    on_arm: dict[str, Any],
    preclose_width_m: float | None,
    pad_envelope: dict[str, Any] | None,
    object_width_m: float | None,
    squeeze_m: float,
    settle_steps: int,
    descent_planner: bool = False,
) -> tuple[dict[str, Any], int, bool]:
    """Pre-close, descend, centre inside the pad envelope, then close."""
    if preclose_width_m is not None and float(preclose_width_m) > 0.0:
        ctx.tool("robot.set_grip", width_m=float(preclose_width_m), ramp_steps=40, settle_steps=40, **on_arm)
    # The planned line gets the hand to the contact neighbourhood; the Cartesian
    # contact controller below finishes the last millimetres either way.
    planned = _planned_descent(ctx, grasp_pose, on_arm) if descent_planner else False

    def settle(target: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, float]:
        ctx.tool("robot.go_to_pose_cartesian", pose=target, **on_arm)
        ctx.tool("robot.wait_steps", steps=int(settle_steps))
        reached = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
        error = _vec(reached["position"]) - _vec(grasp_pose["position"])
        angle = float((_rotation(reached).inv() * _rotation(grasp_pose)).magnitude())
        return reached, error, angle

    actual, world_error, angle = settle(grasp_pose)
    corrections = 0
    if pad_envelope is not None:
        across = 1 if str(pad_envelope.get("across_axis", "x")) == "y" else 0
        along = 1 - across
        limits = {
            across: float(pad_envelope["across_m"]),
            along: float(pad_envelope["along_m"]),
            2: float(pad_envelope["vertical_m"]),
        }
        max_rotation = float(pad_envelope["rotation_rad"])

        def inside(error: np.ndarray, rotation_error: float) -> tuple[bool, np.ndarray]:
            local = _rotation(grasp_pose).inv().apply(error)
            ok = all(abs(local[i]) <= limits[i] for i in range(3)) and rotation_error <= max_rotation
            return ok, local

        ok, local = inside(world_error, angle)
        for _ in range(int(pad_envelope.get("max_corrections", 0))):
            if ok:
                break
            if np.linalg.norm(world_error) > float(pad_envelope.get("max_correction_m", 0.012)) or angle > float(
                pad_envelope.get("max_correction_rad", 0.06)
            ):
                break
            # Command the grasp minus the measured bias; the error is still
            # measured against the grasp itself, never the corrected target.
            corrected = {
                "position": {
                    key: float(grasp_pose["position"][key]) - float(world_error[index])
                    for index, key in enumerate(("x", "y", "z"))
                },
                "rotation": dict(grasp_pose["rotation"]),
            }
            actual, world_error, angle = settle(corrected)
            corrections += 1
            ok, local = inside(world_error, angle)
        if not ok:
            raise RuntimeError(
                "grasp alignment outside pad envelope: "
                f"local error {(local * 1000).round(1).tolist()} mm, {np.degrees(angle):.1f} deg"
            )
    if object_width_m is not None:
        ctx.tool(
            "robot.set_grip",
            object_width_m=float(object_width_m),
            squeeze_m=float(squeeze_m),
            ramp_steps=100,
            settle_steps=120,
            **on_arm,
        )
    return actual, corrections, planned
