"""Execute typed planner, Cartesian-servo and contact waypoints of a placement plan.

The tolerances below are the body's own and are what a graph gets when it
binds only ``placement_plan``. A graph that carries one profile per object
kind (the tool-hanging graphs) passes ``contact_profile``, a table whose keys
OVERRIDE these one at a time; every key it leaves out keeps the constant here,
so the table cannot move a default by being present.

Three further things are optional and off unless asked, because each one is
a recorded tool call and the promotion-parity gate compares call sequences:

- ``arm_id`` names the hand on a bimanual cell and is threaded as an ABSENT
  keyword when unset (not ``arm_id=0``).
- ``verify_cartesian`` re-reads the TCP after a Cartesian servo and retries
  up to ``cartesian_attempts`` times when it is off by more than
  ``cartesian_position_tolerance_m`` / ``cartesian_rotation_tolerance_deg``.
  The rotation check asks ``robot.describe_arm`` whether the solver honours
  roll; a free wrist is judged on its tool axis alone.
- ``measure_errors`` reads the TCP before and after every waypoint for the
  ``waypoint_reports`` output.
"""

import math
from typing import Any, TypedDict

from gap import NodeContext

#: Fixture engagement has millimetres of usable radial clearance. A 10 mm
#: "already there" shortcut discards the wrist-registration correction and
#: starts insertion almost against the aperture rim, so only a genuinely
#: zero-length waypoint is skipped: about 1.5 mm and 2 degrees.
_REACHED_DISTANCE_M = 0.0015
_REACHED_DOT = 0.99985
#: After a final wrist re-registration only a small correction remains. A
#: fresh joint-space trajectory can make a loosely held object slip while the
#: TCP moves a few millimetres; the robot's smooth Cartesian servo is used for
#: such local corrections and the sampled planner kept for larger motions.
_SERVO_DISTANCE_M = 0.03
_SERVO_DOT = 0.995
#: The sampled planner can miss a narrow, valid engagement corridor that it
#: finds on another seed. Replanning is cheap and never weakens or discards
#: the collision world.
_PLAN_ATTEMPTS = 3
#: 10 mm tracking suits free-space carry but consumes almost all of a
#: ring/shaft clearance: engagement waypoints are tracked to 2 mm.
_TRACK_TOLERANCE_M = 0.002
_MAX_STEPS_PER_WAYPOINT = 60

#: The profile keys and what each defaults to -- the constants above, stated
#: once, so a profile that names a key moves exactly that one.
_DEFAULT_PROFILE: dict[str, Any] = {
    "skip_position_tolerance_m": _REACHED_DISTANCE_M,
    "skip_min_quaternion_dot": _REACHED_DOT,
    "local_cartesian_max_distance_m": _SERVO_DISTANCE_M,
    "local_cartesian_min_quaternion_dot": _SERVO_DOT,
    "planner_attempts": _PLAN_ATTEMPTS,
    "trajectory_tolerance_m": _TRACK_TOLERANCE_M,
    "max_steps_per_waypoint": _MAX_STEPS_PER_WAYPOINT,
    "contact_margin_m": 0.005,
    # Only read under ``verify_cartesian``.
    "cartesian_attempts": 2,
    "cartesian_position_tolerance_m": 0.015,
    "cartesian_rotation_tolerance_deg": 7.25,
    "local_cartesian_position_tolerance_m": 0.030,
}


class Output(TypedDict):
    final_pose: dict[str, Any]
    waypoint_count: int
    waypoint_reports: list[dict[str, Any]]
    fallback_count: int
    registration_uncertainty_m: float


def _pose_distance(current: dict[str, Any], target: dict[str, Any]) -> tuple[float, float]:
    """Return (translation distance in metres, |quaternion dot|) between two poses."""
    cp, tp = current["position"], target["position"]
    distance = math.sqrt(sum((float(cp[axis]) - float(tp[axis])) ** 2 for axis in ("x", "y", "z")))
    cq, tq = current["rotation"], target["rotation"]
    dot = abs(sum(float(cq[axis]) * float(tq[axis]) for axis in ("w", "x", "y", "z")))
    return distance, dot


def _angle_deg(dot: float) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def _tool_axis(pose: dict[str, Any]) -> tuple[float, float, float]:
    q = pose["rotation"]
    w, x, y, z = (float(q[key]) for key in ("w", "x", "y", "z"))
    norm = max(math.sqrt(w * w + x * x + y * y + z * z), 1.0e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))


def _orientation_ok(current: dict[str, Any], target: dict[str, Any], honours_roll: bool, tolerance_deg: float) -> bool:
    if honours_roll:
        return _angle_deg(_pose_distance(current, target)[1]) <= tolerance_deg
    axes = zip(_tool_axis(current), _tool_axis(target), strict=True)
    return sum(a * b for a, b in axes) >= math.cos(math.radians(tolerance_deg))


def run(
    ctx: NodeContext,
    placement_plan: dict[str, Any],
    relation: str | None = None,
    contact_profile: dict[str, Any] | None = None,
    arm_id: int | None = None,
    registration_uncertainty_m: float | None = None,
    verify_cartesian: bool = False,
    measure_errors: bool = False,
) -> Output:
    profile = dict(_DEFAULT_PROFILE)
    profile.update(contact_profile or {})
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement plan has no waypoints")
    plan_relation = placement_plan.get("relation")
    if relation is not None and plan_relation is not None and relation != plan_relation:
        raise ValueError(f"relation mismatch: plan={plan_relation!r}, requested={relation!r}")
    world = placement_plan.get("world_config")
    attachment = placement_plan.get("attached_object")
    # An arm is only ever NAMED, never assumed.
    if arm_id is None:
        carried = (attachment or {}).get("arm_id") if isinstance(attachment, dict) else None
        arm_id = None if carried is None else int(carried)
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}
    if registration_uncertainty_m is None:
        registration_uncertainty_m = float((attachment or {}).get("translation_uncertainty_m", 0.0)) if isinstance(attachment, dict) else 0.0
    honours_roll = True
    if verify_cartesian:
        arm = ctx.tool("robot.describe_arm", **on_arm)
        honours_roll = bool((arm.get("solver") or {}).get("honours_roll", False))

    def ee_pose() -> dict[str, Any]:
        return ctx.tool("robot.get_ee_pose", **on_arm)["pose"]

    def cartesian(pose: dict[str, Any], position_tolerance: float) -> int:
        """Servo to *pose*; under ``verify_cartesian`` re-read and retry. Returns attempts."""
        attempts = max(1, min(int(profile["cartesian_attempts"]), 3)) if verify_cartesian else 1
        for attempt in range(1, attempts + 1):
            ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm)
            if not verify_cartesian:
                return attempt
            actual = ee_pose()
            distance, _ = _pose_distance(actual, pose)
            if distance <= position_tolerance and _orientation_ok(
                actual, pose, honours_roll, float(profile["cartesian_rotation_tolerance_deg"])
            ):
                return attempt
        distance, dot = _pose_distance(ee_pose(), pose)
        raise RuntimeError(
            f"Cartesian mating waypoint not reached: {distance * 1000:.1f} mm / {_angle_deg(dot):.1f} deg"
        )

    reports: list[dict[str, Any]] = []
    fallback_count = 0
    final_pose: dict[str, Any] | None = None
    for index, waypoint in enumerate(waypoints):
        final_pose, mode = waypoint["pose"], waypoint.get("mode")
        if mode is None:
            mode = "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        report: dict[str, Any] = {"index": index, "mode": mode, "attempts": 0, "fallback": "none",
                                  "contact_status": "not_requested"}
        if measure_errors:
            distance, dot = _pose_distance(ee_pose(), final_pose)
            report.update(position_error_before_m=distance, rotation_error_before_deg=_angle_deg(dot))
        if mode == "contact_seat":
            result = ctx.tool("robot.move_cartesian_until_contact", pose=final_pose, **on_arm)
            report.update(attempts=1, contact_status=str((result or {}).get("status", "target_or_stall")))
        elif mode == "cartesian_cross":
            # Crossing a fixture mouth must not stop at the first incidental
            # touch. The pose is a short local segment, typically recomputed
            # from the final wrist observation, for which Cartesian servoing
            # is smoother and more repeatable than a new sampled plan.
            report["attempts"] = cartesian(final_pose, float(profile["cartesian_position_tolerance_m"]))
        elif mode in {"planned_joint", "planned_linear"}:
            # The preceding carry plan normally terminates at the first
            # engagement waypoint. Do not ask the planner to solve a
            # zero-motion problem there: at fixture clearance its start-contact
            # check can reject it even though no swept motion is requested.
            current_pose = ee_pose()
            distance, orientation_dot = _pose_distance(current_pose, final_pose)
            if (
                distance <= float(profile["skip_position_tolerance_m"])
                and orientation_dot >= float(profile["skip_min_quaternion_dot"])
            ):
                report["fallback"] = "already_reached"
            elif (
                mode == "planned_joint"
                and distance <= float(profile["local_cartesian_max_distance_m"])
                and orientation_dot >= float(profile["local_cartesian_min_quaternion_dot"])
            ):
                report["attempts"] = cartesian(final_pose, float(profile["local_cartesian_position_tolerance_m"]))
                report["fallback"] = "local_cartesian"
                fallback_count += 1
            else:
                trajectory = None
                attempts = max(1, min(int(waypoint.get("max_attempts", profile["planner_attempts"])), 3))
                for attempt in range(1, attempts + 1):
                    # Both branches name their tool literally: a call whose tool
                    # name arrives in a variable cannot be checked against an
                    # allowlist before the graph runs, and harnesses that check
                    # one statically (gap-self-learning's guard) reject it.
                    options = dict(
                        world_config=world,
                        attached_object=attachment if waypoint.get("use_attachment", True) else None,
                        allow_start_contact=bool(waypoint.get("allow_start_contact", False)),
                        allow_goal_contact=bool(waypoint.get("allow_goal_contact", False)),
                        contact_margin=float(waypoint.get("contact_margin", profile["contact_margin_m"])),
                        **on_arm,
                    )
                    if mode == "planned_linear":
                        result = ctx.tool("motion.plan_linear", end=final_pose, orientation="lock", **options)
                    else:
                        result = ctx.tool("motion.plan_to_pose", pose=final_pose, **options)
                    trajectory = result.get("trajectory") if result.get("planned") else None
                    report["attempts"] = attempt
                    if trajectory and trajectory.get("waypoints"):
                        break
                if not trajectory or not trajectory.get("waypoints"):
                    raise RuntimeError(
                        f"planner refused a typed fixture-engagement waypoint after {attempts} attempts"
                    )
                ctx.tool(
                    "robot.execute_trajectory",
                    trajectory=trajectory,
                    tolerance=float(profile["trajectory_tolerance_m"]),
                    max_steps_per_waypoint=int(profile["max_steps_per_waypoint"]),
                    **on_arm,
                )
        else:
            raise ValueError(f"unknown placement waypoint mode {mode!r}")
        if measure_errors:
            distance, dot = _pose_distance(ee_pose(), final_pose)
            report.update(position_error_m=distance, rotation_error_deg=_angle_deg(dot))
        reports.append(report)
    assert final_pose is not None
    return {
        "final_pose": final_pose,
        "waypoint_count": len(waypoints),
        "waypoint_reports": reports,
        "fallback_count": fallback_count,
        "registration_uncertainty_m": float(registration_uncertainty_m),
    }
