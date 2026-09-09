"""Execute and diagnose a declarative feature-mating waypoint sequence."""

import math
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    final_pose: dict[str, Any]
    waypoint_count: int
    waypoint_reports: list[dict[str, Any]]
    fallback_count: int
    registration_uncertainty_m: float


_DEFAULT_PROFILE = {
    "skip_position_tolerance_m": 0.0015,
    "skip_rotation_tolerance_deg": 2.0,
    "cartesian_position_tolerance_m": 0.015,
    "cartesian_rotation_tolerance_deg": 7.25,
    "cartesian_attempts": 2,
    "planner_attempts": 3,
    "local_cartesian_max_distance_m": 0.030,
    "local_cartesian_min_quaternion_dot": 0.995,
    "local_cartesian_position_tolerance_m": 0.030,
    "trajectory_tolerance_m": 0.002,
    "max_steps_per_waypoint": 60,
    "contact_margin_m": 0.005,
}


def _errors(current: dict[str, Any], target: dict[str, Any]) -> tuple[float, float, float]:
    cp, tp = current["position"], target["position"]
    distance = math.sqrt(sum(
        (float(cp[key]) - float(tp[key])) ** 2 for key in ("x", "y", "z")
    ))
    cq, tq = current["rotation"], target["rotation"]
    dot = min(1.0, abs(sum(float(cq[key]) * float(tq[key])
                           for key in ("w", "x", "y", "z"))))
    return distance, math.degrees(2.0 * math.acos(dot)), dot


def _tool_axis(pose: dict[str, Any]) -> tuple[float, float, float]:
    q = pose["rotation"]
    w, x, y, z = (float(q[key]) for key in ("w", "x", "y", "z"))
    norm = max(math.sqrt(w*w + x*x + y*y + z*z), 1.0e-12)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return (2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0 - 2.0*(x*x + y*y))


def _orientation_ok(current: dict[str, Any], target: dict[str, Any],
                    honours_roll: bool, tolerance_deg: float) -> bool:
    if honours_roll:
        return _errors(current, target)[1] <= tolerance_deg
    return sum(a*b for a, b in zip(_tool_axis(current), _tool_axis(target), strict=True)) >= \
        math.cos(math.radians(tolerance_deg))


def _verified_cartesian(ctx: NodeContext, pose: dict[str, Any], arm_id: int,
                        profile: dict[str, Any], position_tolerance: float,
                        honours_roll: bool) -> tuple[dict[str, Any], int]:
    attempts = max(1, min(int(profile["cartesian_attempts"]), 3))
    actual = None
    for attempt in range(1, attempts + 1):
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, arm_id=arm_id)
        actual = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
        distance, _, _ = _errors(actual, pose)
        if distance <= position_tolerance and _orientation_ok(
                actual, pose, honours_roll, float(profile["cartesian_rotation_tolerance_deg"])):
            return actual, attempt
    assert actual is not None
    distance, angle, _ = _errors(actual, pose)
    raise RuntimeError(
        f"Cartesian mating waypoint not reached: {distance * 1000:.1f} mm / {angle:.1f} deg"
    )


def run(ctx: NodeContext, placement_plan: dict[str, Any], relation: str | None = None,
        contact_profile: dict[str, Any] | None = None, arm_id: int | None = None,
        registration_uncertainty_m: float | None = None) -> Output:
    profile = dict(_DEFAULT_PROFILE)
    profile.update(contact_profile or {})
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement_plan must contain at least one waypoint")
    plan_relation = placement_plan.get("relation")
    if relation is not None and plan_relation is not None and relation != plan_relation:
        raise ValueError(f"relation mismatch: plan={plan_relation!r}, requested={relation!r}")
    attachment = placement_plan.get("attached_object") or {}
    if arm_id is None:
        arm_id = int(attachment.get("arm_id", 0))
    arm = ctx.tool("robot.describe_arm", arm_id=int(arm_id))
    honours_roll = bool((arm.get("solver") or {}).get("honours_roll", False))
    if registration_uncertainty_m is None:
        registration_uncertainty_m = float(attachment.get("translation_uncertainty_m", 0.0))
    world = placement_plan.get("world_config")
    reports: list[dict[str, Any]] = []
    fallback_count = 0
    final_pose: dict[str, Any] | None = None

    for index, item in enumerate(waypoints):
        pose = item.get("pose", item)
        mode = item.get("mode") or (
            "planned_linear" if item.get("cartesian", False) else "planned_joint"
        )
        current = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
        before_distance, before_angle, before_dot = _errors(current, pose)
        report: dict[str, Any] = {
            "index": index, "mode": mode, "attempts": 0,
            "fallback": "none", "contact_status": "not_requested",
            "position_error_before_m": before_distance,
            "rotation_error_before_deg": before_angle,
        }

        if mode == "contact_seat":
            result = ctx.tool("robot.move_cartesian_until_contact", pose=pose, arm_id=int(arm_id))
            report["attempts"] = 1
            report["contact_status"] = str((result or {}).get("status", "target_or_stall"))
        elif mode == "cartesian_cross":
            actual, attempts = _verified_cartesian(
                ctx, pose, int(arm_id), profile,
                float(profile["cartesian_position_tolerance_m"]),
                honours_roll,
            )
            report["attempts"] = attempts
            current = actual
        elif mode in {"planned_joint", "planned_linear"}:
            if (before_distance <= float(profile["skip_position_tolerance_m"])
                    and _orientation_ok(current, pose, honours_roll,
                                        float(profile["skip_rotation_tolerance_deg"]))):
                report["fallback"] = "already_reached"
            elif (mode == "planned_joint"
                  and before_distance <= float(profile["local_cartesian_max_distance_m"])
                  and before_dot >= float(profile["local_cartesian_min_quaternion_dot"])):
                current, attempts = _verified_cartesian(
                    ctx, pose, int(arm_id), profile,
                    float(profile["local_cartesian_position_tolerance_m"]),
                    honours_roll,
                )
                report["attempts"] = attempts
                report["fallback"] = "local_cartesian"
                fallback_count += 1
            else:
                trajectory = None
                attempts = max(1, min(int(item.get(
                    "max_attempts", profile["planner_attempts"])), 3))
                for attempt in range(1, attempts + 1):
                    tool = "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose"
                    inputs = ({"end": pose, "orientation": "lock"}
                              if mode == "planned_linear" else {"pose": pose})
                    inputs.update(
                        arm_id=int(arm_id), world_config=world,
                        attached_object=attachment if item.get("use_attachment", True) else None,
                        allow_start_contact=bool(item.get("allow_start_contact", False)),
                        allow_goal_contact=bool(item.get("allow_goal_contact", False)),
                        contact_margin=float(item.get(
                            "contact_margin", profile["contact_margin_m"])),
                    )
                    result = ctx.tool(tool, **inputs)
                    trajectory = result.get("trajectory") if result.get("planned") else None
                    report["attempts"] = attempt
                    if trajectory and trajectory.get("waypoints"):
                        break
                if not trajectory or not trajectory.get("waypoints"):
                    raise RuntimeError(
                        f"collision-aware mating motion failed at waypoint {index} ({mode})"
                    )
                ctx.tool(
                    "robot.execute_trajectory", trajectory=trajectory,
                    tolerance=float(profile["trajectory_tolerance_m"]),
                    max_steps_per_waypoint=int(profile["max_steps_per_waypoint"]),
                    arm_id=int(arm_id),
                )
        else:
            raise ValueError(f"unsupported feature-mating waypoint mode {mode!r}")

        current = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
        distance, angle, _ = _errors(current, pose)
        report.update(position_error_m=distance, rotation_error_deg=angle)
        reports.append(report)
        final_pose = pose

    assert final_pose is not None
    return {
        "final_pose": final_pose,
        "waypoint_count": len(waypoints),
        "waypoint_reports": reports,
        "fallback_count": fallback_count,
        "registration_uncertainty_m": float(registration_uncertainty_m),
    }
