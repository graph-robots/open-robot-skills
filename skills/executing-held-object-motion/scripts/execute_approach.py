"""Execute the free-space approach leg of a placement plan."""

import math
from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    approach_pose: dict[str, Any]
    waypoint_report: dict[str, Any]


def _errors(current: dict[str, Any], target: dict[str, Any]) -> tuple[float, float, float]:
    distance = math.sqrt(sum(
        (float(current["position"][key]) - float(target["position"][key])) ** 2
        for key in ("x", "y", "z")
    ))
    dot = min(1.0, abs(sum(
        float(current["rotation"][key]) * float(target["rotation"][key])
        for key in ("w", "x", "y", "z")
    )))
    return distance, math.degrees(2.0 * math.acos(dot)), dot


def run(ctx: NodeContext, placement_plan: dict[str, Any], arm_id: int | None = None,
        execution_profile: dict[str, Any] | None = None,
        registration_uncertainty_m: float | None = None) -> Output:
    profile = execution_profile or {}
    waypoints = list(placement_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("placement plan has no approach waypoint")
    waypoint, pose = waypoints[0], waypoints[0]["pose"]
    attachment = placement_plan.get("attached_object") or {}
    if arm_id is None:
        arm_id = int(attachment.get("arm_id", 0))
    if registration_uncertainty_m is None:
        registration_uncertainty_m = float(attachment.get("translation_uncertainty_m", 0.0))
    current = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
    distance, angle, _ = _errors(current, pose)
    local_distance = float(profile.get("local_cartesian_max_distance_m", 0.050))
    local_angle = float(profile.get("local_cartesian_max_rotation_deg", 25.0))
    method, attempts = "cartesian", 1
    if distance <= local_distance and angle <= local_angle:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, arm_id=int(arm_id))
    else:
        trajectory = None
        attempts = max(1, min(int(profile.get("planner_attempts", 3)), 3))
        for attempt in range(1, attempts + 1):
            result = ctx.tool(
                "motion.plan_to_pose", pose=pose, arm_id=int(arm_id),
                world_config=placement_plan.get("world_config"),
                attached_object=attachment, allow_start_contact=False,
                allow_goal_contact=False,
                contact_margin=float(waypoint.get("contact_margin",
                                                  profile.get("contact_margin_m", 0.005))),
            )
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            goal_attempts = max(0, min(int(profile.get("goal_contact_attempts", 2)), 3))
            for goal_attempt in range(1, goal_attempts + 1):
                result = ctx.tool(
                    "motion.plan_to_pose", pose=pose, arm_id=int(arm_id),
                    world_config=placement_plan.get("world_config"),
                    attached_object=attachment, allow_start_contact=False,
                    allow_goal_contact=True,
                    contact_margin=float(profile.get("goal_contact_margin_m", 0.002)),
                )
                trajectory = result.get("trajectory") if result.get("planned") else None
                attempts += 1
                if trajectory and trajectory.get("waypoints"):
                    break
        if trajectory and trajectory.get("waypoints"):
            method = "planned"
            ctx.tool(
                "robot.execute_trajectory", trajectory=trajectory,
                tolerance=float(profile.get("trajectory_tolerance_m", 0.002)),
                max_steps_per_waypoint=int(profile.get("max_steps_per_waypoint", 60)),
                arm_id=int(arm_id),
            )
        elif (distance <= float(profile.get("recovery_max_distance_m", 0.100))
              and angle <= float(profile.get("recovery_max_rotation_deg", 35.0))):
            method = "cartesian_recovery"
            ctx.tool("robot.go_to_pose_cartesian", pose=pose, arm_id=int(arm_id))
        else:
            raise RuntimeError(
                f"planner refused approach ({distance * 1000:.1f} mm / {angle:.1f} deg)"
            )
    reached = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
    residual, rotation_error, _ = _errors(reached, pose)
    if method == "cartesian_recovery" and residual > float(
            profile.get("recovery_position_tolerance_m", 0.015)):
        raise RuntimeError(f"Cartesian approach recovery left {residual * 1000:.1f} mm error")
    return {
        "approach_pose": pose,
        "waypoint_report": {
            "index": 0, "mode": "approach", "method": method,
            "attempts": attempts, "position_error_m": residual,
            "rotation_error_deg": rotation_error,
            "registration_uncertainty_m": float(registration_uncertainty_m),
        },
    }
