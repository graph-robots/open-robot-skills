"""Execute a clearance-first pose sequence while preserving the grasp."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    final_pose: dict[str, Any]
    waypoint_reports: list[dict[str, Any]]
    fallback_count: int
    registration_uncertainty_m: float


def _time_scale(trajectory: dict[str, Any], scale: float) -> dict[str, Any]:
    waypoints = list(trajectory.get("waypoints") or [])
    if scale <= 1.0 or len(waypoints) < 2:
        return trajectory
    rows = np.asarray([waypoint["positions"] for waypoint in waypoints], dtype=float)
    count = max(2, int(round((len(rows) - 1) * scale)) + 1)
    source, target = np.linspace(0.0, 1.0, len(rows)), np.linspace(0.0, 1.0, count)
    scaled = np.stack([np.interp(target, source, rows[:, j]) for j in range(rows.shape[1])], axis=1)
    return {"waypoints": [{"positions": row.tolist()} for row in scaled]}


def _pose_errors(ctx: NodeContext, pose: dict[str, Any], arm_id: int) -> tuple[float, float]:
    actual = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    position = float(np.linalg.norm([
        float(actual["position"][key]) - float(pose["position"][key])
        for key in ("x", "y", "z")
    ]))
    dot = min(1.0, abs(sum(
        float(actual["rotation"][key]) * float(pose["rotation"][key])
        for key in ("w", "x", "y", "z")
    )))
    return position, float(np.degrees(2.0 * np.arccos(dot)))


def run(ctx: NodeContext, reorientation_plan: dict[str, Any], arm_id: int | None = None,
        execution_profile: dict[str, Any] | None = None,
        registration_uncertainty_m: float | None = None) -> Output:
    profile = execution_profile or {}
    waypoints = list(reorientation_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("reorientation_plan must contain at least one waypoint")

    world = reorientation_plan.get("world_config")
    attachment = reorientation_plan.get("attached_object") or {}
    if arm_id is None:
        arm_id = int(reorientation_plan.get("arm_id", attachment.get("arm_id", 0)))
    if registration_uncertainty_m is None:
        registration_uncertainty_m = float(attachment.get("translation_uncertainty_m", 0.0))
    scale = max(1.0, float(reorientation_plan.get("time_scale", 1.0)))
    final_pose: dict[str, Any] | None = None
    reports: list[dict[str, Any]] = []
    fallback_count = 0
    for index, waypoint in enumerate(waypoints):
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode")
        if mode is None:
            mode = "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        if mode == "contact_transition":
            ctx.tool(
                "robot.go_to_pose_cartesian", pose=final_pose, arm_id=int(arm_id),
                speed_scale=float(waypoint.get("speed_scale", profile.get("speed_scale", 1.0))),
            )
            position_error, rotation_error = _pose_errors(ctx, final_pose, int(arm_id))
            reports.append({"index": index, "mode": mode, "attempts": 1,
                            "fallback": "none", "position_error_m": position_error,
                            "rotation_error_deg": rotation_error})
            continue
        if mode not in {"planned_joint", "planned_linear"}:
            raise ValueError(f"unsupported reorientation waypoint mode {mode!r}")
        use_attachment = attachment if waypoint.get("use_attachment", True) else None
        use_world = world if waypoint.get("use_world", True) else None
        attempts = max(1, min(int(waypoint.get("max_attempts", 1)), 3))
        trajectory = None
        for _ in range(attempts):
            planner_tool = "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose"
            inputs = (
                {"end": final_pose, "orientation": "lock"}
                if mode == "planned_linear"
                else {"pose": final_pose}
            )
            inputs["arm_id"] = int(arm_id)
            inputs.update(
                world_config=use_world,
                attached_object=use_attachment,
                allow_start_contact=bool(waypoint.get("allow_start_contact", False)),
                allow_goal_contact=bool(waypoint.get("allow_goal_contact", False)),
                contact_margin=float(waypoint.get("contact_margin", 0.005)),
            )
            result = ctx.tool(planner_tool, **inputs)
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            if bool(waypoint.get("cartesian_fallback",
                                 profile.get("cartesian_fallback", False))):
                ctx.tool(
                    "robot.go_to_pose_cartesian", pose=final_pose, arm_id=int(arm_id),
                    speed_scale=float(waypoint.get("speed_scale", profile.get("speed_scale", 1.0))),
                )
                fallback = "cartesian"
                fallback_count += 1
            else:
                raise RuntimeError(
                    f"collision-aware reorientation failed at waypoint {index} ({mode})"
                )
        else:
            ctx.tool(
                "robot.execute_trajectory", trajectory=_time_scale(trajectory, scale),
                arm_id=int(arm_id),
            )
            fallback = "none"
        position_error, rotation_error = _pose_errors(ctx, final_pose, int(arm_id))
        reports.append({"index": index, "mode": mode, "attempts": attempts,
                        "fallback": fallback, "position_error_m": position_error,
                        "rotation_error_deg": rotation_error})

    assert final_pose is not None
    return {"final_pose": final_pose, "waypoint_reports": reports,
            "fallback_count": fallback_count,
            "registration_uncertainty_m": float(registration_uncertainty_m)}
