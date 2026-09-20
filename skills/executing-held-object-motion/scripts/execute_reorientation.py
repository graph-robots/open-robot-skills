"""Execute a reorientation plan waypoint by waypoint, collision-aware.

Each waypoint is planned with the held object attached and the caller's
collision world, then streamed; ``contact_transition`` waypoints skip the
planner and take the bounded Cartesian servo, because contact is the point.

Two optional layers ride on top of that and both default OFF, so a graph that
binds only ``reorientation_plan`` gets exactly the calls it always got:

- ``arm_id`` names the hand on a bimanual cell. Unset, every robot call is
  made WITHOUT an ``arm_id`` keyword -- not with ``arm_id=0`` -- because the
  promotion-parity gate compares tool names and keyword values, and a
  single-arm graph must replay byte-identically.
- ``execution_profile`` is a per-object-kind table (``speed_scale``,
  ``cartesian_fallback``); a waypoint's own keys override it. ``speed_scale`` is
  passed to the Cartesian servo only when something asked for one, for the
  same reason. ``cartesian_fallback`` turns a planner failure into a bounded
  Cartesian move instead of an error, and is counted.

The report fields (``waypoint_reports``, ``fallback_count``,
``registration_uncertainty_m``) are additive outputs. The per-waypoint pose
error is read off ``robot.get_ee_pose`` only when ``measure_errors`` is set:
that read is a recorded tool call, and a graph that never asked for it must
not gain one per waypoint.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: Per-waypoint step budget handed to ``robot.execute_trajectory``. A carry
#: plan is dense, so a waypoint that has not converged in this many control
#: steps is stuck, not slow; the executor moves on rather than timing out.
_MAX_STEPS_PER_WAYPOINT = 60


class Output(TypedDict):
    final_pose: dict[str, Any]
    waypoint_reports: list[dict[str, Any]]
    fallback_count: int
    registration_uncertainty_m: float


def _time_scale(trajectory: dict[str, Any], scale: float) -> dict[str, Any]:
    """Resample a joint path so execution duration changes, not its geometry."""
    waypoints = list(trajectory.get("waypoints") or [])
    if scale <= 1.0 or len(waypoints) < 2:
        return trajectory
    rows = np.asarray([waypoint["positions"] for waypoint in waypoints], dtype=float)
    count = max(2, int(round((len(rows) - 1) * scale)) + 1)
    source, target = np.linspace(0.0, 1.0, len(rows)), np.linspace(0.0, 1.0, count)
    scaled = np.stack([np.interp(target, source, rows[:, j]) for j in range(rows.shape[1])], axis=1)
    return {"waypoints": [{"positions": row.tolist()} for row in scaled]}


def _pose_errors(ctx: NodeContext, pose: dict[str, Any], on_arm: dict[str, Any]) -> tuple[float, float]:
    """Position [m] and rotation [deg] error between the live TCP and *pose*."""
    actual = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    position = float(np.linalg.norm([
        float(actual["position"][key]) - float(pose["position"][key]) for key in ("x", "y", "z")
    ]))
    dot = min(1.0, abs(sum(
        float(actual["rotation"][key]) * float(pose["rotation"][key]) for key in ("w", "x", "y", "z")
    )))
    return position, float(np.degrees(2.0 * np.arccos(dot)))


def run(
    ctx: NodeContext,
    reorientation_plan: dict[str, Any],
    arm_id: int | None = None,
    execution_profile: dict[str, Any] | None = None,
    registration_uncertainty_m: float | None = None,
    measure_errors: bool = False,
) -> Output:
    waypoints = list(reorientation_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("reorientation_plan must contain at least one waypoint")

    profile = execution_profile or {}
    world = reorientation_plan.get("world_config")
    attachment = reorientation_plan.get("attached_object") or {}
    # An arm is only ever NAMED, never assumed: the plan may carry one from a
    # bimanual graph, and a graph that never mentions arms gets no keyword.
    if arm_id is None:
        carried = reorientation_plan.get("arm_id", attachment.get("arm_id"))
        arm_id = None if carried is None else int(carried)
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}

    def errors(pose: dict[str, Any]) -> tuple[float | None, float | None]:
        return _pose_errors(ctx, pose, on_arm) if measure_errors else (None, None)

    if registration_uncertainty_m is None:
        registration_uncertainty_m = float(attachment.get("translation_uncertainty_m", 0.0))

    scale = max(1.0, float(reorientation_plan.get("time_scale", 1.0)))
    reports: list[dict[str, Any]] = []
    fallback_count = 0
    final_pose: dict[str, Any] | None = None

    def servo(pose: dict[str, Any], waypoint: dict[str, Any]) -> None:
        speed = waypoint.get("speed_scale", profile.get("speed_scale"))
        extra = {} if speed is None else {"speed_scale": float(speed)}
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm, **extra)

    for index, waypoint in enumerate(waypoints):
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode")
        if mode is None:
            mode = "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        if mode == "contact_transition":
            servo(final_pose, waypoint)
            position_error, rotation_error = errors(final_pose)
            reports.append({"index": index, "mode": mode, "attempts": 1, "fallback": "none",
                            "position_error_m": position_error, "rotation_error_deg": rotation_error})
            continue
        if mode not in {"planned_joint", "planned_linear"}:
            raise ValueError(f"unsupported reorientation waypoint mode {mode!r}")
        use_attachment = attachment if waypoint.get("use_attachment", True) else None
        use_world = world if waypoint.get("use_world", True) else None
        attempts = max(1, min(int(waypoint.get("max_attempts", 1)), 3))
        trajectory = None
        for _ in range(attempts):
            # Each branch names its planner literally: a tool name that arrives
            # in a variable cannot be checked against an allowlist before the
            # graph runs, and harnesses that check one statically
            # (gap-self-learning's guard) reject the call.
            options = dict(
                world_config=use_world,
                attached_object=use_attachment,
                allow_start_contact=bool(waypoint.get("allow_start_contact", False)),
                allow_goal_contact=bool(waypoint.get("allow_goal_contact", False)),
                contact_margin=float(waypoint.get("contact_margin", 0.005)),
                **on_arm,
            )
            if mode == "planned_linear":
                result = ctx.tool("motion.plan_linear", end=final_pose, orientation="lock", **options)
            else:
                result = ctx.tool("motion.plan_to_pose", pose=final_pose, **options)
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            if bool(waypoint.get("cartesian_fallback", profile.get("cartesian_fallback", False))):
                servo(final_pose, waypoint)
                fallback = "cartesian"
                fallback_count += 1
            else:
                raise RuntimeError(
                    f"collision-aware reorientation failed at waypoint {index} ({mode})"
                )
        else:
            ctx.tool(
                "robot.execute_trajectory",
                trajectory=_time_scale(trajectory, scale),
                max_steps_per_waypoint=_MAX_STEPS_PER_WAYPOINT,
                **on_arm,
            )
            fallback = "none"
        position_error, rotation_error = errors(final_pose)
        reports.append({"index": index, "mode": mode, "attempts": attempts, "fallback": fallback,
                        "position_error_m": position_error, "rotation_error_deg": rotation_error})

    assert final_pose is not None
    return {
        "final_pose": final_pose,
        "waypoint_reports": reports,
        "fallback_count": fallback_count,
        "registration_uncertainty_m": float(registration_uncertainty_m),
    }
