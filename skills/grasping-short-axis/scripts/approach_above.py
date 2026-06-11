"""Move end-effector to a safe height directly above a target XY position.

Three-step motion so rotation happens *before* any descent onto the object:
1. Lift vertically at current XY with the current gripper rotation
   (``robot.go_to_pose`` — rotation unchanged, so basic IK is happy).
2. Translate laterally to above the target XY, still with current rotation.
3. Rotate in place at safe height to the target grasp rotation via CuRobo
   ``curobo.plan_to_pose``.

The split exists because the connector's basic IK (what ``robot.go_to_pose``
uses) cannot flip the Franka wrist ~90° from the horizontal home pose to a
top-down / handle grasp orientation: it seeds from the joint-limit midpoint,
satisfies position from that seed, and silently returns a near-seed config —
the EE never actually rotates. CuRobo's MotionGen handles wrist flips
robustly via goal-set sampling, so step 3 uses CuRobo. Steps 1–2 keep the
basic IK because they don't change orientation (no flip needed).

If no ``rotation`` is provided, the default downward-facing quaternion is
used. When ``target_obb`` is supplied, the approach height is derived from
the OBB top so no magic number needs to appear in the workflow JSON.
"""

from typing import TypedDict

from gap import NodeContext
from gap.errors import PlanningFailed
from gap.types import OrientedBoundingBox, Quaternion, Se3Pose, Vec3

# Clearance above the OBB top when deriving approach height from a target OBB.
_OBB_APPROACH_CLEARANCE = 0.15


class Output(TypedDict):
    done: bool


def run(
    ctx: NodeContext,
    target_position: Vec3,
    approach_height: float = 0.35,
    rotation: Quaternion | None = None,
    target_obb: OrientedBoundingBox | None = None,
) -> Output:
    down_rot: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

    if target_obb is not None:
        approach_z = max(
            target_obb["center"]["z"] + target_obb["extent"]["z"]
            + _OBB_APPROACH_CLEARANCE,
            approach_height,
        )
    else:
        approach_z = approach_height

    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    current_pos = ee["pose"]["position"]
    current_rot = ee["pose"]["rotation"]

    # 1. Lift to safe height at current XY, preserving current rotation.
    pose: Se3Pose = {
        "position": {"x": current_pos["x"], "y": current_pos["y"], "z": approach_z},
        "rotation": current_rot,
    }
    ctx.tool("robot.go_to_pose", pose=pose)

    # 2. Move above target XY, still with current rotation.
    pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": current_rot,
    }
    ctx.tool("robot.go_to_pose", pose=pose)

    # 3. Rotate in place at safe height via CuRobo (basic IK can't flip the wrist).
    target_rot = rotation if rotation is not None else down_rot
    obs = ctx.tool("robot.get_observation")
    target_pose: Se3Pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": target_rot,
    }
    plan = ctx.tool(
        "curobo.plan_to_pose",
        target_pose=target_pose,
        start_joint_position=obs["arms"][0]["joint_state"],
    )
    if not plan["success"]:
        raise PlanningFailed(
            "approach_above: CuRobo plan_to_pose for in-place rotate failed"
        )
    ctx.tool(
        "robot.execute_trajectory",
        trajectory=plan["trajectory"],
        subsample=4,
        max_steps_per_waypoint=8,
        tolerance=0.02,
    )

    return {"done": True}
