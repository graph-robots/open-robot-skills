"""Move end-effector to a safe height directly above a target XY position.

Three-step motion so rotation happens *before* any descent onto the object:
1. Lift vertically at the current XY with the current gripper rotation.
2. Translate laterally to above the target XY, still with current rotation.
3. Rotate in place at the safe height to the target grasp rotation.

All three legs route through ``robot.go_to_pose_cartesian``
(gap.connector.ik.CuRoboBackend): TCP-aware cartesian linear plan with a
collision-free single-pose fallback (cuRobo ``plan_to_pose``). Earlier
iterations of this script split steps 1–2 onto ``robot.go_to_pose`` and step
3 onto ``curobo.plan_to_pose`` because the connector's old basic-IK path
couldn't reliably flip the Franka wrist ~90° from the horizontal home pose;
the connector now defaults to cuRobo for both ``solve_ik`` and the linear
plan, so the same tool handles the wrist flip robustly. ``plan_linear``
prefers a straight Cartesian segment; when the goal is an in-place rotation
(zero translation), the linear plan returns ``no_interpolated_plan`` and the
backend automatically falls back to the single-pose ``plan_to_pose`` solver,
which is exactly what the old script did manually.

If no ``rotation`` is provided, the default downward-facing quaternion is
used. When ``target_obb`` is supplied, the approach height is derived from
the OBB top so no magic number needs to appear in the workflow JSON.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Quaternion, Se3Pose, Vec3

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
    ctx.tool("robot.go_to_pose_cartesian", pose=pose)

    # 2. Move above target XY, still with current rotation.
    pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": current_rot,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=pose)

    # 3. Rotate in place at safe height to the requested grasp rotation.
    target_rot = rotation if rotation is not None else down_rot
    target_pose: Se3Pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": target_rot,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=target_pose)

    return {"done": True}
