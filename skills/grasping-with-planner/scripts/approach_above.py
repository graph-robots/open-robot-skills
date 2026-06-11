"""Move end-effector to a safe height directly above a target XY position.

Three-step motion so rotation happens *before* any descent onto the object:
1. Lift vertically at current XY with the current gripper rotation.
2. Translate laterally to above the target XY, still with current rotation.
3. Rotate in place to the target grasp rotation at safe height.

If no ``rotation`` is provided, the default downward-facing quaternion is used
and the final rotate step is a no-op. When ``target_obb`` is supplied, the
approach height is derived from the OBB top so no magic number needs to appear
in the workflow JSON.
"""

from typing import TypedDict

from gap import NodeContext
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

    # 3. Rotate in place at safe height to the requested grasp rotation.
    target_rot = rotation if rotation is not None else down_rot
    pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": target_rot,
    }
    ctx.tool("robot.go_to_pose", pose=pose)

    return {"done": True}
