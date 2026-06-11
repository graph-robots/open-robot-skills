"""Lift a grasped object to a safe height and return the final EE pose.

Reads the current end-effector pose, computes a lift target, and executes a
single ``robot.go_to_pose`` upward.  Returns the final EE pose so downstream
transport can use it as context.

When a ``target_obb`` is supplied, the lift target is derived from the OBB top
(object top + clearance) and no magic number flows through the workflow JSON.
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox, Quaternion, Se3Pose

# Margins used when target_obb is provided. Kept inside the skill so a single
# file owns them; callers don't need to reason about safe heights.
_OBB_LIFT_CLEARANCE = 0.05  # meters above OBB top
_MIN_LIFT_DELTA = 0.10      # meters above current EE (lower bound)


class Output(TypedDict):
    pose: Se3Pose


def run(
    ctx: NodeContext,
    lift_height: float = 0.18,
    target_obb: OrientedBoundingBox | None = None,
) -> Output:
    down_rot: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    ee_pos = ee["pose"]["position"]

    if target_obb is not None:
        target_top = target_obb["center"]["z"] + target_obb["extent"]["z"]
        lift_z = max(target_top + _OBB_LIFT_CLEARANCE, ee_pos["z"] + _MIN_LIFT_DELTA)
    else:
        lift_z = max(ee_pos["z"] + lift_height, 0.1)

    pose: Se3Pose = {
        "position": {"x": ee_pos["x"], "y": ee_pos["y"], "z": lift_z},
        "rotation": down_rot,
    }
    ctx.tool("robot.go_to_pose", pose=pose)

    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    return {"pose": ee["pose"]}
