"""Canonical linear drop-and-release implementation from transporting-objects."""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3


_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
    arm_id: int = 0,
) -> Output:
    rotation = drop_rotation if drop_rotation is not None else _DOWN
    end_pose: Se3Pose = {"position": drop_position, "rotation": rotation}
    ctx.tool("robot.go_to_pose_cartesian", pose=end_pose, arm_id=int(arm_id))
    ctx.tool("robot.open_gripper", settle_steps=60, arm_id=int(arm_id))
    # Clear the released object and container before the large return-home
    # motion. Preserve the wrist orientation and use one short Cartesian IK
    # retreat so the fingers cannot strike or drag the newly released tool.
    current = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
    retreat_position = dict(current["position"])
    retreat_position["z"] = float(retreat_position["z"]) + 0.05
    retreat_pose: Se3Pose = {
        "position": retreat_position,
        "rotation": dict(current["rotation"]),
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=retreat_pose, arm_id=int(arm_id))
    ctx.tool("robot.wait_steps", steps=12)
    ctx.tool("robot.go_home")
    return {"drop_position": drop_position}
