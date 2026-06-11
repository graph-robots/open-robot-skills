"""Compute the align-pose for direct-IK grasping.

Constructs an SE(3) pose at ``(grasp_pose.x, grasp_pose.y, target_obb_top + 0.15)``
with the same rotation as the grasp pose. Used by the ``grasping-direct-ik``
skill: the gripper rotates into the grasp orientation at this align-pose
first, then descends straight down to the actual grasp pose.

The 0.15 m clearance above the OBB top is the minimum safe approach height
for the Franka end-effector + a typical 5 cm gripper plus a few cm of
margin. If the held object is much taller, the workflow author may want to
post-process this with ``transporting-objects``'s clearance constants.
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox, Se3Pose

_OBB_TOP_CLEARANCE = 0.15  # meters above the OBB top face


class Output(TypedDict):
    align_pose: Se3Pose


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    target_obb: OrientedBoundingBox,
) -> Output:
    approach_z = (
        target_obb["center"]["z"] + target_obb["extent"]["z"] + _OBB_TOP_CLEARANCE
    )
    align_pose: Se3Pose = {
        "position": {
            "x": grasp_pose["position"]["x"],
            "y": grasp_pose["position"]["y"],
            "z": approach_z,
        },
        "rotation": grasp_pose["rotation"],
    }
    return {"align_pose": align_pose}
