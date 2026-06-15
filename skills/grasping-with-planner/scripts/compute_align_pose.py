"""Compute the align-pose for grasping-with-planner.

Same construction as ``grasping-direct-ik``: lifts the grasp pose's XY by 0.15 m
above the OBB top, preserving the rotation. CuRobo plans into this pre-grasp
align-pose so the trajectory to the actual grasp pose becomes a pure
straight-line descent — no rotation blending, no twist-while-closing.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose

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
