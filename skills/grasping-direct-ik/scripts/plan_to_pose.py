"""Plan a collision-free CuRobo trajectory to a single target pose.

Calls ``curobo.plan_to_pose`` with the collision world, current joint
state, and a target EE pose. Raises :class:`gap.errors.PlanningFailed` on
planning failure so the workflow can route to a recovery path.

Bundle script for ``grasping-direct-ik``: kept here because some flavors of
direct-IK grasping (still no full obstacle planning) opt into a CuRobo
single-pose plan when the platform deploys CuRobo as a fast IK fallback
without using the multi-grasp goalset variant.
"""

from typing import TypedDict

from gap import NodeContext
from gap.errors import PlanningFailed
from gap.types import Observation, Se3Pose, Trajectory, WorldConfig


class Output(TypedDict):
    trajectory: Trajectory


def run(
    ctx: NodeContext,
    world_config: WorldConfig,
    observation: Observation,
    target_pose: Se3Pose,
    robot_file: str = "",
) -> Output:
    start_joints = observation["arms"][0]["joint_state"]

    kwargs = dict(
        world_config=world_config,
        start_joint_position=start_joints,
        target_pose=target_pose,
    )
    if robot_file:
        kwargs["robot_file"] = robot_file

    plan = ctx.tool("curobo.plan_to_pose", **kwargs)
    if not plan["success"]:
        raise PlanningFailed("CuRobo plan_to_pose failed")

    return {"trajectory": plan["trajectory"]}
