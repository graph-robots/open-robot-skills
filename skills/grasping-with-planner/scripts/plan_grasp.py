"""Plan a collision-free CuRobo trajectory to one of several candidate grasp poses.

Calls ``curobo.plan_to_grasp_poses`` with the collision world, current joint
state, and candidate grasp poses. The planner picks whichever pose is
reachable collision-free. Raises :class:`gap.errors.PlanningFailed` on
planning failure so the workflow can route to a recovery path.

The grasp-approach metric is disabled: see
``references/design_grasp_curobo.md`` for why.
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
    grasp_poses: list[Se3Pose] | Se3Pose,
    target_name: str,
    robot_file: str = "",
    grasp_approach_offset: float = 0.12,
    grasp_approach_tstep_fraction: float = 0.7,
) -> Output:
    start_joints = observation["arms"][0]["joint_state"]

    # Auto-wrap a bare Se3Pose into a one-element list. The LLM may wire
    # a single-pose output (top_down_grasp_from_obb returns a bare Se3Pose)
    # here directly; CuRobo's goal-set planner expects a list.
    if isinstance(grasp_poses, dict) and "position" in grasp_poses and "rotation" in grasp_poses:
        grasp_poses = [grasp_poses]

    kwargs = dict(
        world_config=world_config,
        start_joint_position=start_joints,
        grasp_poses=list(grasp_poses),
        grasp_pose_is_fingertip=True,
        use_world_collision=True,
        use_cuda_graph=False,
        robot_collision_sphere_buffer=-0.01,
        # 0.005 (not 0.001): a tighter activation distance gives CuRobo too
        # little room to find collision-free descents from the post-approach
        # pose; cad=0.005 was validated in cartesian_obb_per_task_cad005 to
        # let tasks 06/08 plan successfully where cad=0.001 fails every time.
        collision_activation_distance=0.005,
        ignore_obstacle_names=[target_name],
        # See references/design_grasp_curobo.md — the workflow pre-rotates
        # via the in-skill `approach` state before this plan, so
        # use_grasp_approach only adds spurious termination at the offset pose.
        use_grasp_approach=False,
    )
    _ = grasp_approach_offset
    _ = grasp_approach_tstep_fraction
    if robot_file:
        kwargs["robot_file"] = robot_file

    plan = ctx.tool("curobo.plan_to_grasp_poses", **kwargs)
    if not plan["success"]:
        raise PlanningFailed("Grasp planning failed")

    return {"trajectory": plan["trajectory"]}
