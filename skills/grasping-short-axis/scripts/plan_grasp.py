"""Plan a collision-free CuRobo trajectory to one of several candidate grasp poses.

Calls ``curobo.plan_to_grasp_poses`` with the collision world, current
joint state, and candidate grasp poses. The planner picks whichever pose
is reachable collision-free. Raises :class:`gap.errors.PlanningFailed` on
planning failure so the workflow can route to a recovery path.

The grasp-approach metric is disabled: see
``references/design_grasp_curobo.md`` for why.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap.errors import PlanningFailed
from gap.types import Observation, Se3Pose, Trajectory, WorldConfig

logger = logging.getLogger(__name__)


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
    # NOTE: topk/retries/num_ik_seeds were inflated to 25/3/256 under a
    # since-disproven hypothesis that grasp PLANNING was the task
    # bottleneck. The real bottleneck was mesh-spawn friction; with that
    # fixed, iter 0 planned + grasped + HELD the pan on a single fast
    # pass. The inflated values made every grasp-poor iter grind ~6 min
    # of silent plan_to_grasp_poses calls (up to 25*3=75) and blow the
    # per-iter time_budget, starving the refine loop of clean fast
    # feedback. Lean values: try the 8 best-ranked candidates once each
    # at the default seed count — pose DIVERSITY (8 distinct candidates)
    # gives robustness without slow same-pose re-rolls, and a hard miss
    # fails fast so the loop iterates instead of timing out.
    topk: int = 8,
    retries: int = 1,
    num_ik_seeds: int = 128,
) -> Output:
    start_joints = observation["arms"][0]["joint_state"]

    # Auto-wrap a bare Se3Pose into a one-element list.
    if isinstance(grasp_poses, dict) and "position" in grasp_poses and "rotation" in grasp_poses:
        grasp_poses = [grasp_poses]
    poses = list(grasp_poses)[: int(topk)]
    if not poses:
        raise PlanningFailed("plan_grasp: empty input grasp_poses")

    # PER-POSE planning. A single goalset call over all candidates
    # (``plan_grasp_goalset``) jointly optimises pick + approach + grasp
    # for the whole set in one solver problem — an all-or-nothing solve
    # that fails fast (success=False, empty status) whenever the set is
    # hard, even when individual poses are independently plannable.
    # Looping ``curobo.plan_to_grasp_poses`` with a SINGLETON pose is far
    # more robust: N independent solves, return the first that plans.
    # General planning-robustness fix — no privileged info, no batch-sim
    # infra; first-feasible is deterministic and sufficient here.
    _ = grasp_approach_tstep_fraction
    _ = grasp_approach_offset
    n_attempt = len(poses)
    # cuRobo's per-pose IK uses random seeds (NN + random,
    # ``solve_batch(return_seeds=1)``), so a *feasible* pose can fail one
    # solve and succeed on a re-roll: the same workflow/snapshot gave
    # 1/25 plannable on one run and 0/25 on the next. Retry each
    # candidate up to ``retries`` times (fresh IK seeds each call) and
    # raise ``num_ik_seeds`` so a single solve is itself more thorough.
    # General planning-robustness fix for stochastically-marginal grasps.
    for i, pose in enumerate(poses):
        kwargs = dict(
            world_config=world_config,
            start_joint_position=start_joints,
            grasp_poses=[pose],
            grasp_pose_is_fingertip=True,
            use_world_collision=True,
            use_cuda_graph=False,
            robot_collision_sphere_buffer=-0.01,
            collision_activation_distance=0.005,
            ignore_obstacle_names=[target_name],
            use_grasp_approach=False,
            num_ik_seeds=int(num_ik_seeds),
        )
        if robot_file:
            kwargs["robot_file"] = robot_file
        for attempt in range(max(1, int(retries))):
            try:
                plan = ctx.tool("curobo.plan_to_grasp_poses", **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[plan_grasp] candidate %d/%d attempt %d errored "
                    "(%s); retrying/next",
                    i, n_attempt, attempt, exc,
                )
                continue
            if plan.get("success", False):
                logger.info(
                    "[plan_grasp] candidate %d/%d planned successfully "
                    "(attempt %d; per-pose over %d candidates, "
                    "retries=%d, num_ik_seeds=%d)",
                    i, n_attempt, attempt, n_attempt,
                    retries, num_ik_seeds,
                )
                return {"trajectory": plan["trajectory"]}

    raise PlanningFailed(
        f"plan_grasp: 0/{n_attempt} candidate grasps had a feasible CuRobo "
        f"plan over {retries} retries each (per-pose plan_to_grasp_poses, "
        f"num_ik_seeds={num_ik_seeds}, target_name={target_name!r}). "
        f"The candidate grasp poses are not reachable + collision-free "
        f"from the start configuration. This is a grasp-generation/"
        f"reachability problem, NOT perception (the target OBB is "
        f"correct). Levers: a different grasp generator or approach axis, "
        f"a pre-grasp repositioning move, or relaxed grasp orientation."
    )
