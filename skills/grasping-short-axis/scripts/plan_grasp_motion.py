"""Plan a complete grasp SEQUENCE (approach / grasp / lift) via cuRobo v0.8.

Wraps ``curobo.plan_grasp_motion`` (cuRobo v0.8 ``plan_grasp``).
Unlike the per-pose ``plan_to_grasp_poses`` in ``plan_grasp.py`` — which
solves ONE free-space motion to a reachable goal and returns a single
trajectory — ``plan_grasp_motion`` decomposes the grasp into three
separate, sequentially-feasible trajectories:

  1. ``approach`` — free-space motion from the current joint state to a
     pre-grasp pose (``grasp_pose`` backed off along ``approach_axis``
     by ``approach_distance``).
  2. ``grasp``    — a CONSTRAINED linear segment from the pre-grasp into
     the grasp pose.
  3. ``lift``     — a CONSTRAINED linear segment away from the grasp
     pose along ``lift_axis`` by ``lift_distance``.

Returning the three legs separately lets the workflow interleave a
``robot.close_gripper`` between ``approach``/``grasp`` and the ``lift`` —
the gripper closes on the (now reached) object before the constrained
lift pulls it clear.

This is an ESCALATION lever, not a replacement for ``plan_grasp.py``.
Reach for it on FAR / LOW / OCCLUDED targets where the standard
per-pose ``plan_to_grasp_poses`` returns 0 feasible / "not reachable":
the constrained pre-grasp→grasp decomposition (a short collision-light
linear segment seeded from a separately-solved free-space pose) can
find a feasible path where the all-in-one IK to the grasp pose cannot.

Raises :class:`gap.errors.PlanningFailed` on planning failure so the
workflow can route to a recovery path (mirrors ``plan_grasp.py``).

@note Requires cuRobo v0.8 — ``plan_grasp_motion`` returns
``success=False`` on v0.7; this script raises ``PlanningFailed`` in that
case, which the subgraph's ``on_error`` catches.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PlanningFailed
from gap_core.types import Observation, Se3Pose, Trajectory

logger = logging.getLogger(__name__)


class Output(TypedDict):
    # Free-space motion: current state → pre-grasp offset pose.
    approach_trajectory: Trajectory
    # Constrained linear: pre-grasp → grasp pose. Close the gripper
    # AFTER executing this leg and BEFORE the lift leg.
    grasp_trajectory: Trajectory
    # Constrained linear: grasp pose → post-grasp along lift_axis.
    lift_trajectory: Trajectory


def run(
    ctx: NodeContext,
    observation: Observation,
    grasp_pose: Se3Pose,
    approach_axis: str = "z",
    approach_distance: float = 0.12,
    approach_in_tool_frame: bool = False,
    lift_axis: str = "z",
    lift_distance: float = 0.20,
    lift_in_tool_frame: bool = False,
    robot_file: str = "",
) -> Output:
    start_joints = observation["arms"][0]["joint_state"]

    # ``plan_grasp_motion`` takes a SINGLE grasp pose (not a goalset). If
    # an upstream node hands a list, use the best-ranked candidate — the
    # per-leg constrained decomposition is what buys reachability here,
    # not pose diversity.
    if isinstance(grasp_pose, (list, tuple)):
        if not grasp_pose:
            raise PlanningFailed("plan_grasp_motion: empty input grasp_pose")
        grasp_pose = grasp_pose[0]

    kwargs = dict(
        start_joint_position=start_joints,
        grasp_pose=grasp_pose,
        approach_axis=approach_axis,
        approach_distance=float(approach_distance),
        approach_in_tool_frame=bool(approach_in_tool_frame),
        lift_axis=lift_axis,
        lift_distance=float(lift_distance),
        lift_in_tool_frame=bool(lift_in_tool_frame),
    )
    if robot_file:
        kwargs["robot_file"] = robot_file

    try:
        plan = ctx.tool("curobo.plan_grasp_motion", **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise PlanningFailed(
            f"plan_grasp_motion: curobo.plan_grasp_motion call errored "
            f"({exc}). Requires cuRobo v0.8; check the curobo bundle is "
            f"the v0.8 build (v0.7 lacks this API)."
        ) from exc

    if not plan.get("success", False):
        reason = plan.get("failure_reason", "") or "(no reason given)"
        raise PlanningFailed(
            f"plan_grasp_motion: plan_grasp_motion returned success=False "
            f"({reason}). The free-space pre-grasp, the constrained "
            f"pre-grasp→grasp segment, or the constrained lift was not "
            f"feasible from the start configuration "
            f"(approach_axis={approach_axis!r}, "
            f"approach_distance={approach_distance}, "
            f"lift_axis={lift_axis!r}, lift_distance={lift_distance}). "
            f"This is a grasp-reachability problem, NOT perception. "
            f"Levers: a different approach/lift axis or distance, a "
            f"different grasp generator or pose, or a pre-grasp "
            f"repositioning move. (success=False is also returned by "
            f"cuRobo v0.7 — confirm the install is v0.8.)"
        )

    logger.info(
        "[plan_grasp_motion] plan_grasp_motion planned approach+grasp+lift "
        "(approach_axis=%s dist=%s, lift_axis=%s dist=%s)",
        approach_axis, approach_distance, lift_axis, lift_distance,
    )
    return {
        "approach_trajectory": plan["approach_trajectory"],
        "grasp_trajectory": plan["grasp_trajectory"],
        "lift_trajectory": plan["lift_trajectory"],
    }
