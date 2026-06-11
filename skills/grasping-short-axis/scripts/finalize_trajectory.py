"""Drive the robot to the final waypoint of a trajectory with full convergence.

``robot.execute_trajectory`` with ``subsample=4, max_steps_per_waypoint=8``
(the time-budget tuning) leaves the last waypoint with ~0.1 rad joint lag
because the body PD never has enough sim steps to converge before the next
waypoint is sent. If ``robot.close_gripper`` fires immediately after, the
body PD is still actively driving toward the target and the simulator's
action-manager tick silently drops the binary close action — fingers stay
open and the grasp silently fails (the ``target_held`` postcondition
checkpoint catches it in rehearsal).

This script issues a single ``robot.move_to_joints`` to the trajectory's
last waypoint with full ``max_steps``, so the body PD is quiescent before
the gripper command. Only the *final* waypoint pays the convergence cost.

Insert it between ``execute`` and ``close`` in any grasp subgraph that
runs ``robot.execute_trajectory`` with a subsampled trajectory.
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import Trajectory


class Output(TypedDict):
    done: bool


def run(
    ctx: NodeContext,
    trajectory: Trajectory,
    tolerance: float = 0.01,
    max_steps: int = 120,
) -> Output:
    if not trajectory.get("waypoints"):
        raise RuntimeError("finalize_trajectory: empty trajectory")
    final = trajectory["waypoints"][-1]
    ctx.tool(
        "robot.move_to_joints",
        joint_config={"positions": list(final["positions"])},
        tolerance=tolerance,
        max_steps=max_steps,
    )
    return {"done": True}
