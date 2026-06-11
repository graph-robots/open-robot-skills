"""Collision-aware variant of ``waypoint_move``: lift + lateral move with a
rebuilt world model and the held object attached as a single virtual link.

Drop-in replacement for ``waypoint_move.py`` — same signature, same drop
target (``drop_x``, ``drop_y`` → ``lift_z`` with the canonical top-down
rotation), same downstream contract (``robot.execute_trajectory`` then
return ``{"done": True}``). The only behavioral difference is that this
node:

1. Reads a fresh observation and rebuilds the world via
   ``geometry.build_world_config`` (no target isolation — the held object
   is allowed to remain in the mesh; ``curobo.plan_with_grasped_object``
   can remove it inline via ``remove_obstacles_from_world=True``).
2. Calls ``curobo.plan_with_grasped_object`` instead of
   ``curobo.plan_to_pose`` so the planner respects scene collisions AND
   treats the held object as a sphere attached to the gripper link —
   preserving the collision footprint of "table + walls + distractors +
   held object" without the planner thinking the held object is a free
   obstacle it can knock into.

When to swap from ``waypoint_move`` → ``waypoint_move_carve``: a repair
pass flips this node in when the ``transport`` stage's per-stage
pass-rate drops below the configured threshold, on the hypothesis that
the failure mode is "free-space lift/translate plowed into a known
obstacle the perceived OBBs already captured". Authoring the node
manually is also fine when the coordinator sees a known obstacle on the
transport path at compose time (e.g. an oven door, a shelf above the
table).
"""

from typing import TypedDict

from gap import NodeContext
from gap.errors import PlanningFailed
from gap.types import Quaternion, Se3Pose


class Output(TypedDict):
    done: bool


_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
_LIFT_Z_M = 0.45


def run(
    ctx: NodeContext,
    drop_x: float,
    drop_y: float,
    surface_sphere_radius: float = 0.04,
    robot_distance_threshold: float = 0.20,
) -> Output:
    obs = ctx.tool("robot.get_observation")
    start_joints = obs["arms"][0]["joint_state"]

    world = ctx.tool(
        "geometry.build_world_config",
        cameras=obs["cameras"],
        robot_joint_state=start_joints,
        robot_distance_threshold=robot_distance_threshold,
    )

    target_pose: Se3Pose = {
        "position": {"x": float(drop_x), "y": float(drop_y), "z": _LIFT_Z_M},
        "rotation": _DOWN,
    }

    plan = ctx.tool(
        "curobo.plan_with_grasped_object",
        world_config=world["config"],
        start_joint_position=start_joints,
        target_pose=target_pose,
        object_name="",
        remove_obstacles_from_world=False,
        surface_sphere_radius=float(surface_sphere_radius),
    )
    if not plan["success"]:
        raise PlanningFailed(
            f"waypoint_move_carve: CuRobo plan_with_grasped_object failed for drop "
            f"XY=({drop_x:.3f}, {drop_y:.3f}) at lift_z={_LIFT_Z_M:.3f}"
        )

    ctx.tool(
        "robot.execute_trajectory",
        trajectory=plan["trajectory"],
        subsample=4,
        max_steps_per_waypoint=8,
        tolerance=0.02,
    )
    return {"done": True}
