"""VAB-style release: open the gripper, then a LINEAR retract straight up.

Opens with a settle so the object lands inside the basket, then retracts
vertically with ``robot.go_to_pose_cartesian`` (cuRobo ``plan_directed_linear``,
VAB segment 6) instead of the free-space ``robot.go_home`` swing. Leaving the arm
at hover above the basket is fine for the loop: the next grasp's rise/XY segments
carry it to the next object, and the agentview perception sees the table clearly.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Vec3

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    done: bool


def run(ctx: NodeContext, place_position: Vec3, hover_z: float = 0.353) -> Output:
    # Open + hold so the released object settles in the basket BEFORE retracting.
    ctx.tool("robot.open_gripper", settle_steps=60)
    p = place_position
    # Seg 6: straight-up retract to hover (Z only).
    ctx.tool(
        "robot.go_to_pose_cartesian",
        pose={"position": {"x": float(p["x"]), "y": float(p["y"]), "z": float(hover_z)},
              "rotation": _DOWN},
    )
    return {"done": True}
