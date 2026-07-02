"""Straight-Z placement INTO a walled container: lift (Z) → XY over the
container → LINEAR descend to inside the walls.

The generalized form of ``grocery_packing/transport_move.py`` (the reference
recipe that scores on the VAB packing suites), for any container with an
interior (basket, bin, box, tote). Lift/XY are straight-line cartesian
(``robot.go_to_pose_cartesian``); the descend uses cuRobo's AXIS-CONSTRAINED
linear move (``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``,
``orientation_mode="LOCK"``) — a guaranteed straight vertical drop. The descend
lowers the TCP to ``container_top − place_offset`` (with ``place_offset`` small
or negative to sit just above the rim so the held object clears the walls on the
way down and drops in) so the object is placed IN the container, not from a
lateral swing. For placement ONTO a surface or into a described sub-region/zone,
use the ``compute_drop_pose → waypoint_move → descend_release_linear`` path
instead — this script is the walled-container fast path.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Vec3

logger = logging.getLogger(__name__)

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    place_position: Vec3


def _cartesian(ctx: NodeContext, x: float, y: float, z: float) -> None:
    pose = {"position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": _DOWN}
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    except Exception as exc:  # noqa: BLE001
        # The straight-line solve (and its internal plan_to_pose fallback) can
        # fail to find a path from an awkward post-grasp config for a
        # hard-positioned item. A full planned move (no straight-line
        # constraint) reconfigures the arm to the same pose — keep
        # transporting rather than aborting the whole place.
        logger.warning(
            "[transport_move] straight-line leg to (%.3f, %.3f, %.3f) failed "
            "(%s); falling back to a planned go_to_pose.", x, y, z, exc,
        )
        ctx.tool("robot.go_to_pose", pose=pose)


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    # FINGERTIP-frame heights. Distance is from_z - target_z, NOT
    # get_ee_pose().z - target_z: get_ee_pose returns the panda_hand link
    # (~0.10 m above the fingertip), which overshoots the descent by the TCP
    # offset and rams the object into the basket floor.
    dist = float(from_z) - float(target_z)
    if dist <= 0.002:
        return
    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    res = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=js,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=dist,
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if res.get("success") and res.get("trajectory"):
        ctx.tool("robot.execute_trajectory", trajectory=res["trajectory"])
    else:
        ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]  # XY only for the fallback
        _cartesian(ctx, ee["x"], ee["y"], target_z)


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = 0.353,
    lift_z: float = 0.25,
    place_offset: float = 0.06,
) -> Output:
    c = container_obb["center"]
    e = container_obb["extent"]
    bx, by = float(c["x"]), float(c["y"])
    place_z = float(c["z"]) + float(e["z"]) - float(place_offset)
    cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    # First lift the grasped item straight up so it clears the table before the
    # lateral move (avoids dragging it across the scene). lift_z is kept BELOW
    # transport_z on purpose, but even a MODEST in-place lift can be infeasible
    # at a far-edge grasp XY (x~0.8): the arm grasps low there yet has almost no
    # vertical room, so BOTH the straight-line and the planned solve fail. That
    # is exactly why this leg was dropped before. So the lift is BEST-EFFORT: if
    # it can't be solved we skip it and go straight up-and-over -- Seg 4 lifts
    # the item anyway as it moves toward the central, reachable basket. The lift
    # must never abort the place; only Segs 4/5 are essential.
    try:
        _cartesian(ctx, cur["x"], cur["y"], lift_z)    # Seg 3: lift in place (best-effort)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[transport_move] in-place lift to z=%.3f infeasible at (%.3f, %.3f) "
            "(%s); skipping lift, going straight up-and-over.",
            lift_z, cur["x"], cur["y"], exc,
        )
    _cartesian(ctx, bx, by, transport_z)               # Seg 4: up & over the basket
    _descend_linear(ctx, place_z, transport_z)         # Seg 5: descend INTO basket (constrained Z)
    return {"place_position": {"x": bx, "y": by, "z": place_z}}
