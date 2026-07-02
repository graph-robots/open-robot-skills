"""Grasp via a fast axis-locked linear descend, with a collision-aware fallback.

PRIMARY — cartesian fast path on the top-down candidate pose: rise (Z) → XY over
the object (rotating to the grasp yaw) → **Z-locked linear descend** onto it via
``curobo.plan_directed_linear`` (``allowed_axes=["Z"]``, ``orientation_mode="LOCK"``).
Pure-vertical, orientation-locked, zero lateral drift — and, unlike the planner's
goalset, it happily grips a *flat* object by simply lowering onto it.

FALLBACK — if the straight-line solve is infeasible (a far-edge item where the
fixed top-down wrist has no IK), hand off to the collision-aware cuRobo planner,
which searches the whole candidate fan for a reachable, collision-free wrist.

This is ``grocery_packing/grasp_move.py`` distilled to the general single-object
case (the packing-specific fused-OBB raised-wrist split is dropped). Depth is a
SHALLOW grip near the perceived top, floored a hair above the object base so the
fingers never ram the table — see ``_BASE_CLEARANCE``.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose

logger = logging.getLogger(__name__)

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

# Grasp depth tuning — ADAPTIVE to object height. A top-down grip near the
# perceived TOP tips tall cartons/bottles during the lift + lateral transport
# (the grip point sits well above the centre of mass); a grip near the object's
# vertical CENTRE is stable. So descend a height-proportional amount below the
# OBB top: flat boxes stay shallow (near-top), tall objects (milk/juice cartons,
# bottles) grip toward mid-height. Floored _BASE_CLEARANCE above the base so the
# fingers never ram the table, and the descent is capped so a very tall object
# isn't over-plunged. (grocery_packing used a FIXED shallow grip to compensate
# for a scene where perception UNDER-measured tall cartons; libero perception is
# accurate — validated 99.5% ID — so keying the depth off the true OBB height is
# safe and prevents the tall-object tip-over.)
_DEEPEN_FRAC = 0.5        # descend this fraction of the object's height below its top (0.5 → grip at centre)
_MAX_DEEPEN = 0.07        # cap the descent below the top (m), so a very tall object isn't over-plunged
_BASE_CLEARANCE = 0.012   # keep the grip at least this far above the object base (fingers clear the table)


class Output(TypedDict):
    done: bool


def _cartesian(ctx: NodeContext, x: float, y: float, z: float,
               rotation: dict | None = None) -> None:
    """Straight-line cartesian move. Raises on an infeasible solve — the
    caller's planner fallback handles that, so we do NOT silently retry here."""
    ctx.tool("robot.go_to_pose_cartesian",
             pose={"position": {"x": float(x), "y": float(y), "z": float(z)},
                   "rotation": rotation or _DOWN})


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float,
                    rotation: dict | None = None) -> None:
    """Z-only straight-down descend via cuRobo's constrained linear planner;
    cartesian fallback if the constrained plan can't be found. FINGERTIP-frame
    heights: distance is ``from_z - target_z`` (NOT ``get_ee_pose().z - target_z``,
    which returns panda_hand ~0.10 m above the fingertip and overshoots)."""
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
        ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]  # XY only (top-down: hand XY == fingertip XY)
        _cartesian(ctx, ee["x"], ee["y"], target_z, rotation)


def _cartesian_grasp(ctx: NodeContext, grasp_pose: Se3Pose,
                     target_obb: OrientedBoundingBox, hover_z: float) -> None:
    """Fast path: rise → XY over object (rotating to the grasp yaw) → descend.

    Grip depth is HEIGHT-ADAPTIVE (see the module constants): descend
    ``_DEEPEN_FRAC`` of the object's height below its OBB top (capped at
    ``_MAX_DEEPEN``), floored ``_BASE_CLEARANCE`` above the base. A flat box thus
    gets a shallow near-top grip that still clears the table, while a tall carton
    or bottle is gripped toward its centre of mass so it doesn't tip during the
    lift + lateral transport."""
    g = grasp_pose["position"]
    rot = grasp_pose.get("rotation") or _DOWN      # grasp orientation (top-down candidate yaw)
    gx, gy = float(g["x"]), float(g["y"])
    c_z = float(target_obb["center"]["z"])
    e_z = float(target_obb["extent"]["z"])
    top, base_z, height = c_z + e_z, c_z - e_z, 2.0 * e_z
    deepen = min(_MAX_DEEPEN, _DEEPEN_FRAC * height)  # deeper for tall objects, shallow for flat
    # For objects shorter than ~3.5 cm the fixed 12 mm base clearance
    # exceeds most of the object's height and the pinch engages only the
    # top few mm (butter / cream-cheese boxes slip out). Scale the floor
    # with height for those; objects taller than 0.35*h >= 12 mm keep the
    # exact tuned behavior.
    base_clearance = min(_BASE_CLEARANCE, max(0.006, 0.35 * height))
    grasp_z = max(top - deepen, base_z + base_clearance)  # toward CoM for tall; near-top+floored for flat
    cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, cur["x"], cur["y"], hover_z, _DOWN)   # Seg 0: rise to hover (keep down)
    _cartesian(ctx, gx, gy, hover_z, rot)                 # Seg 1: XY over object + rotate to grasp yaw
    _descend_linear(ctx, grasp_z, hover_z, rot)           # Seg 2: descend INTO it (clamped; LOCK keeps yaw)


def _planner_grasp(
    ctx: NodeContext,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    topk: int = 8,
    num_ik_seeds: int = 128,
) -> None:
    """Collision-aware fallback: build a per-observation collision world (target
    carved out by its OBB volume) and plan to the first reachable, collision-free
    candidate in the fan. Plans from the CURRENT config — no pre-positioning
    needed (the cuRobo solve searches IK seeds + the whole goalset)."""
    obs = ctx.tool("robot.get_observation")
    js = obs["arms"][0]["joint_state"]
    world = ctx.tool(
        "geometry.build_world_config",
        cameras=obs["cameras"],
        robot_joint_state=js,
        target_obb=target_obb,
        target_obb_name="target",
    )["config"]
    poses = list(candidate_poses)[: int(topk)]
    for i, pose in enumerate(poses):
        try:
            plan = ctx.tool(
                "curobo.plan_to_grasp_poses",
                world_config=world,
                start_joint_position=js,
                grasp_poses=[pose],
                grasp_pose_is_fingertip=True,
                use_world_collision=True,
                use_cuda_graph=False,
                robot_collision_sphere_buffer=-0.01,
                collision_activation_distance=0.005,
                ignore_obstacle_names=["target"],
                use_grasp_approach=False,
                num_ik_seeds=int(num_ik_seeds),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[grasp] planner candidate %d/%d errored (%s); next",
                           i, len(poses), exc)
            continue
        if plan.get("success"):
            ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
            logger.info("[grasp] planner fallback grasped via candidate %d/%d", i, len(poses))
            return
    raise RuntimeError(
        f"grasp: cartesian path infeasible AND collision-aware planner found "
        f"0/{len(poses)} reachable candidates"
    )


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    hover_z: float = 0.2,
) -> Output:
    try:
        _cartesian_grasp(ctx, grasp_pose, target_obb, hover_z)
        logger.info("[grasp] cartesian linear-descend fast-path grasp")
        return {"done": True}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[grasp] cartesian path could not be found (%s); falling back to "
            "the collision-aware planner over the candidate fan.", exc,
        )
        _planner_grasp(ctx, candidate_poses, target_obb)
        return {"done": True}
