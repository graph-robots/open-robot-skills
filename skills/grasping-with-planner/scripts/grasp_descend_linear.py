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

import numpy as np

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
_MAX_DEEPEN = 0.035       # cap = usable finger-pad depth. The Panda finger is ~5 cm
                          # from palm to tip: descending more than ~3.5 cm below the
                          # object top puts the PALM below the top and rams tall
                          # cartons/bottles over (observed on milk + ranch dressing).
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


def _refine_xy_at_hover(
    ctx: NodeContext, gx: float, gy: float,
    target_obb: OrientedBoundingBox,
) -> tuple[float, float]:
    """Re-center the grasp XY from the wrist camera at hover.

    One-sided exterior views bias the perceived OBB center toward the
    visible face (a milk carton read ~7 mm off-center with half its true
    depth) so the top-down pinch lands visibly off the centerline. At
    hover the eye-in-hand camera looks straight down at the object's TOP
    face — the one unbiased view — so segment the object under the
    camera and re-center on its top-slab centroid. Best-effort: any
    failure keeps the OBB-derived point."""
    try:
        obs = ctx.tool("robot.get_observation")
        cams = obs.get("cameras") or []
        if isinstance(cams, dict):
            cams = list(cams.values())
        cam = next(
            (c for c in cams if "eye_in_hand" in (c.get("name") or "")), None,
        )
        if cam is None:
            return gx, gy
        depth = np.asarray(cam["depth"])
        H, W = depth.shape[:2]
        # Seed pixel = the expected object TOP projected into the wrist
        # camera. The eye-in-hand optical axis is OFFSET from the TCP, so
        # the object is not at the image centre — a centre seed segments
        # the table instead (measured: 45k-point flat segment).
        c, e = target_obb["center"], target_obb["extent"]
        top_w = np.array([gx, gy, float(c["z"]) + float(e["z"])])
        q = cam["pose"]["rotation"]
        w_, x_, y_, z_ = (float(q[k]) for k in ("w", "x", "y", "z"))
        R = np.array([
            [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
            [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
            [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)],
        ])
        t = cam["pose"]["position"]
        p_cam = R.T @ (top_w - np.array([float(t["x"]), float(t["y"]), float(t["z"])]))
        if p_cam[2] <= 0.01:
            return gx, gy
        K = np.asarray(cam["intrinsics"], dtype=np.float64)
        u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        if not (0 <= u < W and 0 <= v < H):
            return gx, gy
        seg = ctx.tool(
            "sam3.segment_point",
            image=cam["rgb"], pixel_x=float(u), pixel_y=float(v),
        )
        if not seg.get("masks"):
            return gx, gy
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=seg["masks"][0], depth=cam["depth"],
            intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
        )["points"]
        pts = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        if len(pts) < 30:
            return gx, gy
        # Table rejection: an object segment is compact; a table segment
        # spans the workspace.
        if (pts[:, 0].max() - pts[:, 0].min() > 0.25
                or pts[:, 1].max() - pts[:, 1].min() > 0.25):
            return gx, gy
        z_hi = float(np.percentile(pts[:, 2], 95.0))
        slab = pts[pts[:, 2] > z_hi - 0.02]
        if len(slab) < 15:
            return gx, gy
        cx, cy = float(slab[:, 0].mean()), float(slab[:, 1].mean())
        # Sanity: the refined center must stay within the target's OBB
        # footprint (+3 cm) — otherwise the segmenter latched onto a
        # neighbour and the original point is safer.
        c, e = target_obb["center"], target_obb["extent"]
        if (abs(cx - float(c["x"])) > float(e["x"]) + 0.03
                or abs(cy - float(c["y"])) > float(e["y"]) + 0.03):
            return gx, gy
        if abs(cx - gx) < 0.003 and abs(cy - gy) < 0.003:
            return gx, gy
        logger.info(
            "[grasp] wrist hover re-center: (%.3f, %.3f) -> (%.3f, %.3f) "
            "(d = %.0f, %.0f mm)", gx, gy, cx, cy,
            (cx - gx) * 1000, (cy - gy) * 1000,
        )
        return cx, cy
    except Exception as exc:  # noqa: BLE001
        logger.warning("[grasp] hover re-center skipped: %s", exc)
        return gx, gy


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
    rx, ry = _refine_xy_at_hover(ctx, gx, gy, target_obb)  # Seg 1b: wrist top-down re-center (best-effort)
    if abs(rx - gx) > 0.003 or abs(ry - gy) > 0.003:
        _cartesian(ctx, rx, ry, hover_z, rot)
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
