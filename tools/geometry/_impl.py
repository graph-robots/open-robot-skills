"""Geometry math behind the geometry tool bundle.

Back-projection, DBSCAN filtering, the HyRL OBB fitting pipeline,
grasp-pose derivations, and world-config reconstruction, all operating on
:mod:`gap.types` TypedDicts + numpy arrays. Heavy optional deps (open3d,
scikit-learn, cv2) are imported inside the functions that need them so
importing this module stays cheap and the bundle loads without the
``geometry`` extra installed.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from gap_core.errors import PerceptionFailed, PlanningFailed, ToolError
from gap_core.types import (
    CameraFrame,
    OrientedBoundingBox,
    Se3Pose,
    Vec3,
    pose_to_matrix,
)
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Robot point exclusion via FK (matching HyRL's robot segmenter concept)
# ---------------------------------------------------------------------------

# Franka Panda approximate DH link positions (from base to EE).
# Each link is modeled as a sphere at the FK-computed position.
# These radii are approximate and slightly oversized to ensure good exclusion.
_FRANKA_LINK_RADII = [0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]


def _franka_fk_link_positions(joint_positions: np.ndarray) -> list[np.ndarray]:
    """Compute approximate Franka Panda link positions via simplified FK.

    Uses the standard Franka Panda DH parameters to compute the position
    of each link frame in the world (robot base) frame.  This is a simplified
    model (only positions, not orientations of intermediate frames beyond
    what's needed) — sufficient for sphere-based point exclusion.

    Args:
        joint_positions: (7,) joint angles in radians.

    Returns:
        List of (3,) position arrays for each link frame.
    """
    q = np.asarray(joint_positions, dtype=np.float64).flatten()[:7]

    # Franka Panda DH parameters (Modified DH convention):
    #   a_i, d_i, alpha_i (radians)
    # Reference: Franka Emika Panda datasheet
    dh_params = [
        (0.0,    0.333,  0.0),       # joint 1
        (0.0,    0.0,   -np.pi/2),   # joint 2
        (0.0,    0.316,  np.pi/2),   # joint 3
        (0.0825, 0.0,    np.pi/2),   # joint 4
        (-0.0825, 0.384, -np.pi/2),  # joint 5
        (0.0,    0.0,    np.pi/2),   # joint 6
        (0.088,  0.0,    np.pi/2),   # joint 7
    ]

    def _mdh_transform(a, d, alpha, theta):
        """Modified DH transform matrix."""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,      -st,      0,     a],
            [st*ca,    ct*ca,  -sa,  -sa*d],
            [st*sa,    ct*sa,   ca,   ca*d],
            [0,        0,       0,     1],
        ])

    T = np.eye(4)
    positions = [T[:3, 3].copy()]  # base position
    for i, (a, d, alpha) in enumerate(dh_params):
        T = T @ _mdh_transform(a, d, alpha, q[i])
        positions.append(T[:3, 3].copy())

    return positions


def _exclude_robot_points(
    points: np.ndarray,
    joint_positions: np.ndarray,
    distance_threshold: float = 0.15,
) -> np.ndarray:
    """Remove points near the robot body using FK-based sphere exclusion.

    Mirrors HyRL's RobotSegmenter approach but uses simplified FK + sphere
    distance instead of CuRobo's CUDA-accelerated collision spheres.

    Args:
        points: (N, 3) world-frame points.
        joint_positions: (7,) joint angles in radians.
        distance_threshold: Distance (m) from any robot link center to exclude.

    Returns:
        (M, 3) filtered points with robot-near points removed.
    """
    if len(points) == 0:
        return points

    link_positions = _franka_fk_link_positions(joint_positions)
    keep = np.ones(len(points), dtype=bool)

    for i, link_pos in enumerate(link_positions):
        radius = _FRANKA_LINK_RADII[i] if i < len(_FRANKA_LINK_RADII) else 0.05
        effective_radius = radius + distance_threshold
        dists = np.linalg.norm(points - link_pos.reshape(1, 3), axis=1)
        keep &= (dists > effective_radius)

    return points[keep]


# ---------------------------------------------------------------------------
# Numpy / TypedDict converters (the former proto encode/decode layer)
# ---------------------------------------------------------------------------


def as_points(pc) -> np.ndarray:
    """gap.types PointCloud (or bare array) -> (N, 3) float32 array."""
    if isinstance(pc, dict):
        pc = pc.get("points")
    if pc is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(pc, dtype=np.float32)
    return arr.reshape(-1, 3)


def as_mask_bool(mask: np.ndarray) -> np.ndarray:
    """gap.types Mask (uint8 0/255 or bool, [H, W]) -> boolean array."""
    return np.asarray(mask) > 0


def vec3(v) -> Vec3:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def vec3_to_np(v: Vec3) -> np.ndarray:
    return np.array([v["x"], v["y"], v["z"]], dtype=np.float64)


def matrix_to_se3(T: np.ndarray) -> Se3Pose:
    """4x4 homogeneous matrix -> Se3Pose dict (wxyz rotation)."""
    q_xyzw = Rotation.from_matrix(np.asarray(T)[:3, :3]).as_quat()
    return {
        "position": vec3(np.asarray(T)[:3, 3]),
        "rotation": {
            "w": float(q_xyzw[3]),
            "x": float(q_xyzw[0]),
            "y": float(q_xyzw[1]),
            "z": float(q_xyzw[2]),
        },
    }


def _obb_parts(obb: OrientedBoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OBB dict -> (center (3,), half_extent (3,), rotation matrix (3,3))."""
    center = vec3_to_np(obb["center"])
    half_extent = vec3_to_np(obb["extent"])
    q = obb["orientation"]
    R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    return center, half_extent, R


def _depth_to_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a depth image to (N, 3) camera-frame points.

    Parameters
    ----------
    depth : (H, W) float32 array
    K : (3, 3) intrinsics matrix

    Returns
    -------
    points : (N, 3) float32 array – only pixels with depth > 0.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth.flatten()
    valid = Z > 0
    X = (u.flatten()[valid] - cx) * Z[valid] / fx
    Y = (v.flatten()[valid] - cy) * Z[valid] / fy
    return np.stack([X, Y, Z[valid]], axis=1).astype(np.float32)


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    pts_h = np.hstack([points, np.ones((len(points), 1), dtype=points.dtype)])
    return (T @ pts_h.T).T[:, :3].astype(np.float32)


def _project_obb_to_mask(
    obb: OrientedBoundingBox,
    camera: CameraFrame,
    inflation_m: float = 0.02,
    min_extent_m: float = 0.04,
) -> np.ndarray | None:
    """Project an OBB onto a camera image plane, returning a uint8 mask.

    Computes the 8 world-frame corners of the OBB, transforms them into the
    camera frame, projects via the pinhole intrinsics, and fills the
    axis-aligned bounding box of the projections (clipped to image bounds)
    as a 0/255 mask. Returns ``None`` if no corner projects in front of the
    camera.

    The OBB half-extents are inflated before projection — each axis becomes
    ``max(extent + inflation_m, min_extent_m)`` — so that degenerate OBBs
    (thin sliver axes from partial perception) still produce a mask that
    covers the object's true footprint. Without inflation, leftover pixels
    of the object leak into the ``scene`` mesh and block CuRobo's descent
    during grasping. Prefer passing an explicit ``object_masks`` entry over
    relying on this fallback — segmentation masks are pixel-accurate where
    OBB projection is at best a corner-AABB approximation.
    """
    center, extent, R = _obb_parts(obb)
    extent = np.maximum(extent + inflation_m, min_extent_m)

    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    corners_world = center[None, :] + (signs * extent[None, :]) @ R.T  # (8, 3)

    K = np.asarray(camera["intrinsics"], dtype=np.float64)
    T_cam_to_world = pose_to_matrix(camera["pose"])
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    corners_h = np.hstack([corners_world, np.ones((8, 1))])
    corners_cam = (T_world_to_cam @ corners_h.T).T[:, :3]

    z = corners_cam[:, 2]
    valid = z > 1e-3
    if not np.any(valid):
        return None

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * corners_cam[valid, 0] / z[valid] + cx
    v = fy * corners_cam[valid, 1] / z[valid] + cy

    H, W = np.asarray(camera["depth"]).shape
    u0 = int(max(0, np.floor(u.min())))
    u1 = int(min(W - 1, np.ceil(u.max())))
    v0 = int(max(0, np.floor(v.min())))
    v1 = int(min(H - 1, np.ceil(v.max())))
    if u1 < u0 or v1 < v0:
        return None

    mask_arr = np.zeros((H, W), dtype=np.uint8)
    mask_arr[v0:v1 + 1, u0:u1 + 1] = 255
    return mask_arr


# ---------------------------------------------------------------------------
# Top-down grasp pose computation
# ---------------------------------------------------------------------------
# Two variants, with distinct orientation strategies:
#   1. world_aligned     — fixed world-Z-down gripper; yaw is NOT tied to OBB.
#   2. short_axis_aligned — gripper yaw matches OBB's shorter horizontal axis
#                           so fingers close across the narrower dimension.
# Both share the same signature (obb + z_offset → Se3Pose) and the same
# table-Z safety clamp so callers can swap them freely.

# Table-clearance floor for the grasp Z position (meters). The LIBERO table
# top sits at world z=0 and the fingertip needs a thin pad's worth of
# clearance before the gripper closes — anything lower pins the object
# against the table.
_TABLE_Z_MIN = -0.05


def compute_top_down_grasp_world_aligned(
    obb: OrientedBoundingBox,
    z_offset: float,
) -> Se3Pose:
    """Top-down grasp with a fixed world-aligned gripper orientation.

    The gripper always points straight down along world -Z; yaw is NOT
    derived from the OBB. The grasp X/Y sits above the OBB centre and Z
    lands on the world-frame top surface plus ``z_offset``. Works well
    when the object's horizontal cross-section is roughly isotropic or
    when the caller plans to try multiple yaws externally.

    Formula: for OBB with centre C, half-extents (hx,hy,hz) and rotation
    R, the AABB (world-aligned) half-height along Z is
    ``|R[2,0]|*hx + |R[2,1]|*hy + |R[2,2]|*hz``.
    """
    center, half_extent, R = _obb_parts(obb)

    world_z_half_extent = float(np.sum(np.abs(R[2, :]) * half_extent))
    top_z = center[2] + world_z_half_extent

    grasp_quat_wxyz = (0.0, 1.0, 0.0, 0.0)

    raw_z = top_z + z_offset
    if raw_z < _TABLE_Z_MIN:
        logger.warning(
            "TopDownGrasp[world_aligned]: raw Z=%.4f below table min %.4f "
            "(world-Z full_extent=%.4f, z_offset=%.3f). Clamping.",
            raw_z, _TABLE_Z_MIN, world_z_half_extent * 2.0, z_offset,
        )
        raw_z = _TABLE_Z_MIN

    return {
        "position": {"x": float(center[0]), "y": float(center[1]), "z": float(raw_z)},
        "rotation": {
            "w": grasp_quat_wxyz[0],
            "x": grasp_quat_wxyz[1],
            "y": grasp_quat_wxyz[2],
            "z": grasp_quat_wxyz[3],
        },
    }


def compute_top_down_grasp_short_axis_aligned(
    obb: OrientedBoundingBox,
    z_offset: float,
) -> Se3Pose:
    """Top-down grasp with gripper yaw aligned to the OBB's shorter XY axis.

    The gripper still approaches along world -Z, but the fingers close
    across the object's shorter horizontal dimension (the "narrow" axis),
    which is usually the right choice for rectangular boxes and elongated
    objects. The grasp position uses the OBB's own vertical axis to find
    the top face rather than the world-AABB top.

    Falls back to a world-aligned top-down quaternion if the shorter
    horizontal OBB axis projects near-zero onto world XY.

    Kept next to :func:`compute_top_down_grasp_world_aligned` exactly as in
    the servicer (it was never wired to an RPC; same here — no tool).
    """
    center, half_extent, R = _obb_parts(obb)

    # Pick the OBB axis most aligned with world Z as the "vertical" axis.
    z_world = np.array([0.0, 0.0, 1.0])
    vertical_idx = int(np.argmax(np.abs(R.T @ z_world)))
    vertical_axis = R[:, vertical_idx].copy()
    if vertical_axis[2] < 0:
        vertical_axis = -vertical_axis
    top_center = center + vertical_axis * half_extent[vertical_idx]

    # Between the two remaining (horizontal) OBB axes, pick the shorter.
    horiz_indices = [i for i in range(3) if i != vertical_idx]
    i_a, i_b = horiz_indices
    short_idx = i_a if half_extent[i_a] < half_extent[i_b] else i_b

    short_axis_world = R[:, short_idx]
    short_xy = np.array([short_axis_world[0], short_axis_world[1], 0.0])
    short_xy_norm = float(np.linalg.norm(short_xy))

    if short_xy_norm < 1e-6:
        logger.warning(
            "TopDownGrasp[short_axis]: shorter horizontal OBB axis "
            "near-vertical (norm=%.2e); falling back to world-aligned yaw.",
            short_xy_norm,
        )
        grasp_quat_wxyz = (0.0, 1.0, 0.0, 0.0)
    else:
        tool_y = short_xy / short_xy_norm
        tool_z = np.array([0.0, 0.0, -1.0])
        tool_x = np.cross(tool_y, tool_z)
        tool_x = tool_x / np.linalg.norm(tool_x)
        R_grasp = np.column_stack([tool_x, tool_y, tool_z])
        q_xyzw = Rotation.from_matrix(R_grasp).as_quat()
        grasp_quat_wxyz = (
            float(q_xyzw[3]),
            float(q_xyzw[0]),
            float(q_xyzw[1]),
            float(q_xyzw[2]),
        )

    raw_z = top_center[2] + z_offset
    if raw_z < _TABLE_Z_MIN:
        logger.warning(
            "TopDownGrasp[short_axis]: raw Z=%.4f below table min %.4f "
            "(OBB vertical full_extent=%.4f, z_offset=%.3f). Clamping.",
            raw_z, _TABLE_Z_MIN, half_extent[vertical_idx] * 2.0, z_offset,
        )
        raw_z = _TABLE_Z_MIN

    return {
        "position": {
            "x": float(top_center[0]),
            "y": float(top_center[1]),
            "z": float(raw_z),
        },
        "rotation": {
            "w": grasp_quat_wxyz[0],
            "x": grasp_quat_wxyz[1],
            "y": grasp_quat_wxyz[2],
            "z": grasp_quat_wxyz[3],
        },
    }


def top_down_grasp_candidates(
    obb: OrientedBoundingBox,
    z_offset: float,
) -> list[Se3Pose]:
    """Return a fan of top-down grasp candidates around the OBB.

    The canonical 2-pose primary+alt are always the FIRST two entries,
    so pose[0] / pose[1] are stable anchors for callers and planners
    that prefer the leading candidates.

    Layout:
    - poses[0]: canonical primary (world-aligned top-down, world-Z down,
      z = top_z + z_offset).
    - poses[1]: canonical alt — primary rotated 90° around the gripper's
      LOCAL Z (the wrist direction cuRobo IK seeds are tuned against).
      Distinct from ``yaw=90°`` around world Z, which is a different
      wrist orientation.
    - poses[2..]: 8 yaws × 3 z offsets (24 enriched top-down) − the
      one already covered above (world-Z yaw=0 at the primary z), so
      the total at primary z stays at 8.
    - For very flat (full_h < 0.04) OBBs ONLY, append 4 ±30° pitch
      side-grasps so flat objects (LIBERO pudding box, where perception
      only sees the top face) still have a feasible grasp candidate.
      Tall OBBs get NO pitch candidates: the goalset planner happily
      converges onto a tilted pinch for a 13-15 cm bottle/carton, and a
      30°-tilted fingertip pair on a tall object bears gravity
      asymmetrically and slips during transport (measured on the G1
      grocery dev gate: every passing trial's executed grasp was a
      straight top-down candidate; the tall-object slip failures rode
      the tilted ones). FK of every passing trial's executed plan
      confirmed strictly vertical winners, so tall objects offer only
      top-down candidates.
    - Z is clamped to ``_TABLE_Z_MIN`` (5 mm above the LIBERO table)
      so a near-flat OBB doesn't yield a sub-table grasp that cuRobo
      rejects.

    The default primary ``z_offset`` is -0.04 m (the tool's signature
    default): fingertip 4 cm below the OBB top. With z_offset=0 the
    fingertip sits at the top surface and the fingers close above the
    object (silent empty-grip).
    """
    primary_z_offset = z_offset
    z_offsets = (primary_z_offset, primary_z_offset - 0.02, primary_z_offset + 0.02)
    # World-Z yaws used to fan additional candidates. Yaw=0 is the
    # canonical primary (so we skip it at primary_z to avoid a
    # duplicate), but at deeper / shallower z we keep all 8 yaws.
    yaws_deg_full = (0.0, 90.0, 45.0, 135.0, 180.0, 270.0, 225.0, 315.0)

    primary = compute_top_down_grasp_world_aligned(obb, primary_z_offset)
    # Canonical alt: primary rotated 90° around the gripper's LOCAL Z.
    gq = primary["rotation"]
    s = math.sqrt(0.5)
    alt_q = {
        "w": gq["w"] * s - gq["z"] * s,
        "x": gq["x"] * s + gq["y"] * s,
        "y": gq["y"] * s - gq["x"] * s,
        "z": gq["w"] * s + gq["z"] * s,
    }
    alt: Se3Pose = {"position": dict(primary["position"]), "rotation": alt_q}

    # World-aligned half-extent along Z (matches compute_top_down_grasp_world_aligned).
    center, half_extent, R = _obb_parts(obb)
    world_z_half_extent = float(np.sum(np.abs(R[2, :]) * half_extent))
    top_z = float(center[2]) + world_z_half_extent
    full_h = 2.0 * world_z_half_extent
    is_flat = full_h < 0.04

    cx = float(center[0])
    cy = float(center[1])
    base_q = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z) — gripper world-Z down

    def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
        rad = math.radians(yaw_deg)
        cz = math.cos(rad / 2.0)
        sz = math.sin(rad / 2.0)
        bw, bx, by, bz = base_q
        return (
            cz * bw - sz * bz,
            cz * bx - sz * by,
            cz * by + sz * bx,
            cz * bz + sz * bw,
        )

    poses: list[Se3Pose] = [primary, alt]

    # Enriched fan (excluding yaw=0 at primary_z which is already the
    # canonical primary; the world-Z yaw=90 at primary_z IS distinct
    # from the canonical alt — different wrist direction — so we keep
    # it). At deeper / shallower z we emit all 8 yaws.
    for z_off in z_offsets:
        z = max(top_z + z_off, _TABLE_Z_MIN)
        for yaw_deg in yaws_deg_full:
            if z_off == primary_z_offset and yaw_deg == 0.0:
                continue  # already in poses[0]
            rw, rx, ry, rz = _yaw_quat(yaw_deg)
            poses.append({
                "position": {"x": cx, "y": cy, "z": z},
                "rotation": {"w": rw, "x": rx, "y": ry, "z": rz},
            })

    # Only FLAT objects get the four ±30° pitch side-grasps at the
    # primary z_offset (perception often sees just the top face of a
    # flat box, leaving no straight-down pinch surface). Tall objects
    # are deliberately excluded — a tilted pinch on a tall bottle or
    # carton bears gravity asymmetrically and slips during transport
    # (measured; see the docstring above).
    if is_flat:
        z = max(top_z + z_offsets[0], _TABLE_Z_MIN)
        for axis_x, sign in ((True, +1), (True, -1), (False, +1), (False, -1)):
            rad = math.radians(30.0) * sign
            c = math.cos(rad / 2.0)
            ss = math.sin(rad / 2.0)
            qx, qy, qz, qw = (ss, 0.0, 0.0, c) if axis_x else (0.0, ss, 0.0, c)
            bw, bx, by, bz = base_q
            rw = qw * bw - qx * bx - qy * by - qz * bz
            rx = qw * bx + qx * bw + qy * bz - qz * by
            ry = qw * by - qx * bz + qy * bw + qz * bx
            rz = qw * bz + qx * by - qy * bx + qz * bw
            poses.append({
                "position": {"x": cx, "y": cy, "z": z},
                "rotation": {"w": rw, "x": rx, "y": ry, "z": rz},
            })

    return poses


def front_grasp_from_obb(
    obb: OrientedBoundingBox,
    approach_offset: float,
    approach_hint: Vec3 | None,
    z_offset: float,
) -> dict:
    """Compute front-approach grasp poses from an OBB.

    For drawer handle grasping:
    1. Identify handle bar axis (longest XY extent) vs approach axis (shortest).
    2. Resolve approach direction sign via hint or heuristic.
    3. Build gripper rotation: Z toward cabinet, Y vertical (fingers straddle bar).
    4. Compute grasp at OBB center, pre-grasp at standoff distance.
    5. slide_axis = -approach_direction (pull = away from cabinet).
    """
    center, half_extent, R_obb = _obb_parts(obb)

    # --- Step 1: Identify bar axis (longer XY) vs approach axis (shorter XY) ---
    # OBB Z (column 2) is always world-Z from ComputeOBB.
    if half_extent[0] >= half_extent[1]:
        bar_idx, approach_idx = 0, 1
    else:
        bar_idx, approach_idx = 1, 0

    if abs(half_extent[0] - half_extent[1]) < 0.005:
        logger.warning(
            "FrontGrasp: OBB XY extents nearly equal (%.4f vs %.4f); "
            "bar/approach axis assignment may be ambiguous.",
            half_extent[0], half_extent[1],
        )

    approach_axis_raw = R_obb[:, approach_idx].copy()

    logger.debug(
        "FrontGrasp: bar_idx=%d (half_ext=%.4f) approach_idx=%d (half_ext=%.4f)",
        bar_idx, half_extent[bar_idx], approach_idx, half_extent[approach_idx],
    )

    # --- Step 2: Resolve approach direction sign ---
    # approach_hint points from handle toward robot; default = center → origin.
    if approach_hint is not None:
        hint = vec3_to_np(approach_hint)
    else:
        hint = -center.copy()
        hint[2] = 0.0  # XY only

    hint_norm = np.linalg.norm(hint)
    if hint_norm < 1e-6:
        logger.warning(
            "FrontGrasp: hint vector near-zero (OBB at origin?); "
            "defaulting to -Y direction."
        )
        hint = np.array([0.0, -1.0, 0.0])
    else:
        hint = hint / hint_norm

    # toward_robot: from handle outward toward the robot
    toward_robot = approach_axis_raw.copy()
    if np.dot(toward_robot, hint) < 0:
        toward_robot = -toward_robot

    # approach_direction: direction EE moves to reach handle (into cabinet)
    approach_direction = -toward_robot
    # slide_axis: direction drawer slides when pulled (away from cabinet)
    slide_axis = toward_robot.copy()

    # --- Step 3: Build gripper rotation matrix ---
    # g_z = approach_direction (toward cabinet; matches CuRobo grasp_approach_linear_axis=2)
    # g_y = world Z up (fingers open vertically to straddle handle bar)
    # g_x = cross(g_y, g_z)
    g_z = approach_direction
    g_y = np.array([0.0, 0.0, 1.0])
    g_x = np.cross(g_y, g_z)
    g_x_norm = np.linalg.norm(g_x)

    if g_x_norm < 1e-6:
        # approach_direction is near-vertical — degenerate case
        raise PlanningFailed(
            "Approach direction is near-vertical; "
            "use geometry.top_down_grasp_from_obb instead."
        )

    g_x = g_x / g_x_norm
    # Re-derive g_y to ensure strict orthogonality
    g_y = np.cross(g_z, g_x)
    g_y = g_y / np.linalg.norm(g_y)

    R_grasp = np.column_stack([g_x, g_y, g_z])

    # --- Step 4: Compute positions ---
    grasp_pos = center.copy()
    grasp_pos[2] += z_offset

    pre_grasp_pos = grasp_pos - approach_direction * approach_offset

    q_xyzw = Rotation.from_matrix(R_grasp).as_quat()
    quat = {
        "w": float(q_xyzw[3]),
        "x": float(q_xyzw[0]),
        "y": float(q_xyzw[1]),
        "z": float(q_xyzw[2]),
    }

    return {
        "grasp_pose": {"position": vec3(grasp_pos), "rotation": quat},
        "pre_grasp_pose": {"position": vec3(pre_grasp_pos), "rotation": dict(quat)},
        "approach_direction": vec3(approach_direction),
        "slide_axis": vec3(slide_axis),
    }


# ---------------------------------------------------------------------------
# Point cloud filtering + OBB fitting (HyRL algorithms)
# ---------------------------------------------------------------------------


def filter_noise(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Filter noise from point cloud using DBSCAN clustering.

    Mirrors HyRL filter_noise: keeps ALL non-noise points (labels != -1),
    not just the largest cluster.
    """
    from sklearn.cluster import DBSCAN

    if len(points) == 0:
        return points

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    # Keep ALL non-noise points (matching HyRL's filter_noise).
    filtered = points[labels != -1]
    if len(filtered) == 0:
        # All points classified as noise – return original cloud.
        logger.warning("filter_noise: all points are noise, returning original")
        return points

    return filtered


def compute_obb(points: np.ndarray) -> OrientedBoundingBox:
    """Compute oriented bounding box from 3D points.

    Mirrors HyRL get_oriented_bounding_box_from_3d_points exactly:
    1. Statistical outlier removal + tiny noise injection.
    2. Project to XY, rasterize, morphological close.
    3. Contour extraction + polygon approximation.
    4. Min-width rectangle search over contour edges.
    5. Percentile-based extents (2nd/98th).
    6. R3d is identity-in-Z (no 3D tilt).
    """
    import cv2
    import open3d as o3d

    if len(points) < 4:
        raise PerceptionFailed(f"Need at least 4 points for OBB, got {len(points)}")

    # --- HyRL algorithm: get_oriented_bounding_box_from_3d_points ---

    # Outlier removal via statistical filter (with tiny noise to avoid
    # degenerate cases)
    points = points + np.random.normal(0, 0.0001, points.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd, _ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clean_pts = np.asarray(pcd.points)

    if len(clean_pts) < 4:
        raise PerceptionFailed(
            f"Too few points after outlier removal: {len(clean_pts)}"
        )

    pts_2d = clean_pts[:, :2]
    z_vals = clean_pts[:, 2]

    # Rasterize 2D projection into a binary mask
    margin = 0.005
    mins_2d = pts_2d.min(axis=0) - margin
    maxs_2d = pts_2d.max(axis=0) + margin
    resolution = 0.0005  # 0.5 mm per pixel
    pixel_coords = ((pts_2d - mins_2d) / resolution).astype(np.int32)
    h = int(np.ceil((maxs_2d[1] - mins_2d[1]) / resolution)) + 1
    w = int(np.ceil((maxs_2d[0] - mins_2d[0]) / resolution)) + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (pixel_coords[:, 1] < h) & (pixel_coords[:, 0] < w)
    mask[pixel_coords[valid, 1], pixel_coords[valid, 0]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Extract concave contour and approximate as polygon
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        # Fallback to Open3D OBB if contour extraction fails
        logger.warning("No contours found, falling back to Open3D OBB")
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        extent = obb.extent
        R = obb.R
        q_xyzw = Rotation.from_matrix(R).as_quat()
        return {
            "center": vec3(center),
            "extent": {
                "x": float(extent[0] / 2),
                "y": float(extent[1] / 2),
                "z": float(extent[2] / 2),
            },
            "orientation": {
                "w": float(q_xyzw[3]),
                "x": float(q_xyzw[0]),
                "y": float(q_xyzw[1]),
                "z": float(q_xyzw[2]),
            },
        }

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.015 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    verts = approx.reshape(-1, 2).astype(float)

    # Find min-width bounding rectangle using concave boundary edges
    min_width = float('inf')
    best_angle = 0.0
    for i in range(len(verts)):
        edge = verts[(i + 1) % len(verts)] - verts[i]
        if np.linalg.norm(edge) < 1e-6:
            continue
        angle = np.arctan2(edge[1], edge[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = pts_2d @ R2d.T
        width = rotated[:, 1].max() - rotated[:, 1].min()
        if width < min_width:
            min_width = width
            best_angle = angle

    # Build 3D OBB from min-width orientation
    cos_a, sin_a = np.cos(-best_angle), np.sin(-best_angle)
    R2d_best = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_2d = pts_2d @ R2d_best.T

    # Use 2nd / 98th percentile instead of strict min/max
    _LO, _HI = 2, 98
    mins = np.array([
        np.percentile(rotated_2d[:, 0], _LO),
        np.percentile(rotated_2d[:, 1], _LO),
        np.percentile(z_vals, _LO),
    ])
    maxs = np.array([
        np.percentile(rotated_2d[:, 0], _HI),
        np.percentile(rotated_2d[:, 1], _HI),
        np.percentile(z_vals, _HI),
    ])
    extent = maxs - mins  # full extent
    center_local = (mins + maxs) / 2.0

    R3d = np.eye(3)
    R3d[:2, :2] = R2d_best.T
    center = R3d @ center_local

    q_xyzw = Rotation.from_matrix(R3d).as_quat()

    # gap.types stores half-extents (same as the proto did)
    return {
        "center": vec3(center),
        "extent": {
            "x": float(extent[0] / 2),
            "y": float(extent[1] / 2),
            "z": float(extent[2] / 2),
        },
        "orientation": {
            "w": float(q_xyzw[3]),
            "x": float(q_xyzw[0]),
            "y": float(q_xyzw[1]),
            "z": float(q_xyzw[2]),
        },
    }


def mask_to_world_points(
    mask_bool: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_cam: np.ndarray,
) -> np.ndarray:
    """Back-project a 2D mask to 3D world points."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Select foreground pixels with valid depth (matching HyRL bounds).
    valid = mask_bool & ~np.isnan(depth) & ~np.isinf(depth) & (depth >= 0.015) & (depth <= 20.0)
    Z = depth[valid]
    U = u[valid]
    V = v[valid]

    if len(Z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    X = (U - cx) * Z / fx
    Y = (V - cy) * Z / fy
    pts_cam = np.stack([X, Y, Z], axis=1).astype(np.float32)

    # Transform camera-frame points to world frame.
    return _transform_points(pts_cam, T_cam)


def pixel_to_world_point(
    pixel_x: float,
    pixel_y: float,
    depth: np.ndarray,
    K: np.ndarray,
    T_cam: np.ndarray,
) -> np.ndarray:
    """Back-project a single pixel to a 3D world point."""
    px = int(round(pixel_x))
    py = int(round(pixel_y))

    H, W = depth.shape
    if py < 0 or py >= H or px < 0 or px >= W:
        raise ToolError(
            "geometry.pixel_to_world_point",
            f"Pixel ({px}, {py}) out of bounds for {W}x{H} image",
        )

    Z = float(depth[py, px])
    if Z <= 0:
        raise ToolError(
            "geometry.pixel_to_world_point",
            f"Invalid depth ({Z}) at pixel ({px}, {py})",
        )

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (px - cx) * Z / fx
    Y = (py - cy) * Z / fy

    # Camera-frame point -> world frame.
    pt_cam = np.array([X, Y, Z, 1.0], dtype=np.float64)
    pt_world = T_cam @ pt_cam
    return pt_world[:3]


def select_top_down_grasp(
    grasp_poses: list[Se3Pose],
    gripper_position: Vec3 | None,
) -> Se3Pose:
    """Select the most top-down oriented grasp from a set of candidates."""
    if len(grasp_poses) == 0:
        raise ToolError("geometry.select_top_down_grasp", "No grasp candidates provided")

    world_down = np.array([0.0, 0.0, -1.0])
    best_idx = 0
    best_angle = np.inf

    has_gripper = gripper_position is not None
    if has_gripper:
        gripper_pos = vec3_to_np(gripper_position)

    for i, pose in enumerate(grasp_poses):
        rot = pose["rotation"]
        R = Rotation.from_quat([rot["x"], rot["y"], rot["z"], rot["w"]]).as_matrix()
        # Grasp Z-axis in world frame.
        grasp_z = R[:, 2]
        angle = np.arccos(np.clip(np.dot(grasp_z, world_down), -1.0, 1.0))

        # Use distance as tiebreaker when gripper position is available.
        if angle < best_angle or (
            has_gripper
            and np.isclose(angle, best_angle, atol=1e-3)
            and np.linalg.norm(vec3_to_np(pose["position"]) - gripper_pos)
            < np.linalg.norm(
                vec3_to_np(grasp_poses[best_idx]["position"]) - gripper_pos
            )
        ):
            best_angle = angle
            best_idx = i

    return grasp_poses[best_idx]


# ---------------------------------------------------------------------------
# World-config reconstruction (depth → meshes)
# ---------------------------------------------------------------------------


def build_world_config(
    cameras: list[CameraFrame],
    object_masks: list[dict],
    *,
    voxel_size: float,
    noise_eps: float,
    noise_min_samples: int,
    table_z_threshold: float,
    mesh_alpha: float,
    robot_joint_state,
    robot_distance_threshold: float,
    target_obb: OrientedBoundingBox | None,
    target_obb_name: str,
) -> tuple[dict, list[str]]:
    """Build a planner-agnostic collision world from camera observations.

    Pipeline (verbatim from the servicer): merge depth clouds → voxel
    downsample → FK-based robot exclusion → DBSCAN largest-cluster filter →
    table-plane removal → object-mask point marking → alpha-shape scene mesh.

    Returns ``(WorldConfig dict, mesh_names)``.
    """
    import open3d as o3d
    from sklearn.cluster import DBSCAN

    table_z = table_z_threshold

    # If caller didn't supply explicit object_masks but did supply a
    # target_obb, derive a 2D mask by projecting the OBB onto the first
    # camera. This lets callers isolate a known target without running
    # segmentation twice.
    object_masks = list(object_masks)
    if len(object_masks) == 0 and target_obb is not None and len(cameras) > 0:
        mask_arr = _project_obb_to_mask(target_obb, cameras[0])
        if mask_arr is not None:
            target_name = target_obb_name or "target"
            object_masks.append(
                {"name": target_name, "mask": mask_arr, "camera_index": 0}
            )
            logger.info(
                "build_world_config: derived target mask from OBB for '%s'",
                target_name,
            )

    # ----- Step 1: Merge all camera depth images into a single world cloud -----
    all_points = []
    for cam in cameras:
        depth = np.asarray(cam["depth"], dtype=np.float32)
        K = np.asarray(cam["intrinsics"], dtype=np.float64)
        pts_cam = _depth_to_points(depth, K)
        if len(pts_cam) == 0:
            continue
        T_cam = pose_to_matrix(cam["pose"])
        pts_world = _transform_points(pts_cam, T_cam)
        all_points.append(pts_world)

    if len(all_points) == 0:
        logger.warning("build_world_config: no valid points from any camera")
        return {"meshes": []}, []

    merged = np.concatenate(all_points, axis=0)

    # ----- Step 2: Voxel downsample -----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size)
    merged = np.asarray(pcd.points).astype(np.float32)

    # ----- Step 2b: Robot point exclusion (matching HyRL) -----
    # _exclude_robot_points uses Franka-specific FK; skip for non-7-DOF robots.
    if robot_joint_state is not None:
        joint_positions = np.asarray(
            robot_joint_state["positions"]
            if isinstance(robot_joint_state, dict)
            else robot_joint_state,
            dtype=np.float64,
        ).flatten()
        if len(joint_positions) >= 7 and len(merged) > 0:
            before_count = len(merged)
            merged = _exclude_robot_points(
                merged, joint_positions, robot_distance_threshold
            )
            logger.info(
                "build_world_config: %d -> %d points after robot exclusion "
                "(threshold=%.3f)",
                before_count, len(merged), robot_distance_threshold,
            )

    # ----- Step 3: DBSCAN noise filtering -----
    if len(merged) > 0:
        labels = DBSCAN(eps=noise_eps, min_samples=noise_min_samples).fit_predict(
            merged
        )
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique) > 0:
            largest = unique[np.argmax(counts)]
            merged = merged[labels == largest]

    # ----- Step 4: Remove table / ground plane below threshold -----
    if table_z != 0 and len(merged) > 0:
        above_table = merged[:, 2] >= table_z
        merged = merged[above_table]

    # ----- Step 5: Mark object points (consumed by scene-mesh exclusion) -----
    # Per-object meshes are no longer emitted from this pipeline. We keep
    # the mask → world-point projection here only to flag which points
    # the scene background mesh in step 6 should exclude.
    collision_meshes: list[dict] = []
    mesh_names: list[str] = []
    object_point_indices: set[int] = set()

    for entry in object_masks:
        cam_idx = int(entry.get("camera_index", 0))
        if cam_idx < 0 or cam_idx >= len(cameras):
            logger.warning(
                "build_world_config: invalid camera_index %d for mask '%s'",
                cam_idx, entry.get("name"),
            )
            continue

        cam = cameras[cam_idx]
        mask = as_mask_bool(entry["mask"])
        depth = np.asarray(cam["depth"], dtype=np.float32)
        K = np.asarray(cam["intrinsics"], dtype=np.float64)
        T_cam = pose_to_matrix(cam["pose"])

        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        if mask.shape != depth.shape:
            logger.warning(
                "build_world_config: mask shape %s != depth shape %s for '%s', skipping",
                mask.shape, depth.shape, entry.get("name"),
            )
            continue

        valid = mask & ~np.isnan(depth) & ~np.isinf(depth) & (depth >= 0.015) & (depth <= 20.0)
        Z = depth[valid]
        U = u[valid]
        V = v[valid]

        if len(Z) == 0:
            continue

        X = (U - cx) * Z / fx
        Y = (V - cy) * Z / fy
        pts_cam_obj = np.stack([X, Y, Z], axis=1).astype(np.float32)
        pts_world_obj = _transform_points(pts_cam_obj, T_cam)

        # Remove table-plane points from object cloud as well.
        if table_z != 0:
            pts_world_obj = pts_world_obj[pts_world_obj[:, 2] >= table_z]

        if len(pts_world_obj) < 4:
            logger.warning(
                "build_world_config: too few points (%d) for object '%s'",
                len(pts_world_obj), entry.get("name"),
            )
            continue

        # Mark these world points as "consumed" so step 6's scene
        # background mesh doesn't double-cover the object.
        if len(merged) > 0:
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(
                pts_world_obj.astype(np.float64)
            )
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(
                merged.astype(np.float64)
            )
            dists = np.asarray(merged_pcd.compute_point_cloud_distance(obj_pcd))
            close_mask = dists < voxel_size * 3
            object_point_indices.update(np.where(close_mask)[0].tolist())

    # ----- Step 6: Scene mesh from remaining points -----
    if len(merged) > 0:
        scene_mask = np.ones(len(merged), dtype=bool)
        for idx in object_point_indices:
            if idx < len(scene_mask):
                scene_mask[idx] = False
        scene_pts = merged[scene_mask]

        if len(scene_pts) >= 4:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(
                scene_pts.astype(np.float64)
            )
            try:
                scene_mesh = (
                    o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        scene_pcd, mesh_alpha
                    )
                )
                scene_mesh.compute_vertex_normals()
                s_verts = np.asarray(scene_mesh.vertices).astype(np.float32)
                s_faces = np.asarray(scene_mesh.triangles).astype(np.int32)

                if len(s_verts) > 0 and len(s_faces) > 0:
                    collision_meshes.append({
                        "name": "scene",
                        "vertices": s_verts,
                        "faces": s_faces,
                        "pose": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                        },
                    })
                    mesh_names.append("scene")
            except Exception as exc:
                logger.warning(
                    "build_world_config: scene alpha shape failed: %s", exc
                )

    return {"meshes": collision_meshes}, mesh_names
