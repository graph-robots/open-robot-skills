"""Top-down grasp pose with gripper yaw locked to the OBB's shorter XY axis.

This is the deterministic, geometry-locked grasp-pose generator that makes
elongated-object grasps (pan handles, bottles, tools) robust: the gripper
approaches straight down (gripper Z = world -Z) but its finger-opening axis
(tool Y) is forced onto the OBB's *shorter* horizontal axis, so the jaws
close ACROSS the narrow dimension of the bar rather than along it. The
position uses the OBB's own vertical axis to find the top face, then
descends ``z_offset`` into it so the fingers wrap the bar instead of
skimming the top.

Ported verbatim from the proven pan pick-and-place geometry servicer
(``compute_top_down_grasp_short_axis_aligned``) so the capability is
self-contained in this skill bundle — it is pure numpy/scipy OBB math and
needs no model deployment.

Falls back to a world-aligned top-down quaternion if the shorter
horizontal OBB axis projects near-zero onto world XY (degenerate /
near-vertical handle).
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# Mirror the geometry-bundle table-Z clamp (LIBERO table top = world z=0).
_TABLE_Z_MIN = -0.05


class Output(TypedDict):
    grasp_pose: Se3Pose


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    z_offset: float = -0.04,
) -> Output:
    """Compute a single short-axis-aligned top-down grasp pose.

    Args:
        target_obb: OBB of the thing to grasp (e.g. the pan handle). Its
            ``extent`` is the gap half-extent convention.
        z_offset: signed offset added to the OBB top face Z. Negative
            (default -0.04) descends the fingertip 4 cm into the OBB top
            so the jaws wrap the bar instead of skimming it.

    Returns:
        ``{"grasp_pose": Se3Pose}`` — a fingertip-frame pose
        (``grasp_pose_is_fingertip=True`` downstream in plan_grasp).
    """
    c = target_obb["center"]
    e = target_obb["extent"]
    center = np.array([c["x"], c["y"], c["z"]])
    half_extent = np.array([e["x"], e["y"], e["z"]])

    q = target_obb["orientation"]
    R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()

    # Pick the OBB axis most aligned with world Z as the "vertical" axis.
    z_world = np.array([0.0, 0.0, 1.0])
    vertical_idx = int(np.argmax(np.abs(R.T @ z_world)))
    vertical_axis = R[:, vertical_idx].copy()
    if vertical_axis[2] < 0:
        vertical_axis = -vertical_axis
    top_center = center + vertical_axis * half_extent[vertical_idx]

    # Between the two remaining (horizontal) OBB axes, pick the shorter:
    # the gripper closes ACROSS this one.
    horiz_indices = [i for i in range(3) if i != vertical_idx]
    i_a, i_b = horiz_indices
    short_idx = i_a if half_extent[i_a] < half_extent[i_b] else i_b
    long_idx = i_b if short_idx == i_a else i_a

    short_axis_world = R[:, short_idx]
    short_xy = np.array([short_axis_world[0], short_axis_world[1], 0.0])
    short_xy_norm = float(np.linalg.norm(short_xy))

    if short_xy_norm < 1e-6:
        logger.warning(
            "short_axis_grasp_pose: shorter horizontal OBB axis "
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

    # Collision-aware floor: a top-down grasp must never target a point
    # BELOW the object's vertical centre. ``z_offset`` (default -0.04) is
    # tuned for tall objects (descend 4 cm from a high top to wrap the
    # side); for a THIN bar (e.g. a ~2 cm pan handle) that descent plunges
    # below the OBB entirely — into the support surface — which a
    # collision-aware planner (CuRobo v0.8 with the object mesh) correctly
    # refuses to plan into (empirically: such poses are 100% infeasible;
    # clamping to >= OBB centre is feasible). This is GENERAL: for tall
    # objects ``top + z_offset`` already stays above centre so behaviour is
    # unchanged; only over-descent on thin objects is corrected. The old
    # ``_TABLE_Z_MIN`` floor stays as a secondary safety net.
    obb_center_z = float(center[2])
    floor_z = max(_TABLE_Z_MIN, obb_center_z)
    if raw_z < floor_z:
        logger.warning(
            "short_axis_grasp_pose: raw Z=%.4f below collision-aware floor "
            "%.4f (OBB centre_z=%.4f, OBB vertical full_extent=%.4f, "
            "z_offset=%.3f). Clamping to OBB centre so the grasp stays in "
            "the object, not the support surface.",
            raw_z, floor_z, obb_center_z,
            half_extent[vertical_idx] * 2.0, z_offset,
        )
        raw_z = floor_z

    logger.info(
        "short_axis_grasp_pose: center=(%.4f,%.4f,%.4f) "
        "grasp=(%.4f,%.4f,%.4f) vertical_idx=%d short_idx=%d "
        "short_half_ext=%.4f long_half_ext=%.4f "
        "quat_wxyz=(%.4f,%.4f,%.4f,%.4f)",
        center[0], center[1], center[2],
        center[0], center[1], raw_z,
        vertical_idx, short_idx,
        half_extent[short_idx], half_extent[long_idx],
        grasp_quat_wxyz[0], grasp_quat_wxyz[1],
        grasp_quat_wxyz[2], grasp_quat_wxyz[3],
    )

    grasp_pose: Se3Pose = {
        "position": {
            "x": float(center[0]), "y": float(center[1]), "z": float(raw_z),
        },
        "rotation": {
            "w": grasp_quat_wxyz[0],
            "x": grasp_quat_wxyz[1],
            "y": grasp_quat_wxyz[2],
            "z": grasp_quat_wxyz[3],
        },
    }
    return {"grasp_pose": grasp_pose}
