"""Slide the grasp position outward along the handle, away from the body.

For elongated handles where the perceived OBB may include part of the
heavier object body (a pan bowl, a pot belly), grasping at the OBB
centroid puts the jaws near the body — a marginal grip that holds at
close but shears out under the lift acceleration. This script computes
the in-plane direction from the *body* centroid toward the *handle*
centroid and slides the grasp position that way by a fraction of the
handle's long half-extent, so the jaws land on the actual handle bar,
clear of the body. Orientation is left unchanged.

Ported from the proven pan pick-and-place workflow. Generalized:
``base_obb`` is OPTIONAL — when it is not wired (no separate body
perception was authored), the grasp pose is returned unchanged, so this
node is a safe no-op and the skill degrades to a plain short-axis grasp.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap.types import OrientedBoundingBox, Se3Pose
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


class Output(TypedDict):
    adjusted_grasp: Se3Pose


def run(
    ctx: NodeContext,
    handle_obb: OrientedBoundingBox,
    grasp_pose: Se3Pose,
    base_obb: OrientedBoundingBox | None = None,
    offset_fraction: float = 0.3,
) -> Output:
    """Offset the grasp away from the object body along the handle axis.

    Args:
        handle_obb: OBB of the handle/subpart being grasped.
        grasp_pose: Initial grasp pose (from short_axis_grasp_pose).
        base_obb: OBB of the heavier object body (e.g. pan base/bowl).
            OPTIONAL — when None, the grasp is returned unchanged.
        offset_fraction: Fraction of the handle long half-extent to
            slide outward. 0.3 is the proven value for a pan handle.

    Returns:
        ``{"adjusted_grasp": Se3Pose}`` — same orientation, position
        slid along the (body -> handle) in-plane direction.
    """
    if base_obb is None:
        logger.info(
            "offset_grasp_from_base: no base_obb wired; "
            "returning grasp unchanged (plain short-axis grasp)."
        )
        return {"adjusted_grasp": grasp_pose}

    bc = base_obb["center"]
    hc = handle_obb["center"]
    base_center = np.array([bc["x"], bc["y"], bc["z"]])
    handle_center = np.array([hc["x"], hc["y"], hc["z"]])

    base_to_handle = handle_center - base_center
    base_to_handle_xy = np.array([base_to_handle[0], base_to_handle[1], 0.0])
    distance_xy = float(np.linalg.norm(base_to_handle_xy))

    if distance_xy < 1e-6:
        logger.warning(
            "offset_grasp_from_base: base and handle centers near-"
            "coincident (%.4f m); no offset applied.", distance_xy
        )
        return {"adjusted_grasp": grasp_pose}

    direction = base_to_handle_xy / distance_xy

    q = handle_obb["orientation"]
    R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    he = handle_obb["extent"]
    half_extents = np.array([he["x"], he["y"], he["z"]])

    # Longest horizontal axis ~ the handle length.
    z_world = np.array([0.0, 0.0, 1.0])
    z_alignments = np.abs(R.T @ z_world)
    horizontal_indices = [i for i in range(3) if z_alignments[i] < 0.7]

    if len(horizontal_indices) >= 2:
        h0, h1 = horizontal_indices[0], horizontal_indices[1]
        long_idx = h0 if half_extents[h0] > half_extents[h1] else h1
        offset_distance = half_extents[long_idx] * offset_fraction
    else:
        offset_distance = float(np.max(half_extents)) * offset_fraction

    gp = grasp_pose["position"]
    grasp_pos = np.array([gp["x"], gp["y"], gp["z"]])
    adjusted_pos = grasp_pos + direction * offset_distance

    logger.info(
        "offset_grasp_from_base: base=(%.4f,%.4f,%.4f) "
        "handle=(%.4f,%.4f,%.4f) dir=(%.4f,%.4f) offset=%.4f "
        "orig=(%.4f,%.4f,%.4f) adj=(%.4f,%.4f,%.4f)",
        base_center[0], base_center[1], base_center[2],
        handle_center[0], handle_center[1], handle_center[2],
        direction[0], direction[1], offset_distance,
        grasp_pos[0], grasp_pos[1], grasp_pos[2],
        adjusted_pos[0], adjusted_pos[1], adjusted_pos[2],
    )

    adjusted_grasp: Se3Pose = {
        "position": {
            "x": float(adjusted_pos[0]),
            "y": float(adjusted_pos[1]),
            "z": float(adjusted_pos[2]),
        },
        "rotation": grasp_pose["rotation"],  # orientation unchanged
    }
    return {"adjusted_grasp": adjusted_grasp}
