"""Reorder top-down grasp candidates to prefer short-horizontal-axis alignment.

For a thin/elongated target (frypan handle, screwdriver, spoon, rod) the
two-finger gripper only encloses the object when the jaws close
*perpendicular* to the long axis — i.e. when the gripper's inter-finger
Y direction aligns with the OBB's *short horizontal* axis.

``geometry.top_down_grasp_candidates`` emits a yaw-fan of top-down poses
all centered on the OBB XY; it does NOT bias yaw to OBB extents.
`plan_grasp.py` then picks the first IK-feasible singleton, which on a
handle is usually a perpendicular grasp that closes on empty air.

This script reorders the candidate list so the most short-axis-aligned
poses come first. Original count and pose values are preserved — only
the ordering changes — so the downstream `plan_grasp.py` per-pose loop
still has full fallback set if the short-axis candidates are
IK-unreachable.

Definition of "short horizontal axis":
  1. Compute the world-frame OBB axes from the orientation quaternion.
  2. Take the two axes most-parallel to the world XY plane (smallest
     |z|-component). These are the two "horizontal" OBB axes.
  3. Of those two, pick the one with the smaller OBB half-extent.

The desired gripper-Y in world is then this axis projected onto XY.
Gripper-Y direction for a candidate pose is the second column of its
rotation matrix; alignment score is |dot(gripper_Y_xy, short_axis_xy)|
(the absolute value collapses ±direction so the jaws may close from
either side).
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
    poses: list[Se3Pose]
    short_axis_xy: tuple[float, float]


def _quat_to_R(q) -> np.ndarray:
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()


def _compute_short_horizontal_axis(target_obb: OrientedBoundingBox) -> np.ndarray:
    """Return the unit vector in world XY of the OBB's short horizontal axis."""
    R_obb = _quat_to_R(target_obb["orientation"])
    e = target_obb["extent"]
    half_ext = np.array([e["x"], e["y"], e["z"]], dtype=np.float64)
    z_component = np.abs(R_obb[2, :])
    horiz_order = np.argsort(z_component)
    i0, i1 = int(horiz_order[0]), int(horiz_order[1])
    short_idx = i0 if half_ext[i0] <= half_ext[i1] else i1
    axis_world = R_obb[:, short_idx].astype(np.float64).copy()
    axis_world[2] = 0.0
    n = float(np.linalg.norm(axis_world))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (axis_world / n)[:2]


def _gripper_y_xy(pose: Se3Pose) -> np.ndarray:
    R = _quat_to_R(pose["rotation"])
    v = R[:2, 1].astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0], dtype=np.float64)
    return v / n


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    candidate_poses: list[Se3Pose],
) -> Output:
    n_in = len(candidate_poses)
    if n_in == 0:
        raise RuntimeError("select_short_axis: empty candidate_poses")

    short_xy = _compute_short_horizontal_axis(target_obb)

    scored = []
    for i, p in enumerate(candidate_poses):
        gy = _gripper_y_xy(p)
        if np.linalg.norm(gy) < 1e-9:
            score = 0.0
        else:
            score = float(abs(np.dot(gy, short_xy)))
        scored.append((score, i, p))

    scored.sort(key=lambda s: s[0], reverse=True)
    reordered = [p for _, _, p in scored]

    top_scores = [s for s, _, _ in scored[: min(5, n_in)]]
    logger.info(
        "[select_short_axis] short_axis_xy=(%+.3f, %+.3f); reordered %d "
        "candidates by gripper-Y alignment; top-5 |cos(align)|=%s",
        float(short_xy[0]), float(short_xy[1]), n_in,
        ", ".join(f"{s:.3f}" for s in top_scores),
    )
    return {
        "poses": reordered,
        "short_axis_xy": (float(short_xy[0]), float(short_xy[1])),
    }
