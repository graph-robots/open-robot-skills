"""Refine a top-down grasp against the hand that is actually on the arm.

Two corrections, both read off ``robot.describe_gripper`` rather than assumed:

1. **Close-axis rotation.** The grasp rotation is rebuilt so the hand's
   declared closing axis lies along the shorter horizontal axis of the target
   OBB. ``robot.grasp_frame`` composes that orientation from the hand's
   measured approach and close axes and its TCP twist, so it is right for a
   hand that closes along tool-local x as well as one that closes along
   tool-local y -- a literal quaternion is right for exactly one of them, and
   a hand whose TCP carries a twist would otherwise lock the jaws a quarter
   turn away from the object. Working from the OBB's own axes also handles an
   arbitrarily rotated object and avoids taking the first (world-aligned) pose
   of a generic grasp-candidate fan.
2. **Fingertip floor.** The TCP is kept high enough that the fingertips below
   it do not enter the support surface: ``support_z + finger.reach_m +
   finger.clearance_m``. Thin objects otherwise command a geometrically
   centred grasp that jams the jaws against the table before they can close.
   Applied only when the hand states its finger envelope, so a hand that
   never measured it is left alone rather than trusted to a zero.

Returns the refined ``grasp_pose``. Feed it to ``compute_align_pose`` (the
pre-rotated hover) and to the descend state, so hover and grasp share one
rotation.

**Opt-in: a packed-tray grasp.** Everything below is off unless its input is
given, and none of it makes a tool call, so a graph that binds only
``grasp_pose`` and ``target_obb`` gets the calls and the pose it always got.
These came from the sharps disposal graph (sweep s8: 16/30 cluttered trays
fully solved, 152/176 syringes posted), where a thin barrel is taken from
among neighbours 37-49 mm away:

- ``support_z``: the perceived support height (e.g. a depth ring beside the
  object) replaces the OBB bottom in the fingertip floor. The OBB bottom
  follows floor pixels mixed into the target cloud; on ``tray_clutter_t03`` it
  sat 1.5 mm above the tray floor and the pads drove the syringe into it.
- ``preclose_max_width_m`` (+ ``jaw_clearance_m``): descend with the jaw
  pre-closed. Fully open, a 2F-85's fingertip band spans +/-49 mm across the
  jaw; ``tray_clutter_t16``/``t17`` have neighbours 37-49 mm away, the hand
  stopped 9 mm high on a neighbour and the grasp was refused. The width is
  ``2 * (jaw_clearance_m - finger_band_past_jaw_m - preclose_neighbour_margin_m)``
  (the widest when no clearance is measured), clipped to
  ``[object_width_m + preclose_object_margin_m, preclose_max_width_m]``, and
  returned as ``preclose_width_m`` for the execute state's ``robot.set_grip``.
  ``0.0`` means "descend open", which is the default.
- ``fingertip_beyond_tcp_m``: a ``[[jaw gap, fingertip distance beyond the
  TCP], ...]`` table for this hand. The fingertips of a linkage hand swing down
  as it closes (2F-85 on ``tray_clutter_t03``: +5.9 mm beyond the TCP fully
  open, +17.0 mm at a 30 mm gap), so a pre-closed descent raises the TCP by
  ``beyond(preclose_width) - beyond(widest gap in the table)``. The table is
  hand data and ``robot.describe_gripper`` does not report it, so it is a
  parameter with no default -- without it no drop is applied.
- ``width_percentile`` (+ ``target_cloud``): ``object_width_m`` from the
  cloud's transverse radius about the OBB's long axis at this percentile, times
  two. The percentile rejects sparse wider features (a syringe's flange and
  cap); the graph used 35. An estimate outside ``width_range_m`` falls back to
  ``object_width_m`` (or the OBB's short width), for when transparent RGB-D is
  too incomplete for a stable estimate.
- ``end_feature_center`` + ``station_offset_m``: move the grasp XY to a known
  distance along the object's long axis from a perceived end feature, pointing
  away from it. Measured over twelve episodes (2026-09-09) the syringe's
  centre of mass sits 43.0 +/- 0.2 mm from the perceived needle-cap centre,
  while the OBB centre is 14-17 mm off it (the cloud misses ~35 mm of the thin
  cap end) and the pinch landed 15-25 mm from the CoM; gravity's lever about
  the pinch axis pivoted the syringe 33 deg during the wrist turn on ``t08``.
  Anchored to the feature, the lever is gone. The axis origin is the cloud
  median when ``target_cloud`` has ``station_min_points``, else the OBB centre;
  the rotation and the height are still this script's own (close axis through
  ``robot.grasp_frame``, fingertip floor).

The additive outputs ``object_width_m``, ``preclose_width_m`` and
``grasp_station`` (``"candidate"`` or ``"end_feature_offset"``) are there for
``execute_grasp_align``'s pre-close and close.
"""

from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import OrientedBoundingBox, PointCloud, Se3Pose, Vec3
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    grasp_pose: Se3Pose
    object_width_m: float
    preclose_width_m: float
    grasp_station: str


def _fingertip_drop(table: list[list[float]], width: float) -> float:
    """TCP rise that keeps a pre-closed fingertip where the open one would be."""
    rows = sorted((float(gap), float(beyond)) for gap, beyond in table)
    gaps, beyond = zip(*rows, strict=True)
    return float(np.interp(width, gaps, beyond)) - beyond[-1]


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    target_obb: OrientedBoundingBox,
    support_z: float | None = None,
    target_cloud: PointCloud | None = None,
    object_width_m: float | None = None,
    width_percentile: float | None = None,
    width_range_m: list[float] | None = None,
    jaw_clearance_m: float | None = None,
    preclose_max_width_m: float | None = None,
    preclose_object_margin_m: float = 0.006,
    preclose_neighbour_margin_m: float = 0.002,
    finger_band_past_jaw_m: float | None = None,
    fingertip_beyond_tcp_m: list[list[float]] | None = None,
    end_feature_center: Vec3 | None = None,
    station_offset_m: float | None = None,
    station_min_points: int = 30,
) -> Output:
    refined: Se3Pose = {
        "position": dict(grasp_pose["position"]),
        "rotation": dict(grasp_pose["rotation"]),
    }
    gripper = ctx.tool("robot.describe_gripper")

    # The OBB's shorter horizontal axis is where the jaws should close.
    q = target_obb["orientation"]
    obb_rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([target_obb["extent"][k] for k in ("x", "y", "z")], dtype=np.float64)
    vertical = int(np.argmax(np.abs(obb_rotation.T @ np.array([0.0, 0.0, 1.0]))))
    horizontal = [i for i in range(3) if i != vertical]
    short = min(horizontal, key=lambda i: extents[i])
    short_axis = obb_rotation[:, short].copy()
    short_axis[2] = 0.0
    if np.linalg.norm(short_axis) > 1.0e-6:
        short_axis /= np.linalg.norm(short_axis)
        # ``close_axis`` is measured in the physical hand frame while commanded
        # poses are stated at the TCP; the connector composes the two, so the
        # heading is asked for as a world direction rather than rotated by hand.
        heading_deg = float(np.degrees(np.arctan2(short_axis[1], short_axis[0])))
        grasp_frame = ctx.tool(
            "robot.grasp_frame",
            approach={"x": 0.0, "y": 0.0, "z": -1.0},
            close_heading_deg=heading_deg,
        )
        refined["rotation"] = dict(grasp_frame["rotation"])

    # -- opt-in: cloud-derived object width and end-feature station ----------
    long = next(i for i in horizontal if i != short)
    long_axis = obb_rotation[:, long].copy()
    obb_center = np.array([target_obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    points = (
        np.asarray(target_cloud["points"], dtype=np.float64).reshape(-1, 3)
        if target_cloud is not None
        else np.zeros((0, 3))
    )
    width = float(object_width_m) if object_width_m is not None else 2.0 * float(extents[short])
    if width_percentile is not None and len(points):
        origin = np.median(points, axis=0)
        offsets = points - origin
        transverse = offsets - np.outer(offsets @ long_axis, long_axis)
        observed = 2.0 * float(np.percentile(np.linalg.norm(transverse, axis=1), float(width_percentile)))
        low, high = (width_range_m or (0.0, np.inf))[:2]
        if float(low) <= observed <= float(high):
            width = observed

    grasp_station = "candidate"
    if end_feature_center is not None and station_offset_m is not None:
        axis = long_axis.copy()
        axis[2] = 0.0
        if np.linalg.norm(axis) > 1.0e-6:
            axis /= np.linalg.norm(axis)
            origin = np.median(points, axis=0) if len(points) >= int(station_min_points) else obb_center
            feature = np.array([end_feature_center[k] for k in ("x", "y", "z")], dtype=np.float64)
            # OBB axis signs are arbitrary: point the axis from the feature
            # into the body, then project the feature onto that line.
            if float(axis @ (obb_center - feature)) < 0.0:
                axis = -axis
            anchor = origin + axis * float((feature - origin) @ axis)
            station = anchor + float(station_offset_m) * axis
            refined["position"]["x"] = float(station[0])
            refined["position"]["y"] = float(station[1])
            grasp_station = "end_feature_offset"

    preclose = 0.0
    if preclose_max_width_m is not None:
        band = (
            float(finger_band_past_jaw_m)
            if finger_band_past_jaw_m is not None
            # Pad centres stand this far outside the gap on each side; on the
            # 2F-85 (0.104 - 0.085) / 2 = 9.5 mm against the 8.1-8.3 mm the
            # graph measured from recorded pad poses.
            else max(0.0, 0.5 * (float(gripper.get("open_footprint_m", 0.0)) - float(gripper.get("span_m", 0.0))))
        )
        fit = (
            float(preclose_max_width_m)
            if jaw_clearance_m is None
            else 2.0 * (float(jaw_clearance_m) - band - float(preclose_neighbour_margin_m))
        )
        preclose = float(
            np.clip(fit, width + float(preclose_object_margin_m), float(preclose_max_width_m))
        )

    # Keep the fingertips below the TCP out of the support surface.
    finger = gripper.get("finger") or {}
    if finger.get("stated"):
        floor_z = (
            float(support_z)
            if support_z is not None
            else float(target_obb["center"]["z"]) - float(target_obb["extent"]["z"])
        )
        min_tcp_z = (
            floor_z + float(finger.get("reach_m", 0.0)) + float(finger.get("clearance_m", 0.0))
        )
        refined["position"]["z"] = max(float(refined["position"]["z"]), min_tcp_z)
    if preclose > 0.0 and fingertip_beyond_tcp_m:
        refined["position"]["z"] = float(refined["position"]["z"]) + _fingertip_drop(
            fingertip_beyond_tcp_m, preclose
        )
    return {
        "grasp_pose": refined,
        "object_width_m": float(width),
        "preclose_width_m": preclose,
        "grasp_station": grasp_station,
    }
