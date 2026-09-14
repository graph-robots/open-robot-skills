"""Lift, reobserve the held object, and reject an empty or slipped grasp.

After a short Cartesian lift the object is looked for from the wrist camera
first and the overhead camera second, by ``object_description`` and, when
given, ``marker_description``. Confidence only admits a candidate: metric
depth, table clearance and hand proximity decide. A small held object can
occupy only a few dozen wrist pixels, so the score floor is deliberately low.

**Opt-in: finding the held one among look-alikes.** Off by default -- the
first query/camera whose top mask clears ``score_min`` decides, as before.
From the sharps disposal graph (sweep s8), where the lifted wrist looks down
on a tray still holding the held object's twins:

- ``nearest_to_hand``: consider every returned mask at or above ``score_min``,
  not just the top one, and judge the one nearest the lifted hand across all
  queries and cameras so far (returning as soon as it passes, in the same
  query order). From the lifted wrist the tray's syringes are large, well-lit
  and outscore the held one, which is foreshortened and seen end-on. On
  ``tray_clutter_t03`` (2026-09-13), with the syringe held 1.4 cm from the TCP,
  "syringe" returned three tray syringes (0.052/0.050/0.041, 217-281 mm from
  the hand) and the held one only fourth (0.040, 23 mm from the hand); the
  top-1 check rejected a successful grasp and the abort's open dropped it.
- ``max_masks_per_query``: how many masks ``sam3.segment_text`` returns per
  query (``0`` = all). With 6-8 syringes in the tray (``t00``/``t01``/``t07``,
  sweep s2) the 8 best were all tray syringes, nearest 215 mm from the hand,
  so a held one was declared dropped; the graph asked for all of them.
- ``max_span_m`` / ``max_width_m``: reject a candidate whose cloud is longer
  (97th - 3rd percentile along its principal axis) or wider (twice the 90th
  percentile radius across it) than one object. Keeps the arm itself -- a
  395k-px mask 176 mm from the hand in the same view -- from passing as the
  held object. The graph used 0.25 m and 0.06 m for a 150 mm syringe.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    route: str
    verified: bool
    observed_center: dict[str, float]
    point_count: int
    camera: str
    reason: str


def _not_held(
    reason: str, camera: str = "", point_count: int = 0, center: np.ndarray | None = None
) -> Output:
    if center is None:
        observed = {"x": 0.0, "y": 0.0, "z": 0.0}
    else:
        observed = {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}
    return {
        "route": "not_held",
        "verified": False,
        "observed_center": observed,
        "point_count": int(point_count),
        "camera": camera,
        "reason": reason,
    }


def _size(points: np.ndarray) -> tuple[float, float]:
    """Robust span along the principal axis and full width across it."""
    centered = points - np.median(points, axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    along = centered @ basis[0]
    across = centered - np.outer(along, basis[0])
    span = float(np.percentile(along, 97) - np.percentile(along, 3))
    width = 2.0 * float(np.percentile(np.linalg.norm(across, axis=1), 90))
    return span, width


def run(
    ctx: NodeContext,
    object_description: str,
    marker_description: str = "",
    lift_m: float = 0.04,
    score_min: float = 0.005,
    min_points: int = 20,
    min_above_table_m: float = 0.03,
    max_hand_distance_m: float = 0.20,
    wrist_camera_keyword: str = "eye_in_hand",
    overhead_camera_name: str = "overhead",
    max_masks_per_query: int = 3,
    nearest_to_hand: bool = False,
    max_span_m: float | None = None,
    max_width_m: float | None = None,
) -> Output:
    """Route ``verified`` when the object rides with the lifted hand, else ``not_held``."""
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    lifted = {"position": dict(ee["position"]), "rotation": dict(ee["rotation"])}
    lifted["position"]["z"] = float(lifted["position"]["z"]) + float(lift_m)
    ctx.tool("robot.go_to_pose_cartesian", pose=lifted)

    observation = ctx.tool("robot.get_observation")
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    wrist = next((c for c in cameras if wrist_camera_keyword in str(c.get("name", ""))), None)
    if wrist is None:
        raise RuntimeError(
            f"grasp verification requires an RGB-D camera whose name contains "
            f"{wrist_camera_keyword!r}"
        )
    candidates = [wrist]
    overhead = next((c for c in cameras if c.get("name") == overhead_camera_name), None)
    if overhead is not None:
        candidates.append(overhead)
    queries = [q for q in (object_description, marker_description) if q]
    hand = np.array([lifted["position"][key] for key in ("x", "y", "z")], dtype=np.float64)
    size_gated = max_span_m is not None or max_width_m is not None

    def points_of(camera: dict[str, Any], mask: Any) -> np.ndarray:
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)

    def oversize(points: np.ndarray) -> str:
        if not size_gated:
            return ""
        span, width = _size(points)
        if (max_span_m is not None and span > float(max_span_m)) or (
            max_width_m is not None and width > float(max_width_m)
        ):
            return f"span {span * 1000.0:.1f} mm, width {width * 1000.0:.1f} mm"
        return ""

    if nearest_to_hand:
        return _nearest_to_hand(
            ctx, candidates, queries, hand, points_of, oversize, object_description,
            score_min, min_points, min_above_table_m, max_hand_distance_m, max_masks_per_query,
        )

    camera: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    for candidate in candidates:
        for query in queries:
            detected = ctx.tool(
                "sam3.segment_text", image=candidate["rgb"], query=query,
                max_results=int(max_masks_per_query),
            )
            if (
                detected.get("masks")
                and detected.get("scores")
                and float(detected["scores"][0]) >= float(score_min)
            ):
                camera, result = candidate, detected
                break
        if camera is not None:
            break
    if camera is None or result is None:
        return _not_held(f"no camera sees {object_description!r} after lifting")

    camera_name = str(camera.get("name", "unknown"))
    points = points_of(camera, result["masks"][0])
    if len(points) < int(min_points):
        return _not_held(
            f"{camera_name} mask of {object_description!r} has too few valid depth points "
            f"({len(points)} < {int(min_points)})",
            camera=camera_name,
            point_count=len(points),
        )
    too_big = oversize(points)
    if too_big:
        return _not_held(
            f"{camera_name} mask of {object_description!r} is larger than one object ({too_big})",
            camera=camera_name,
            point_count=len(points),
            center=np.median(points, axis=0),
        )

    center = np.median(points, axis=0)
    workspace = ctx.tool("robot.describe_workspace")
    above_table = float(center[2] - float(workspace["surface_z"]))
    hand_distance = float(np.linalg.norm(center - hand))
    if above_table < float(min_above_table_m) or hand_distance > float(max_hand_distance_m):
        return _not_held(
            f"{camera_name} reobservation shows {object_description!r} was not lifted with "
            f"the hand (above table {above_table * 1000.0:.1f} mm, hand distance "
            f"{hand_distance * 1000.0:.1f} mm)",
            camera=camera_name,
            point_count=len(points),
            center=center,
        )
    return {
        "route": "verified",
        "verified": True,
        "observed_center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "point_count": int(len(points)),
        "camera": camera_name,
        "reason": "",
    }


def _nearest_to_hand(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    queries: list[str],
    hand: np.ndarray,
    points_of: Any,
    oversize: Any,
    object_description: str,
    score_min: float,
    min_points: int,
    min_above_table_m: float,
    max_hand_distance_m: float,
    max_masks_per_query: int,
) -> Output:
    """Every admitted mask is a candidate; the one nearest the hand is judged."""
    surface_z: float | None = None  # asked for only once a candidate has a cloud
    nearest: tuple[float, float, np.ndarray, int, str] | None = None
    sparse: tuple[int, str] | None = None
    too_big: tuple[str, int, str] | None = None
    for camera in cameras:
        camera_name = str(camera.get("name", "unknown"))
        for query in queries:
            detected = ctx.tool(
                "sam3.segment_text", image=camera["rgb"], query=query,
                max_results=int(max_masks_per_query),
            )
            for mask, score in zip(detected.get("masks") or [], detected.get("scores") or [], strict=False):
                if float(score) < float(score_min):
                    continue
                points = points_of(camera, mask)
                if len(points) < int(min_points):
                    if sparse is None or len(points) > sparse[0]:
                        sparse = (len(points), camera_name)
                    continue
                size = oversize(points)
                if size:
                    too_big = too_big or (size, len(points), camera_name)
                    continue
                if surface_z is None:
                    surface_z = float(ctx.tool("robot.describe_workspace")["surface_z"])
                center = np.median(points, axis=0)
                distance = float(np.linalg.norm(center - hand))
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, float(center[2] - surface_z), center, len(points), camera_name)
            if nearest is not None and nearest[0] <= float(max_hand_distance_m) and nearest[1] >= float(
                min_above_table_m
            ):
                _, _, center, count, name = nearest
                return {
                    "route": "verified",
                    "verified": True,
                    "observed_center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                    "point_count": int(count),
                    "camera": name,
                    "reason": "",
                }
    if nearest is not None:
        distance, above_table, center, count, name = nearest
        return _not_held(
            f"{name} reobservation shows {object_description!r} was not lifted with the hand "
            f"(nearest candidate: above table {above_table * 1000.0:.1f} mm, hand distance "
            f"{distance * 1000.0:.1f} mm)",
            camera=name,
            point_count=count,
            center=center,
        )
    if sparse is not None:
        count, name = sparse
        return _not_held(
            f"{name} mask of {object_description!r} has too few valid depth points "
            f"({count} < {int(min_points)})",
            camera=name,
            point_count=count,
        )
    if too_big is not None:
        size, count, name = too_big
        return _not_held(
            f"{name} mask of {object_description!r} is larger than one object ({size})",
            camera=name,
            point_count=count,
        )
    return _not_held(f"no camera sees {object_description!r} after lifting")
