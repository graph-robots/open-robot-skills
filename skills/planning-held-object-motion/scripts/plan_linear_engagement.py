"""Plan orientation-locked approach and linear feature engagement.

Without any of the parameters below this is the original body: one planned
joint leg to a pre-contact pose, one orientation-locked linear leg past the
opening, and the wrist re-observation of the held tip replacing the carried
feature whenever it is visible. Every addition is opt-in; the only change a
caller who asks for nothing sees is five additive, call-free plan keys
(``aim_source``, ``leading_end_in_hand``, ``aperture``, ``fixture_axis``,
``rim_z``) that a downstream seat check reads.

The additions come from the sharps disposal graph (``gap_perception_v2``'s
``plan_visual_insert.py``, sweep s8: 16/30 trays, 152/176 syringes):

- ``stroke_depths_m`` replaces the two legs with staged strokes to signed
  depths along the fixture axis; every stroke after the first allows start and
  goal contact with ``stroke_contact_margin_m``. One 40 mm leg left an uncarved
  gap at the rim and cuRobo refused an otherwise physical insertion at 56-75%
  of the stroke; 10 mm phases carve bounded volumes at each boundary. The
  first stroke defaults to ``planned_linear``: as a free joint goal the 60 mm
  descent re-solved the arm into another elbow branch (joints 1, 3, 5 swinging
  0.8-1.2 rad) and the held syringe pivoted 40 deg in the fingers.
- ``mirror_leading_end`` aims whichever end of the held object leads along the
  fixture axis. A wrist-only stand-up may leave the carried feature trailing;
  with the grasp at the object's middle the other end is the feature offset
  mirrored through the TCP. The wrist marker re-observation then applies only
  when the carried feature is the one leading, because the marker marks it.
- ``leading_end_band_m`` / ``leading_end_max_lateral_m`` bound where an aimed
  end may sit in the hand frame: on the fixture axis, ahead of the TCP within
  the band. A perceived end outside it is ignored. On ``tray_clutter_t03`` a
  wrist "red cap" detection took a cap in the scene 34 mm off the barrel axis
  and moved the approach 140 mm.
- ``tip_source="carried_unless_implausible"`` keeps the carried feature
  whenever it is inside the band and lets perception replace it only when it
  is not. Measured against the simulator's mesh end: the registered offset
  57.0 / 68.4 mm against true 58.4 / 64.0 mm, the side view 24.8 / 49.1 mm
  (the fingers hide the barrel's upper half). Its short reads cost ``t06`` its
  approach (a refused 92 mm descent) and ``t07`` its seat (-23 mm). The
  default, ``"perception"``, keeps perception winning.
- ``side_camera_names`` reads the leading end from a fixed side camera first:
  the held object hangs along the fixture axis at the transit pose and a side
  view sees its length, so the points farthest along the axis are the end
  that meets the opening. Helpers live in ``engagement_perception.py``.
"""

import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    placement_plan: dict[str, Any]


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def _visible_tip(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    direction_marker_description: str,
) -> np.ndarray | None:
    """Return the best wrist-observed physical endpoint of the held object.

    The endpoint lies along the object's principal axis, directed toward its
    visible direction marker (a cap, a coloured band, a head).
    """
    best = None
    for camera in observation.get("cameras", []):
        if "eye_in_hand" not in camera.get("name", ""):
            continue
        body = ctx.tool(
            "sam3.segment_text", image=camera["rgb"], query=object_description, max_results=3
        )
        marker = ctx.tool(
            "sam3.segment_text",
            image=camera["rgb"],
            query=direction_marker_description,
            max_results=3,
        )
        score = float((body.get("scores") or [0.0])[0])
        if not body.get("masks") or not marker.get("masks") or score < 0.04:
            continue
        marker_score = float((marker.get("scores") or [0.0])[0])
        if marker_score < 0.04 or (best is not None and score + marker_score <= best[0]):
            continue
        best = (score + marker_score, camera, body["masks"][0], marker["masks"][0])
    if best is None:
        return None
    _, camera, body_mask, marker_mask = best

    def points(mask):
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)

    body_points, marker_points = points(body_mask), points(marker_mask)
    if len(body_points) < 20 or len(marker_points) < 6:
        return None
    center = np.median(body_points, axis=0)
    _, _, basis = np.linalg.svd(body_points - center, full_matrices=False)
    axis = basis[0]
    if float(axis @ (np.median(marker_points, axis=0) - center)) < 0.0:
        axis = -axis
    along = (body_points - center) @ axis
    return np.median(body_points[along >= np.percentile(along, 98)], axis=0)


#: World-axis box around the hand, (below, above) per axis [m], cropped from the
#: side camera before segmenting: the held object hangs below the hand.
_SIDE_VIEW_BOX_M = ((0.16, 0.16, 0.18), (0.16, 0.16, 0.16))
#: A side detection must run within this of the fixture axis (|cos|): the
#: horizontal gripper fingers are not the held shaft.
_SIDE_VIEW_MIN_ALIGNMENT = 0.85


def _visible_leading_end(
    ctx: NodeContext,
    observation: dict[str, Any],
    camera_names: list[str],
    object_description: str,
    hand: np.ndarray,
    axis: np.ndarray,
    held_max_distance_m: float,
) -> np.ndarray | None:
    """The held object's end leading along ``axis``, seen from a fixed side camera."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from engagement_perception import crop_for_world_box, detect, rod_like  # noqa: PLC0415

    best = None
    for camera in observation.get("cameras", []):
        if camera.get("name") not in camera_names or "depth" not in camera:
            continue
        lower, upper = (np.asarray(v, dtype=np.float64) for v in _SIDE_VIEW_BOX_M)
        try:
            crop = crop_for_world_box(camera, hand - lower, hand + upper)
        except ValueError:
            continue  # the hand is not in this camera's view
        for detected in detect(ctx, camera, crop=crop, prompt=object_description, floor=0.04):
            if not rod_like(detected) or abs(float(detected["axis"] @ axis)) < _SIDE_VIEW_MIN_ALIGNMENT:
                continue
            points = detected["points"]
            points = points[np.linalg.norm(points - hand, axis=1) <= held_max_distance_m]
            if len(points) < 20:
                continue
            distance = float(np.linalg.norm(np.median(points, axis=0) - hand))
            along = points @ axis
            end = np.median(points[along >= np.percentile(along, 95)], axis=0)
            if float((end - hand) @ axis) < 0.015:
                continue
            if best is None or distance < best[0]:
                best = (distance, end)
    return None if best is None else best[1]


def run(
    ctx: NodeContext,
    held_feature_in_tcp: dict[str, Any],
    fixture_feature: dict[str, Any],
    relation: str,
    world_config: dict[str, Any],
    attached_object: dict[str, Any],
    precontact_clearance_m: float = 0.02,
    engagement_depth_m: float = 0.035,
    observation: dict[str, Any] | None = None,
    object_description: str = "",
    direction_marker_description: str = "",
    stroke_depths_m: list[float] | None = None,
    first_stroke_mode: str = "planned_linear",
    stroke_contact_margin_m: float = 0.008,
    mirror_leading_end: bool = False,
    tip_source: str = "perception",
    leading_end_band_m: list[float] | None = None,
    leading_end_max_lateral_m: float = 0.02,
    side_camera_names: list[str] | None = None,
    held_max_distance_m: float = 0.15,
) -> Output:
    if relation not in {
        "shaft_into_aperture",
        "tip_through_aperture",
        "insert_through",
        "loop_over_shaft",
        "feature_to_fixture",
    }:
        raise ValueError(f"unsupported feature relation {relation!r}")
    if tip_source not in {"perception", "carried_unless_implausible"}:
        raise ValueError(f"unsupported tip_source {tip_source!r}")
    if tip_source == "carried_unless_implausible" and leading_end_band_m is None:
        raise ValueError("tip_source 'carried_unless_implausible' needs leading_end_band_m")
    if first_stroke_mode not in {"planned_joint", "planned_linear"}:
        raise ValueError(f"unsupported first_stroke_mode {first_stroke_mode!r}")
    axis = np.array([fixture_feature["axis"][k] for k in ("x", "y", "z")], dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    center = np.array(
        [fixture_feature["pose"]["position"][k] for k in ("x", "y", "z")], dtype=float
    )
    ee = ctx.tool("robot.get_ee_pose")["pose"]
    ee_matrix, feature_matrix = _matrix(ee), _matrix(held_feature_in_tcp)
    rotation, feature_tcp = ee_matrix[:3, :3], feature_matrix[:3, 3]
    hand_origin = ee_matrix[:3, 3]
    tip_leads = True
    if mirror_leading_end:
        tip_leads = float((rotation @ feature_tcp) @ axis) >= float((rotation @ -feature_tcp) @ axis)
        feature_tcp = feature_tcp if tip_leads else -feature_tcp
    aim_source = "carried" if tip_leads else "carried_mirrored"

    def plausible(in_hand: np.ndarray) -> bool:
        if leading_end_band_m is None:
            return True
        along_axis = rotation.T @ axis
        along = float(in_hand @ along_axis)
        lateral = float(np.linalg.norm(in_hand - along * along_axis))
        low, high = (float(v) for v in leading_end_band_m)
        return low <= along <= high and lateral <= float(leading_end_max_lateral_m)

    accepted = tip_source == "carried_unless_implausible" and plausible(feature_tcp)
    if not accepted and observation is not None and side_camera_names and object_description:
        leading = _visible_leading_end(ctx, observation, list(side_camera_names), object_description,
                                       hand_origin, axis, float(held_max_distance_m))
        if leading is not None and plausible(rotation.T @ (leading - hand_origin)):
            feature_tcp, aim_source, accepted = rotation.T @ (leading - hand_origin), "side_view", True
    if (not accepted and tip_leads and observation is not None and object_description
            and direction_marker_description):
        # Re-observe the held feature right before engagement: the object may
        # have settled in the hand since registration, and the wrist view of
        # the physical endpoint is more precise than the carried estimate.
        visible_tip = _visible_tip(
            ctx, observation, object_description, direction_marker_description
        )
        if visible_tip is not None and plausible(rotation.T @ (visible_tip - hand_origin)):
            feature_tcp, aim_source = rotation.T @ (visible_tip - hand_origin), "wrist_tip"

    def hand_pose(feature_target: np.ndarray) -> dict[str, Any]:
        position = feature_target - rotation @ feature_tcp
        return {
            "position": dict(zip(("x", "y", "z"), map(float, position), strict=True)),
            "rotation": dict(ee["rotation"]),
        }

    if stroke_depths_m:
        waypoints = []
        for index, depth in enumerate(stroke_depths_m):
            mode = first_stroke_mode if index == 0 else "planned_linear"
            waypoint = {"pose": hand_pose(center + float(depth) * axis), "mode": mode,
                        "cartesian": mode == "planned_linear"}
            if index > 0:
                waypoint.update(allow_start_contact=True, allow_goal_contact=True,
                                contact_margin=float(stroke_contact_margin_m))
            waypoints.append(waypoint)
    else:
        precontact = center - max(0.005, float(precontact_clearance_m)) * axis
        engaged = center + max(0.005, float(engagement_depth_m)) * axis
        waypoints = [
            {"pose": hand_pose(precontact), "mode": "planned_joint", "cartesian": False},
            {
                "pose": hand_pose(engaged),
                "mode": "planned_linear",
                "cartesian": True,
                "allow_goal_contact": True,
            },
        ]
    return {
        "placement_plan": {
            "waypoints": waypoints,
            "world_config": world_config,
            "attached_object": attached_object,
            # Additive, call-free annotations for a downstream seat check.
            "aim_source": aim_source,
            "leading_end_in_hand": dict(zip(("x", "y", "z"), map(float, feature_tcp), strict=True)),
            "aperture": dict(zip(("x", "y", "z"), map(float, center), strict=True)),
            "fixture_axis": dict(zip(("x", "y", "z"), map(float, axis), strict=True)),
            "rim_z": float(center[2]),
        }
    }
