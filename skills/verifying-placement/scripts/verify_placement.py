"""Camera-only evidence that a released object passed below an aperture.

After insertion and release, the object is looked for in the named camera.
An opaque container hides a contained object, so not seeing it at all counts
as success; seeing it below the rim within the aperture's footprint counts
too; seeing it above the rim or away from the aperture routes ``not_placed``.

Multi-view mode (``views``, opt-in; default ``None`` is the path above,
unchanged). Promoted from the sharps_disposal benchmark's perception graph, whose loop
posts one item after another through the same lid hole. There, one camera
and "absence is success" were both wrong in measurable ways:

* An item that lands standing on the ones already inside is only visible from
  the side, and a second view is what separates "it went in" from "the camera
  cannot see it". So ``min_views`` (default 2) calibrated views are required,
  and with none visible the verdict is ``clear`` -- no obstruction -- never
  ``verified``: absence is not proof of delivery.
* With six or more items in the scene, the one that matters was not among the
  segmenter's top three masks. Each view is cropped to a world box around the
  aperture and every returned mask is judged (``max_results=0``).
* Masks lying flat on the lid (nearly all points within ``flat_lid_tolerance_m``
  of the rim) are lid texture or a reflection, not an object protruding.
* The gripper, still near the hole after release, segments as the object.
  Masks centred within ``hand_exclusion_m`` of the live TCP
  (``robot.get_ee_pose``) are ignored and counted in ``ignored_robot_masks``.
* "Above the rim" means at least ``min_points`` points and a fraction
  ``above_rim_min_fraction`` of them more than ``above_rim_tolerance_m`` above
  ``rim_z``. See that parameter for the measured 10 mm.
"""

import os
import sys
from typing import Any, NotRequired, TypedDict

import numpy as np
from gap import NodeContext

# Helpers for the multi-view mode live beside this script (the tipcommon pattern).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placementcommon import crop_for_world_box, detect, summary, xyz  # noqa: E402

ABOVE_RIM_TOLERANCE_M = 0.010
"""How far above the rim an item's points must reach to count as left above the
aperture (multi-view mode). In the sharps_disposal benchmark sweep s7, trials t04 and t07
aborted after an unjam push over syringes standing on the pile with their tops
only +3 and +4 mm above the rim, 13-17 mm off the hole's centre. Their centres
were 70 mm inside the box, where the containment goal counts them. The previous
3 mm margin made a full box read as a failed delivery; with 10 mm the next sweep
(s8) solved 16/30 trials against 13/30."""


class Output(TypedDict):
    route: str
    verified: bool
    evidence: str
    # Multi-view mode only (``views`` given); additive, absent on the default path.
    status: NotRequired[str]
    observations: NotRequired[list]
    ignored_robot_masks: NotRequired[int]


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def _verdict(status: str, evidence: str, observations: list, ignored: int) -> Output:
    return {
        "route": status,
        "verified": status == "verified",
        "evidence": evidence,
        "status": status,
        "observations": observations,
        "ignored_robot_masks": ignored,
    }


def _multi_view(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    aperture_center: dict[str, float],
    rim_z: float,
    *,
    views: list[str],
    min_views: int,
    xy_tolerance_m: float,
    min_points: int,
    strong_score_min: float,
    detection_score_floor: float,
    crop_half_width_m: float,
    crop_below_m: float,
    crop_above_m: float,
    crop_padding_px: int,
    flat_lid_tolerance_m: float,
    flat_lid_fraction: float,
    hand_exclusion_m: float,
    above_rim_tolerance_m: float,
    above_rim_min_fraction: float,
) -> Output:
    rim_z = float(rim_z)
    hand = None
    if hand_exclusion_m and float(hand_exclusion_m) > 0.0:
        hand = xyz(ctx.tool("robot.get_ee_pose")["pose"]["position"])
    center = xyz(aperture_center)
    lower = center + [-float(crop_half_width_m), -float(crop_half_width_m), -float(crop_below_m)]
    upper = center + [float(crop_half_width_m), float(crop_half_width_m), float(crop_above_m)]
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    wanted = set(views)
    strong: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    ignored = 0
    seen = 0
    for camera in cameras:
        if camera.get("name") not in wanted:
            continue
        try:
            crop = crop_for_world_box(camera, lower, upper, padding=int(crop_padding_px))
        except ValueError:
            # The aperture box is behind this camera or off its image: not a
            # calibrated view of the hole. It does not count toward min_views.
            continue
        seen += 1
        for d in detect(ctx, camera, object_description, crop=crop, floor=float(detection_score_floor)):
            if np.linalg.norm(d["center"][:2] - center[:2]) > float(xy_tolerance_m):
                continue
            # A segmentation on the flat lid is not an object protruding above it.
            if np.mean(np.abs(d["points"][:, 2] - rim_z) < float(flat_lid_tolerance_m)) > float(flat_lid_fraction):
                continue
            # The live hand envelope, not a stale planner sphere model.
            if hand is not None and np.linalg.norm(d["center"] - hand) < float(hand_exclusion_m):
                ignored += 1
                continue
            (strong if d["score"] >= float(strong_score_min) else weak).append(d)
    if seen < int(min_views):
        return _verdict(
            "uncertain",
            f"placement uncertain: {int(min_views)} calibrated inspection views are required, {seen} available",
            [],
            ignored,
        )

    def above_rim(d: dict[str, Any]) -> bool:
        above = d["points"][:, 2] > rim_z + float(above_rim_tolerance_m)
        return np.count_nonzero(above) >= int(min_points) and np.mean(above) >= float(above_rim_min_fraction)

    outside = [d for d in strong if above_rim(d)]
    if outside:
        return _verdict(
            "blocked",
            f"{object_description} remains above the aperture after release",
            [summary(d) for d in outside],
            ignored,
        )
    if any(above_rim(d) for d in weak):
        return _verdict(
            "uncertain",
            "weak above-rim evidence requires another observation",
            [summary(d) for d in weak],
            ignored,
        )
    if strong:
        return _verdict(
            "verified",
            f"visible {object_description} centres are below the rim",
            [summary(d) for d in strong],
            ignored,
        )
    return _verdict(
        "clear",
        f"no visible obstruction in {seen} views; delivery is not established by absence",
        [],
        ignored,
    )


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    aperture_center: dict[str, float],
    rim_z: float,
    score_min: float = 0.03,
    xy_tolerance_m: float = 0.10,
    depth_margin_m: float = 0.005,
    min_points: int = 8,
    camera_name: str = "overhead",
    views: list[str] | None = None,
    min_views: int = 2,
    strong_score_min: float = 0.03,
    detection_score_floor: float = 0.01,
    crop_half_width_m: float = 0.11,
    crop_below_m: float = 0.12,
    crop_above_m: float = 0.15,
    crop_padding_px: int = 16,
    flat_lid_tolerance_m: float = 0.001,
    flat_lid_fraction: float = 0.90,
    hand_exclusion_m: float = 0.14,
    above_rim_tolerance_m: float = ABOVE_RIM_TOLERANCE_M,
    above_rim_min_fraction: float = 0.20,
) -> Output:
    """Single-camera check by default; ``views=[...]`` selects the multi-view mode.

    Every parameter after ``camera_name`` is read only in multi-view mode.
    There ``route`` equals ``status``: ``verified``, ``clear``, ``blocked`` or
    ``uncertain``, and ``not_placed`` is never returned. ``xy_tolerance_m`` and
    ``min_points`` are shared with the default path. In multi-view mode they
    bound which masks count as at the aperture, and how many points must stand
    above the rim.
    """
    if views is not None:
        return _multi_view(
            ctx,
            observation,
            object_description,
            aperture_center,
            rim_z,
            views=list(views),
            min_views=min_views,
            xy_tolerance_m=xy_tolerance_m,
            min_points=min_points,
            strong_score_min=strong_score_min,
            detection_score_floor=detection_score_floor,
            crop_half_width_m=crop_half_width_m,
            crop_below_m=crop_below_m,
            crop_above_m=crop_above_m,
            crop_padding_px=crop_padding_px,
            flat_lid_tolerance_m=flat_lid_tolerance_m,
            flat_lid_fraction=flat_lid_fraction,
            hand_exclusion_m=hand_exclusion_m,
            above_rim_tolerance_m=above_rim_tolerance_m,
            above_rim_min_fraction=above_rim_min_fraction,
        )
    camera = _camera(observation, camera_name)
    result = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=object_description, max_results=3
    )
    if not result.get("masks") or float((result.get("scores") or [0.0])[0]) < float(score_min):
        # An opaque container hides a contained object. This evidence is
        # meaningful only after insertion and release completed.
        return {
            "route": "verified",
            "verified": True,
            "evidence": f"{object_description} disappeared after aperture release",
        }
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(result["masks"][0], dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < int(min_points):
        return {
            "route": "not_placed",
            "verified": False,
            "evidence": (
                f"visible {object_description} has too few valid depth points "
                f"({len(points)} < {int(min_points)}) to confirm placement"
            ),
        }
    center = np.median(points, axis=0)
    aperture_xy = np.array([aperture_center["x"], aperture_center["y"]], dtype=np.float64)
    offset = float(np.linalg.norm(center[:2] - aperture_xy))
    if float(center[2]) >= float(rim_z) - float(depth_margin_m) or offset > float(xy_tolerance_m):
        return {
            "route": "not_placed",
            "verified": False,
            "evidence": (
                f"{object_description} remains visible outside or above the aperture "
                f"(centre z {center[2]:.3f} vs rim {float(rim_z):.3f}, offset {offset * 1000.0:.0f} mm)"
            ),
        }
    return {
        "route": "verified",
        "verified": True,
        "evidence": f"visible {object_description} centre is below the aperture",
    }
