"""Count the described items still left in the workspace; a loop terminator.

Promoted from the sharps_disposal benchmark's perception graph, where it decides after each
posted item whether another pass is needed. It is written to fail toward
``uncertain`` rather than toward ``none``, because the two errors cost
differently: a false ``more`` sends the loop to a perception pass that finds
nothing and aborts, while a false ``none`` ends the task with items left.

* **Positive evidence is enough; negative evidence needs views.** The first
  camera (in ``camera_names`` order) with a detection at or above ``score_min``
  returns ``more``. ``none`` needs ``min_negative_views`` (default 2) cameras
  that saw nothing at all, not even weakly. In the graph a single overhead
  view missed syringes lying in the shadow of the box and under the parked
  arm; the side view saw them.
* **Weak evidence is re-examined, then kept.** When a view has only detections
  below ``score_min``, the segmenter runs again on a crop padded by
  ``recrop_padding_px`` around them. At full resolution a lone syringe in a
  cluttered tray scored under 0.08 and scored well above it in the crop. A
  detection that stays weak after the re-crop makes the result ``uncertain``
  (``remaining = -1``), never ``none``.
* **Every mask is judged** (``max_results=0``): with six or more items, a real
  one was not in the segmenter's top results.
* **Eligibility** filters what is not a remaining item. Masks centred above
  ``max_z`` (the graph passes the container's rim height, so posted items
  that stick out do not count). Masks within ``exclude_radius_m`` in XY of
  ``exclude_center`` (the graph passes the aperture: 0.12 m). Masks within
  ``hand_exclusion_m`` of the live TCP (``robot.get_ee_pose``), because the
  gripper segments as the item. Optionally, masks whose principal extents
  fail the shape limits. The graph's syringe values were length 0.025-0.22 m,
  width at most 0.045 m and aspect at least 2; they are parameters here and
  default to no shape gate.
"""

import os
import sys
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

# Helpers live beside this script (the tipcommon pattern).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from countingcommon import detect, distinct, expanded_bounds, rod_like, summary, xyz  # noqa: E402

DEFAULT_CAMERAS = ("overhead", "agentview")


class Output(TypedDict):
    route: str
    remaining: int
    status: str
    centers: list
    evidence: list


def _result(status: str, remaining: int, centers: list, evidence: list) -> Output:
    return {
        "route": status,
        "remaining": remaining,
        "status": status,
        "centers": centers,
        "evidence": evidence,
    }


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    exclude_center: dict[str, float] | None = None,
    exclude_radius_m: float = 0.12,
    max_z: float | None = None,
    camera_names: list[str] | None = None,
    score_min: float = 0.08,
    min_negative_views: int = 2,
    hand_exclusion_m: float = 0.14,
    min_length_m: float | None = None,
    max_length_m: float | None = None,
    max_width_m: float | None = None,
    min_aspect: float | None = None,
    detection_score_floor: float = 0.01,
    recrop_weak: bool = True,
    recrop_padding_px: int = 50,
    distinct_m: float = 0.02,
) -> Output:
    """Route ``more`` (``remaining`` >= 1), ``none`` (0) or ``uncertain`` (-1).

    ``camera_names`` defaults to ``["overhead", "agentview"]`` and fixes the
    order the cameras are consulted in. Cameras absent from the observation
    do not count toward ``min_negative_views``.
    """
    names = list(camera_names) if camera_names is not None else list(DEFAULT_CAMERAS)
    hand = None
    if hand_exclusion_m and float(hand_exclusion_m) > 0.0:
        hand = xyz(ctx.tool("robot.get_ee_pose")["pose"]["position"])
    excluded = xyz(exclude_center) if exclude_center is not None else None

    def eligible(d: dict[str, Any]) -> bool:
        if max_z is not None and not d["center"][2] < float(max_z):
            return False
        if excluded is not None and not np.linalg.norm(d["center"][:2] - excluded[:2]) > float(exclude_radius_m):
            return False
        if hand is not None and not np.linalg.norm(d["center"] - hand) > float(hand_exclusion_m):
            return False
        return rod_like(
            d,
            min_length_m=min_length_m,
            max_length_m=max_length_m,
            max_width_m=max_width_m,
            min_aspect=min_aspect,
        )

    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    ordered = sorted(
        (c for c in cameras if c.get("name") in names),
        key=lambda c: names.index(c["name"]),
    )
    floor = float(detection_score_floor)
    strong: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    views = 0
    for camera in ordered:
        views += 1
        items = [d for d in detect(ctx, camera, object_description, floor=floor) if eligible(d)]
        accepted = [d for d in items if d["score"] >= float(score_min)]
        if recrop_weak and not accepted and items:
            crop = expanded_bounds(items, camera, padding=int(recrop_padding_px))
            recovered = [
                d for d in detect(ctx, camera, object_description, crop=crop, floor=floor) if eligible(d)
            ]
            items += recovered
            accepted = [d for d in items if d["score"] >= float(score_min)]
        strong += accepted
        weak += [d for d in items if d["score"] < float(score_min)]
        if strong:
            found = distinct(strong, float(distinct_m))
            return _result(
                "more",
                len(found),
                [summary(d)["center"] for d in found],
                [summary(d) for d in found],
            )
    if weak or views < int(min_negative_views):
        evidence = [summary(d) for d in distinct(weak, float(distinct_m))]
        if views < int(min_negative_views):
            evidence.append({"negative_views": views, "required_views": int(min_negative_views)})
        return _result("uncertain", -1, [], evidence)
    return _result("none", 0, [], [{"negative_views": views}])
