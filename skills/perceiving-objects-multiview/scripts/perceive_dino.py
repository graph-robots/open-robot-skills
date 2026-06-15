"""DINO-enhanced multiview perception: Molmo + GDINO -> box selection -> SAM3 -> 3D fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import BoundingBox2D, CameraFrame, Mask, PointCloud

logger = logging.getLogger(__name__)


@dataclass
class CamResult:
    """Result from a single camera perception."""

    cloud: Any
    mask: Any
    score: float


def _empty_cloud() -> PointCloud:
    return {"points": np.zeros((0, 3), dtype=np.float32)}


def _empty_mask() -> Mask:
    return np.zeros((0, 0), dtype=np.uint8)


def _select_box_by_point(
    detections: list,
    px: float,
    py: float,
    margin: int = 20,
) -> BoundingBox2D | None:
    """Pick the detection box that best matches the Molmo point.

    Strategy:
      1. Find boxes that CONTAIN the point (with margin)
      2. Among those, pick the smallest (tightest fit)
      3. If none contain it, pick the box whose center is closest
    """
    if not detections:
        return None

    containing = []
    for det in detections:
        b = det["box"]
        if (b["x1"] - margin <= px <= b["x2"] + margin) and (b["y1"] - margin <= py <= b["y2"] + margin):
            area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            containing.append((det, area))

    if containing:
        return min(containing, key=lambda t: t[1])[0]

    best_det = None
    best_dist = float("inf")
    for det in detections:
        b = det["box"]
        cx = (b["x1"] + b["x2"]) / 2.0
        cy = (b["y1"] + b["y2"]) / 2.0
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_det = det
    return best_det


def _perceive_single_camera_dino(
    ctx: Any,
    cam: CameraFrame,
    object_name: str,
    text_prompts: list[str],
    min_score: float,
    min_points: int,
    box_threshold: float,
    text_threshold: float,
    box_margin: int,
    use_point_and_box: bool,
) -> CamResult | None:
    """Run DINO-enhanced perception on a single camera view.

    Fixed cameras: Molmo point + GDINO boxes -> select box -> SAM3 box-prompted.
    Wrist cameras: SAM3 text prompt directly.
    """
    is_wrist = cam["name"] and "eye_in_hand" in cam["name"]

    seg_mask = None
    seg_score = 0.0

    if not is_wrist:
        # Molmo is optional — when the endpoint is offline (no
        # GAP_MOLMO_BASE_URL) or transiently fails, fall through to
        # GDINO-box-prompted SAM3 (using the top-scoring detection) and then
        # SAM3 text, mirroring perceive_point.py.
        try:
            point_resp = ctx.tool(
                "molmo.point_prompt",
                image=cam["rgb"], query=object_name,
            )
        except Exception as exc:
            logger.debug(
                "molmo.point_prompt unavailable for '%s' (%s: %s); falling "
                "back to GDINO box + SAM3 text",
                object_name, type(exc).__name__, exc,
            )
            point_resp = None

        gdino_detections = []
        try:
            gdino_resp = ctx.tool(
                "grounding-dino.detect",
                image=cam["rgb"], query=object_name,
                box_threshold=box_threshold, text_threshold=text_threshold,
            )
            gdino_detections = list(gdino_resp["detections"])
        except Exception as e:
            logger.warning("GDINO detect failed, falling back to point-only: %s", e)

        if point_resp is not None and point_resp["found"] and gdino_detections:
            selected = _select_box_by_point(
                gdino_detections,
                point_resp["pixel_x"], point_resp["pixel_y"],
                margin=box_margin,
            )

            if selected is not None:
                kwargs: dict[str, Any] = {
                    "image": cam["rgb"],
                    "box": selected["box"],
                }
                if use_point_and_box:
                    kwargs["pixel_x"] = point_resp["pixel_x"]
                    kwargs["pixel_y"] = point_resp["pixel_y"]
                    kwargs["use_point"] = True

                seg_resp = ctx.tool("sam3.segment_box", **kwargs)
                if seg_resp["masks"] and seg_resp["scores"]:
                    seg_mask = seg_resp["masks"][0]
                    seg_score = seg_resp["scores"][0]
        elif point_resp is None and gdino_detections:
            # No Molmo point — pick the top-scoring GDINO box and ask SAM3 to
            # segment within it.  This is the dino-only path.
            selected = max(
                gdino_detections,
                key=lambda d: d.get("score", 0.0),
            )
            seg_resp = ctx.tool(
                "sam3.segment_box",
                image=cam["rgb"], box=selected["box"],
            )
            if seg_resp["masks"] and seg_resp["scores"]:
                seg_mask = seg_resp["masks"][0]
                seg_score = seg_resp["scores"][0]

        if (
            (seg_mask is None or seg_score < min_score)
            and point_resp is not None
            and point_resp["found"]
        ):
            seg_resp = ctx.tool(
                "sam3.segment_point",
                image=cam["rgb"],
                pixel_x=point_resp["pixel_x"],
                pixel_y=point_resp["pixel_y"],
            )
            if seg_resp["masks"] and seg_resp["scores"]:
                if seg_resp["scores"][0] > seg_score:
                    seg_mask = seg_resp["masks"][0]
                    seg_score = seg_resp["scores"][0]

    if seg_mask is None or seg_score < min_score:
        for prompt in text_prompts:
            seg_resp = ctx.tool(
                "sam3.segment_text",
                image=cam["rgb"], query=prompt,
            )
            if seg_resp["masks"] and seg_resp["scores"] and seg_resp["scores"][0] >= min_score:
                seg_mask = seg_resp["masks"][0]
                seg_score = seg_resp["scores"][0]
                break

    if seg_mask is None or seg_score < min_score:
        logger.debug("No acceptable mask for '%s' in cam '%s' (dino)",
                     object_name, cam["name"])
        return None

    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=seg_mask, depth=cam["depth"],
        intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
    )["points"]

    num_points = len(cloud["points"])
    if num_points < min_points:
        logger.debug(
            "Too few points (%d < %d) for '%s' in cam '%s' (dino)",
            num_points, min_points, object_name, cam["name"],
        )
        return None

    return CamResult(cloud=cloud, mask=seg_mask, score=seg_score)


def _merge_multiview(results: list[CamResult]) -> PointCloud:
    """Merge point clouds from multiple views using KD-tree intersection check."""
    from scipy.spatial import cKDTree

    _INTERSECTION_DIST = 0.01
    _MIN_INTERSECTION = 1

    if len(results) < 2:
        return results[0].cloud

    clouds_np = []
    for r in results:
        pts = np.asarray(r.cloud["points"], dtype=np.float64)
        if len(pts) > 0:
            clouds_np.append(pts.reshape(-1, 3))
        else:
            clouds_np.append(np.zeros((0, 3)))

    pts_a, pts_b = clouds_np[0], clouds_np[1]

    do_merge = False
    if len(pts_a) > 0 and len(pts_b) > 0:
        tree = cKDTree(pts_a)
        dists, _ = tree.query(pts_b)
        intersection = int(np.sum(dists < _INTERSECTION_DIST))
        logger.info(
            "Multiview intersection: %d points within %.0fmm "
            "(views have %d and %d points)",
            intersection, _INTERSECTION_DIST * 1000,
            len(pts_a), len(pts_b),
        )
        do_merge = intersection >= _MIN_INTERSECTION

    if do_merge:
        all_points = np.concatenate(
            [np.asarray(r.cloud["points"], dtype=np.float32).reshape(-1, 3)
             for r in results],
            axis=0,
        )
        merged: PointCloud = {"points": all_points}
        if all(r.cloud.get("colors") is not None for r in results):
            merged["colors"] = np.concatenate(
                [np.asarray(r.cloud["colors"], dtype=np.float32).reshape(-1, 3)
                 for r in results],
                axis=0,
            )
        return merged
    else:
        best = max(results, key=lambda r: r.score)
        logger.info(
            "Multiview disagreement: using single view (score=%.3f)",
            best.score,
        )
        return best.cloud


class Output(TypedDict):
    found: bool
    cloud: PointCloud
    mask: Mask
    score: float


def run(
    ctx: NodeContext,
    cameras: list[CameraFrame],
    object_name: str,
    text_prompts: list[str] | None = None,
    min_points: int = 10,
    min_score: float = 0.3,
    use_multiview: bool = True,
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
    box_margin: int = 20,
    use_point_and_box: bool = True,
) -> Output:
    """Execute DINO-enhanced perception pipeline."""
    if text_prompts is None:
        text_prompts = [object_name]

    results_per_cam: list[CamResult] = []

    for cam in cameras:
        result = _perceive_single_camera_dino(
            ctx, cam, object_name, text_prompts,
            min_score, min_points,
            box_threshold, text_threshold, box_margin, use_point_and_box,
        )
        if result is not None:
            results_per_cam.append(result)

    if not results_per_cam:
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    if use_multiview and len(results_per_cam) > 1:
        merged_cloud = _merge_multiview(results_per_cam)
    else:
        merged_cloud = results_per_cam[0].cloud

    best = max(results_per_cam, key=lambda r: r.score)

    return {
        "found": True,
        "cloud": merged_cloud,
        "mask": best.mask,
        "score": best.score,
    }
