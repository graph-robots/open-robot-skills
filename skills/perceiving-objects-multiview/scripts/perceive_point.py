"""Point-based multiview perception: Molmo -> SAM3 point/text -> depth-to-3D -> fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap.types import CameraFrame, Mask, PointCloud

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


def _perceive_single_camera_point(
    ctx: Any,
    cam: CameraFrame,
    object_name: str,
    text_prompts: list[str],
    min_score: float,
    min_points: int,
) -> CamResult | None:
    """Run point-based perception on a single camera view.

    Fixed cameras: Molmo point -> SAM3 point prompt (text fallback).
    Wrist cameras: SAM3 text prompt directly (skip Molmo).
    """
    is_wrist = cam["name"] and "eye_in_hand" in cam["name"]

    seg_mask = None
    seg_score = 0.0

    if not is_wrist:
        # Molmo is optional — if the endpoint is not deployed or the call
        # fails (GAP_MOLMO_BASE_URL unset, transient HTTP error), fall
        # through to the SAM3 text-prompt path below instead of crashing
        # the subgraph.
        try:
            point_resp = ctx.tool(
                "molmo.point_prompt",
                image=cam["rgb"], query=object_name,
            )
        except Exception as exc:
            logger.debug(
                "molmo.point_prompt unavailable for '%s' (%s: %s); falling "
                "back to SAM3 text prompt",
                object_name, type(exc).__name__, exc,
            )
            point_resp = None

        if point_resp is not None and point_resp["found"]:
            seg_resp = ctx.tool(
                "sam3.segment_point",
                image=cam["rgb"],
                pixel_x=point_resp["pixel_x"],
                pixel_y=point_resp["pixel_y"],
            )
            if seg_resp["masks"] and seg_resp["scores"]:
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
        logger.debug("No acceptable mask for '%s' in cam '%s'",
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
            "Too few points (%d < %d) for '%s' in cam '%s'",
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
) -> Output:
    """Execute point-based perception pipeline."""
    if text_prompts is None:
        text_prompts = [object_name]

    results_per_cam: list[CamResult] = []

    for cam in cameras:
        result = _perceive_single_camera_point(
            ctx, cam, object_name, text_prompts, min_score, min_points,
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
