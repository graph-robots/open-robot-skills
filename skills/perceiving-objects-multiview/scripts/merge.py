"""Merge results from multiple perception methods and select the best via VLM.

Collects candidates from up to three perception methods (DINO, point-based,
DINO+VLM).  When multiple candidates are found, uses the VLM-based
``select_best`` sibling script to pick the best mask.  Calls
``geometry.filter_and_compute_obb`` on the winner's cloud to produce an OBB.
Raises :class:`gap.errors.PerceptionFailed` if no method detected the
object (the subgraph's ``on_error: "not_found"`` catches the raise).
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PerceptionFailed
from gap_core.types import CameraFrame, Mask, OrientedBoundingBox, PointCloud

logger = logging.getLogger(__name__)


class Output(TypedDict):
    cloud: PointCloud
    mask: Mask
    obb: OrientedBoundingBox


def _is_geometrically_plausible(result: dict) -> bool:
    # Reject thin face-only masks: a candidate whose vertical extent is much
    # smaller than its lateral extent AND that is supported by very few points
    # is typically the front face of a box back-projected onto a vertical sliver
    # (e.g. the salad-dressing wrong-target-bottle bug, where the VLM segmented
    # only the visible face and the synthetic-depth back-projection landed on
    # the neighbouring object).
    cloud = result.get("cloud")
    if cloud is None:
        return True
    points = cloud.get("points")
    num_points = 0 if points is None else len(points)
    if num_points >= 500:
        return True
    pre_obb = result.get("obb")
    if pre_obb is None:
        return True
    ex = pre_obb["extent"]["x"]
    ey = pre_obb["extent"]["y"]
    ez = pre_obb["extent"]["z"]
    lateral = max(ex, ey)
    if lateral <= 0:
        return True
    return ez >= 0.3 * lateral


def run(ctx: NodeContext, cameras: list[CameraFrame], object_name: str,
        dino_result: dict | None = None,
        point_result: dict | None = None,
        vlm_result: dict | None = None) -> Output:
    candidates = []
    for result, label in [
        (dino_result, "dino"),
        (point_result, "point"),
        (vlm_result, "vlm"),
    ]:
        if result is not None and result.get("found") and result.get("mask") is not None:
            candidates.append((result, label))

    if not candidates:
        raise PerceptionFailed(f"No perception path found '{object_name}'")

    # Geometry-plausibility pre-filter: drop thin face-only masks before VLM
    # selection.  If every candidate fails the check we fall through with the
    # original list rather than aborting -- the workflow then gets a noisier
    # OBB instead of going straight to abort.
    plausible = [c for c in candidates if _is_geometrically_plausible(c[0])]
    if plausible:
        if len(plausible) != len(candidates):
            dropped = [c[1] for c in candidates if c not in plausible]
            logger.info(
                "merge: dropped geometrically-implausible candidates %s for '%s'",
                dropped, object_name,
            )
        candidates = plausible
    else:
        logger.warning(
            "merge: all %d candidates for '%s' failed geometry plausibility check; "
            "keeping them anyway to avoid abort",
            len(candidates), object_name,
        )

    best_idx = 0
    if len(candidates) > 1:
        try:
            from .select_best import run as select_best
            masks = [c[0]["mask"] for c in candidates]
            labels = [c[1] for c in candidates]
            sel = select_best(
                ctx, image=cameras[0]["rgb"], masks=masks,
                labels=labels, object_name=object_name,
            )
            idx = sel["selected_index"]
            if 0 <= idx < len(candidates):
                best_idx = idx
        except Exception:
            pass

    best_cloud = candidates[best_idx][0]["cloud"]
    best_mask = candidates[best_idx][0]["mask"]

    obb = ctx.tool(
        "geometry.filter_and_compute_obb",
        points=best_cloud, eps=0.005, min_samples=10,
    )["obb"]
    return {"cloud": best_cloud, "mask": best_mask, "obb": obb}
