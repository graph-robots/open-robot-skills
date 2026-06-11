"""DINO+VLM multiview perception: GDINO broad detect → VLM box selection → SAM3 → 3D fusion.

One of the three detector scripts of the ``perceiving-objects-multiview``
skill bundle. The generated subgraph references this as a
``type: script`` state. Internally it calls:

- ``grounding-dino.detect`` (broad ``object.`` text prompt)
- ``vlm.query`` to disambiguate which DINO box matches the target
- ``sam3.segment_box`` to segment the chosen box
- ``geometry.mask_to_world_points`` to fuse depth into a world-frame cloud

The VLM prompt template lives in ``prompts/vlm_select_box.md`` and is
loaded at call time via :func:`gap.skills.load_prompt`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap.skills import load_prompt
from gap.types import CameraFrame, Mask, PointCloud

logger = logging.getLogger(__name__)

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_INTERSECTION_DIST = 0.01
_MIN_INTERSECTION = 1


def _empty_cloud() -> PointCloud:
    return {"points": np.zeros((0, 3), dtype=np.float32)}


def _empty_mask() -> Mask:
    return np.zeros((0, 0), dtype=np.uint8)


def _render_boxes_on_image(
    rgb: np.ndarray,
    detections: list,
) -> np.ndarray:
    import cv2

    img_np = rgb.copy()

    original = img_np.copy()
    cv2.putText(original, "Original", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(original, "Original", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

    annotated = img_np.copy()
    _COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 128, 0), (128, 0, 255),
    ]

    for i, det in enumerate(detections):
        if i >= len(_LABELS):
            break
        color = _COLORS[i % len(_COLORS)]
        b = det["box"]
        pt1 = (int(b["x1"]), int(b["y1"]))
        pt2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(annotated, pt1, pt2, color, 3)

        label = _LABELS[i]
        tx, ty = int(b["x1"]), max(int(b["y1"]) - 8, 20)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    composite = np.concatenate([original, annotated], axis=1)
    return composite


@dataclass
class _CamResult:
    cloud: Any
    mask: Any
    score: float


def _parse_letter(text: str, n: int) -> int:
    text = text.strip().upper()
    if len(text) == 1 and text in _LABELS[:n]:
        return _LABELS.index(text)

    last_idx = -1
    for match in re.finditer(r"\b([A-H])\b", text):
        letter = match.group(1)
        idx = _LABELS.index(letter) if letter in _LABELS[:n] else -1
        if idx >= 0:
            last_idx = idx

    return last_idx


def _perceive_single_camera(
    ctx: Any,
    cam: CameraFrame,
    object_name: str,
    text_prompts: list[str],
    min_score: float,
    min_points: int,
    box_threshold: float,
    text_threshold: float,
    dino_prompt: str,
    object_description: str,
) -> _CamResult | None:
    is_wrist = cam["name"] and "eye_in_hand" in cam["name"]

    seg_mask = None
    seg_score = 0.0

    if not is_wrist:
        gdino_detections: list = []
        try:
            gdino_resp = ctx.tool(
                "grounding-dino.detect",
                image=cam["rgb"], query=dino_prompt,
                box_threshold=box_threshold, text_threshold=text_threshold,
            )
            gdino_detections = list(gdino_resp["detections"])
        except Exception as e:
            logger.warning("GDINO detect failed (perceive_dino_vlm): %s", e)

        if gdino_detections:
            n = min(len(gdino_detections), len(_LABELS))
            annotated_image = _render_boxes_on_image(cam["rgb"], gdino_detections[:n])

            vlm_prompt = load_prompt(
                __package__, "vlm_select_box",
                n=n,
                label_list=", ".join(_LABELS[:n]),
                object_name=object_name,
                object_description=object_description,
            )

            try:
                vlm_resp = ctx.tool(
                    "vlm.query",
                    prompt=vlm_prompt, image=annotated_image,
                )

                selected_idx = _parse_letter(vlm_resp["text"], n)
                logger.info(
                    "VLM selected box %d (text=%r) for '%s' from %d detections",
                    selected_idx, vlm_resp["text"][:80], object_name, n,
                )

                if 0 <= selected_idx < n:
                    selected_det = gdino_detections[selected_idx]
                    seg_resp = ctx.tool(
                        "sam3.segment_box",
                        image=cam["rgb"], box=selected_det["box"],
                    )
                    if seg_resp["masks"] and seg_resp["scores"]:
                        seg_mask = seg_resp["masks"][0]
                        seg_score = seg_resp["scores"][0]
            except Exception as e:
                logger.warning("VLM box selection failed (perceive_dino_vlm): %s", e)

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

    return _CamResult(cloud=cloud, mask=seg_mask, score=seg_score)


def _merge_multiview(results: list[_CamResult]) -> PointCloud:
    from scipy.spatial import cKDTree

    if len(results) < 2:
        return results[0].cloud

    clouds_np: list[np.ndarray] = []
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
    dino_prompt: str = "object.",
    object_description: str = "",
) -> Output:
    """Execute DINO+VLM perception pipeline across all cameras."""
    if text_prompts is None:
        text_prompts = [object_name]

    results_per_cam: list[_CamResult] = []

    for cam in cameras:
        result = _perceive_single_camera(
            ctx, cam, object_name, text_prompts,
            min_score, min_points,
            box_threshold, text_threshold, dino_prompt, object_description,
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
