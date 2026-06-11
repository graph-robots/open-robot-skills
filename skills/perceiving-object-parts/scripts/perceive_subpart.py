"""Hierarchical subpart perception.

DINO detects the parent → VLM disambiguates which box is the actual
parent if multiple were detected → crop the RGB to the parent's box
→ SAM3 segment-text for the subpart inside the crop → uncrop back to
the original image frame → depth-fuse → OBB.

Why crop: SAM3 text segmentation accuracy degrades sharply when the
target occupies <5 % of the image. A frypan handle in a 1024×768
agentview image is ~3 % of pixels; after cropping to the pan box it
becomes ~30 %, well within SAM3's reliable range. The crop also
eliminates false positives from other handles in the scene (drawer
pulls, microwave doors, cabinet handles).

The depth + intrinsics + camera pose stay in the ORIGINAL frame —
only the RGB is cropped. After uncropping the mask back to full
resolution, ``geometry.mask_to_world_points`` consumes the same
original-frame depth/K as ``perceiving-objects`` would.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap.types import (
    BoundingBox2D,
    CameraFrame,
    Mask,
    OrientedBoundingBox,
    PointCloud,
)

logger = logging.getLogger(__name__)

_VLM_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_VLM_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (255, 128, 0), (128, 0, 255),
]


def _empty_cloud() -> PointCloud:
    return {"points": np.zeros((0, 3), dtype=np.float32)}


def _empty_mask() -> Mask:
    return np.zeros((0, 0), dtype=np.uint8)


def _empty_obb() -> OrientedBoundingBox:
    return {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "extent": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


@dataclass
class _CamResult:
    cloud: PointCloud
    mask: Mask
    parent_mask: Mask
    obb: OrientedBoundingBox
    score: float
    # Hold a reference to the source CameraFrame so downstream code
    # (parent_obb computation) can call geometry.mask_to_world_points
    # with the same depth/intrinsics/pose without having to re-find the
    # camera by mask resolution.
    cam: Any = None


def _clip_box_to_image(
    box: BoundingBox2D, img_h: int, img_w: int, pad_px: int,
) -> tuple[int, int, int, int]:
    """Return ``(x1, y1, x2, y2)`` int clipped to image bounds with padding."""
    x1 = max(0, int(box["x1"]) - pad_px)
    y1 = max(0, int(box["y1"]) - pad_px)
    x2 = min(img_w, int(box["x2"]) + pad_px)
    y2 = min(img_h, int(box["y2"]) + pad_px)
    if x2 <= x1 or y2 <= y1:
        # Degenerate box — return the whole image.
        return 0, 0, img_w, img_h
    return x1, y1, x2, y2


def _crop_rgb(
    rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    return rgb[y1:y2, x1:x2, :].copy()


def _uncrop_mask_to_full(
    crop_mask: Mask,
    full_h: int, full_w: int,
    x1: int, y1: int, x2: int, y2: int,
) -> Mask:
    """Paste ``crop_mask`` into a zero ``(full_h, full_w)`` array at (y1, x1)."""
    crop_arr = np.asarray(crop_mask, dtype=np.uint8)
    # SAM3 sometimes returns a mask with shape (h_crop, w_crop) that
    # doesn't exactly match (y2-y1, x2-x1) due to model padding. Resize
    # if necessary.
    expected_h = y2 - y1
    expected_w = x2 - x1
    if crop_arr.shape != (expected_h, expected_w):
        try:
            import cv2
            crop_arr = cv2.resize(
                crop_arr, (expected_w, expected_h),
                interpolation=cv2.INTER_NEAREST,
            )
        except Exception:
            logger.warning(
                "perceive_subpart: crop mask shape %s != expected %s and "
                "cv2 missing; padding with zeros",
                crop_arr.shape, (expected_h, expected_w),
            )
            # Fallback: pad/crop manually.
            tmp = np.zeros((expected_h, expected_w), dtype=np.uint8)
            h_copy = min(crop_arr.shape[0], expected_h)
            w_copy = min(crop_arr.shape[1], expected_w)
            tmp[:h_copy, :w_copy] = crop_arr[:h_copy, :w_copy]
            crop_arr = tmp
    full = np.zeros((full_h, full_w), dtype=np.uint8)
    full[y1:y2, x1:x2] = crop_arr
    return full


def _parse_letter(text: str, n: int) -> int:
    text = text.strip().upper()
    if len(text) == 1 and text in _VLM_LABELS[:n]:
        return _VLM_LABELS.index(text)
    last_idx = -1
    for match in re.finditer(r"\b([A-H])\b", text):
        letter = match.group(1)
        idx = _VLM_LABELS.index(letter) if letter in _VLM_LABELS[:n] else -1
        if idx >= 0:
            last_idx = idx
    return last_idx


def _render_boxes_on_image(
    rgb: np.ndarray, detections: list,
) -> np.ndarray:
    """Annotate the image with labeled boxes for VLM disambiguation."""
    import cv2
    img = rgb.copy()
    original = img.copy()
    cv2.putText(original, "Original", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(original, "Original", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    annotated = img.copy()
    for i, det in enumerate(detections):
        if i >= len(_VLM_LABELS):
            break
        color = _VLM_COLORS[i % len(_VLM_COLORS)]
        b = det["box"]
        cv2.rectangle(annotated, (int(b["x1"]), int(b["y1"])),
                      (int(b["x2"]), int(b["y2"])), color, 3)
        label = _VLM_LABELS[i]
        tx, ty = int(b["x1"]), max(int(b["y1"]) - 8, 20)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    composite = np.concatenate([original, annotated], axis=1)
    return composite


def _pick_parent_box(
    ctx: Any,
    rgb: np.ndarray,
    parent_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> BoundingBox2D | None:
    """Pick the parent box by DINO + (when ambiguous) VLM disambiguation.

    DINO runs with the parent-class prompt to catch every salient
    region. If only one detection matches the parent class we take it
    directly; for multiple boxes we ask the VLM which one is the
    actual ``parent_prompt`` — same disambiguation pattern as
    ``perceiving-objects``.
    """
    # Broad DINO sweep — ``parent_prompt.`` works for class-level
    # queries; the VLM step picks the right instance.
    dino_prompt = parent_prompt if parent_prompt.endswith(".") else parent_prompt + "."
    try:
        resp = ctx.tool(
            "grounding-dino.detect",
            image=rgb, query=dino_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold,
        )
    except Exception as exc:
        logger.warning("perceive_subpart: DINO detect failed: %s", exc)
        return None
    detections = list(resp["detections"])
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]["box"]

    # Multiple boxes → use VLM to pick the one that's actually the
    # parent_prompt (filters out e.g. a stove burner that DINO grouped
    # under "frying pan." because the prompts are loose).
    n = min(len(detections), len(_VLM_LABELS))
    annotated = _render_boxes_on_image(rgb, detections[:n])
    vlm_prompt = (
        f"The image on the right shows {n} candidate bounding boxes "
        f"labeled {', '.join(_VLM_LABELS[:n])}. Which box is the actual "
        f"{parent_prompt}? Reply with a single letter."
    )
    try:
        vlm_resp = ctx.tool(
            "vlm.query",
            prompt=vlm_prompt, image=annotated,
        )
        idx = _parse_letter(vlm_resp["text"], n)
        if 0 <= idx < n:
            logger.info(
                "perceive_subpart: VLM picked box %d/%d for parent %r (text=%r)",
                idx, n, parent_prompt, vlm_resp["text"][:80],
            )
            return detections[idx]["box"]
    except Exception as exc:
        logger.warning(
            "perceive_subpart: VLM disambiguation failed (%s); "
            "falling back to top-confidence box", exc,
        )
    # Fallback: top-confidence DINO detection (which is what DINO
    # ranks first by score anyway).
    return detections[0]["box"]


def _perceive_one_camera(
    ctx: Any,
    cam: CameraFrame,
    parent_prompt: str,
    subpart_prompt: str,
    padding_px: int,
    min_score: float,
    min_points: int,
    box_threshold: float,
    text_threshold: float,
    max_subpart_area_frac: float,
) -> _CamResult | None:
    # Wrist camera is too zoomed for parent localization — skip.
    if cam["name"] and "eye_in_hand" in cam["name"]:
        return None

    img_h, img_w = cam["rgb"].shape[:2]

    parent_box = _pick_parent_box(
        ctx, cam["rgb"], parent_prompt, box_threshold, text_threshold,
    )
    if parent_box is None:
        logger.debug(
            "perceive_subpart: no DINO box for parent %r in cam %r",
            parent_prompt, cam["name"],
        )
        return None

    x1, y1, x2, y2 = _clip_box_to_image(
        parent_box, img_h=img_h, img_w=img_w,
        pad_px=padding_px,
    )
    cropped = _crop_rgb(cam["rgb"], x1, y1, x2, y2)

    try:
        parent_seg = ctx.tool(
            "sam3.segment_box",
            image=cam["rgb"], box=parent_box,
        )
    except Exception as exc:
        logger.warning(
            "perceive_subpart: sam3.segment_box(parent) failed in cam %r: %s",
            cam["name"], exc,
        )
        return None
    if not parent_seg["masks"]:
        logger.debug(
            "perceive_subpart: SAM3 returned no parent mask for %r in cam %r",
            parent_prompt, cam["name"],
        )
        return None
    parent_mask = parent_seg["masks"][0]

    # Locate the SUBPART inside the parent crop via DINO. SAM3 segment_text
    # with a part-name prompt frequently returns the whole parent (its
    # strongest in-crop segment) because the part is a small fraction of
    # the cropped image. Running grounding-dino.detect with the subpart
    # prompt on the crop gives a tight bounding box; sam3.segment_box on
    # that box then produces an accurate subpart mask.
    try:
        sub_dino = ctx.tool(
            "grounding-dino.detect",
            image=cropped,
            query=subpart_prompt if subpart_prompt.endswith(".") else subpart_prompt + ".",
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    except Exception as exc:
        logger.warning(
            "perceive_subpart: DINO subpart detect failed in cam %r: %s",
            cam["name"], exc,
        )
        return None
    sub_dets = list(sub_dino["detections"])
    if not sub_dets:
        logger.debug(
            "perceive_subpart: DINO returned no subpart box for %r in cam %r",
            subpart_prompt, cam["name"],
        )
        return None

    # A subpart is — by definition — substantially SMALLER than its
    # parent. DINO, given a part-name prompt on the parent crop,
    # routinely returns the whole-parent box too (often with a HIGHER
    # score than the true part, because the parent is the dominant
    # salient region). Picking purely by score then selects the whole
    # pan, not the handle. Filter to detections whose area is a small
    # fraction of the PARENT BOX footprint (NOT the padded crop — the
    # crop adds ~2× padding area, so a whole-pan box is only ~54% of
    # the crop but ~114% of the parent box). Then take the best-scoring
    # of those. ``max_subpart_area_frac`` default 0.55: a true subpart
    # (frypan handle ≈ 17% of the pan box) passes; whole-pan boxes
    # (≈ 97–114% of the parent box) are rejected.
    #
    # ``parent_box`` is in FULL-image coords; translating it into crop
    # coords doesn't change its area, so compare raw box areas.
    parent_area = float(
        max(1.0, (parent_box["x2"] - parent_box["x1"]) * (parent_box["y2"] - parent_box["y1"]))
    )

    def _box_area(box: BoundingBox2D) -> float:
        return float(max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"]))

    small = [
        d for d in sub_dets
        if _box_area(d["box"]) / parent_area <= max_subpart_area_frac
    ]
    if small:
        small.sort(key=lambda d: float(d["score"]), reverse=True)
        subpart_box = small[0]["box"]
        subpart_score = float(small[0]["score"])
        logger.info(
            "perceive_subpart: subpart %r → box area %.0f%% of parent "
            "(score %.3f); rejected %d whole-parent box(es)",
            subpart_prompt,
            100.0 * _box_area(subpart_box) / parent_area,
            subpart_score,
            len(sub_dets) - len(small),
        )
    else:
        # Every detection is ~parent-sized: DINO never isolated the
        # part. Fall back to the SMALLEST box (closest to a part) so
        # we at least bias toward a sub-region rather than the union.
        sub_dets.sort(key=lambda d: _box_area(d["box"]))
        subpart_box = sub_dets[0]["box"]
        subpart_score = float(sub_dets[0]["score"])
        logger.warning(
            "perceive_subpart: no subpart box < %.0f%% of parent for %r "
            "in cam %r; falling back to smallest (area %.0f%%)",
            100.0 * max_subpart_area_frac, subpart_prompt, cam["name"],
            100.0 * _box_area(subpart_box) / parent_area,
        )

    score = subpart_score
    if score < min_score:
        logger.debug(
            "perceive_subpart: DINO subpart score %.3f < %.3f for %r",
            score, min_score, subpart_prompt,
        )
        return None

    # SECOND-LEVEL CROP. ``subpart_box`` is in parent-crop coords;
    # translate to FULL-image coords, then crop the original RGB tight
    # to the handle. Running SAM3 on the *parent* crop with the handle
    # box as a mere prompt makes SAM3 expand the box to the whole
    # connected object (the handle is physically attached to the pan,
    # so the segment grows to the entire pan). Cropping to the handle
    # box first removes most of the pan body from the frame, so SAM3
    # can only segment the handle. This is the hierarchical zoom-in the
    # skill promises: parent-detect → parent-crop → subpart-detect →
    # subpart-crop → subpart-segment.
    sb_full: BoundingBox2D = {
        "x1": float(subpart_box["x1"]) + x1,
        "y1": float(subpart_box["y1"]) + y1,
        "x2": float(subpart_box["x2"]) + x1,
        "y2": float(subpart_box["y2"]) + y1,
    }
    # Tight padding (8px): enough for SAM context, small enough that the
    # pan body stays out of frame.
    hx1, hy1, hx2, hy2 = _clip_box_to_image(
        sb_full, img_h=img_h, img_w=img_w, pad_px=8,
    )
    subpart_crop = _crop_rgb(cam["rgb"], hx1, hy1, hx2, hy2)

    try:
        seg_resp = ctx.tool(
            "sam3.segment_text",
            image=subpart_crop, query=subpart_prompt,
        )
    except Exception as exc:
        logger.warning(
            "perceive_subpart: sam3.segment_text(subpart-crop) failed in cam %r: %s",
            cam["name"], exc,
        )
        return None
    if not seg_resp["masks"]:
        logger.debug(
            "perceive_subpart: SAM3 no mask for subpart %r in subpart-crop, cam %r",
            subpart_prompt, cam["name"],
        )
        return None
    sub_crop_mask = seg_resp["masks"][0]

    # Uncrop the handle mask from subpart-crop coords directly to FULL
    # image coords (single paste at the handle box origin).
    full_mask = _uncrop_mask_to_full(
        sub_crop_mask, full_h=img_h, full_w=img_w,
        x1=hx1, y1=hy1, x2=hx2, y2=hy2,
    )

    # Depth-fuse the FULL mask against the FULL camera. Intrinsics and
    # depth are unchanged from the original observation; only the RGB
    # was cropped (for SAM's benefit).
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=full_mask, depth=cam["depth"],
        intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
    )["points"]
    num_points = len(cloud["points"])
    if num_points < min_points:
        logger.debug(
            "perceive_subpart: %d < %d points for subpart %r in cam %r",
            num_points, min_points, subpart_prompt, cam["name"],
        )
        return None

    # Deliberately split geometry.filter_noise + geometry.compute_obb
    # rather than the fused geometry.filter_and_compute_obb: a thin
    # subpart cloud (a handle is a few hundred surface points) can lose
    # most of its points to DBSCAN, so we keep the unfiltered cloud as
    # the OBB source whenever filtering strips it below ``min_points``
    # — a fallback the fused tool cannot express. (Historical note: the
    # legacy service's FilterAndComputeOBB also ended with a
    # rehearsal-only snap-to-ground-truth step that silently replaced a
    # subpart OBB with the WHOLE-PARENT OBB; the landed geometry tool
    # no longer snaps, but the split-call structure stays.)
    filtered = ctx.tool(
        "geometry.filter_noise",
        points=cloud,
    )["points"]
    obb_src = filtered if len(filtered["points"]) >= min_points else cloud
    obb = ctx.tool(
        "geometry.compute_obb",
        points=obb_src,
    )["obb"]
    return _CamResult(
        cloud=cloud, mask=full_mask, parent_mask=parent_mask,
        obb=obb, score=score, cam=cam,
    )


class Output(TypedDict):
    found: bool
    obb: OrientedBoundingBox
    mask: Mask
    cloud: PointCloud
    subpart_mask: Mask
    score: float
    # Parent geometry — the WHOLE object's OBB and point cloud (e.g. the
    # full frying pan when the subpart is its handle). Consumed by
    # downstream nodes that need the parent body for placement /
    # collision reasoning, e.g. a transport skill's drop-offset pose
    # which shifts the drop pose so the parent centroid (not the held
    # subpart) lands at the zone centroid.
    parent_obb: OrientedBoundingBox
    parent_cloud: PointCloud


def run(
    ctx: NodeContext,
    cameras: list[CameraFrame],
    parent_prompt: str,
    subpart_prompt: str,
    padding_px: int = 30,
    min_score: float = 0.3,
    min_points: int = 10,
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
    max_subpart_area_frac: float = 0.55,
) -> Output:
    """Hierarchical subpart perception fused across all cameras.

    Each camera that detects the subpart contributes a world-frame
    point cloud of the subpart's *visible surface*. A single camera
    only images the front shell of a thin part (e.g. a pan handle), so
    its OBB centroid sits ON that surface — biased toward the camera by
    ~half the part's unseen thickness, which makes a parallel-jaw grasp
    close just off the bar. We therefore UNION the per-camera clouds
    (concatenate — NOT a KD-tree intersection, which would be noisy for
    a small part) and recompute the OBB on the fused cloud: two
    complementary views (e.g. an oblique exterior + a top-down
    eye-in-hand) cover enough of the part that the centroid converges
    to the true mid. Self-limiting: with one camera the fused cloud is
    that one cloud, so behavior is unchanged from single-view.
    """
    results: list[_CamResult] = []
    for cam in cameras:
        r = _perceive_one_camera(
            ctx, cam, parent_prompt, subpart_prompt,
            padding_px=padding_px, min_score=min_score,
            min_points=min_points,
            box_threshold=box_threshold, text_threshold=text_threshold,
            max_subpart_area_frac=max_subpart_area_frac,
        )
        if r is not None:
            results.append(r)
    if not results:
        logger.info(
            "perceive_subpart: no acceptable detection for parent=%r "
            "subpart=%r across %d cameras",
            parent_prompt, subpart_prompt, len(cameras),
        )
        return {
            "found": False,
            "obb": _empty_obb(),
            "mask": _empty_mask(),
            "cloud": _empty_cloud(),
            "subpart_mask": _empty_mask(),
            "score": 0.0,
            "parent_obb": _empty_obb(),
            "parent_cloud": _empty_cloud(),
        }
    best = max(results, key=lambda r: r.score)

    # Fuse the per-camera subpart clouds by UNION (concatenate world
    # points), then recompute the OBB on the fused cloud. With ≥2
    # complementary views this moves the OBB centroid off the single
    # front-surface shell toward the part's true mid, so a parallel-jaw
    # grasp closes ON the bar instead of ~half-a-thickness off it. With
    # one camera the union is that one cloud → identical to before.
    fused_pts = np.concatenate(
        [np.asarray(r.cloud["points"], dtype=np.float32).reshape(-1, 3)
         for r in results],
        axis=0,
    )
    fused_cloud: PointCloud = {"points": fused_pts}
    if len(results) >= 2 and len(fused_cloud["points"]) >= min_points:
        filtered = ctx.tool(
            "geometry.filter_noise",
            points=fused_cloud,
        )["points"]
        obb_src = (
            filtered if len(filtered["points"]) >= min_points else fused_cloud
        )
        fused_obb = ctx.tool(
            "geometry.compute_obb",
            points=obb_src,
        )["obb"]
        out_obb = fused_obb
        out_cloud = fused_cloud
    else:
        # Single camera hit — keep the per-camera OBB/cloud as-is.
        out_obb = best.obb
        out_cloud = best.cloud
    logger.info(
        "perceive_subpart: parent=%r subpart=%r → mask score %.3f, "
        "fused %d 3D points from %d camera hit(s) "
        "(OBB center=(%.3f,%.3f,%.3f) extent=(%.3f,%.3f,%.3f))",
        parent_prompt, subpart_prompt, best.score,
        len(out_cloud["points"]), len(results),
        out_obb["center"]["x"], out_obb["center"]["y"], out_obb["center"]["z"],
        out_obb["extent"]["x"], out_obb["extent"]["y"], out_obb["extent"]["z"],
    )

    # ----------------------------------------------------------------
    # Compute the PARENT OBB + cloud from the best camera's parent_mask
    # so downstream nodes (e.g. a transport skill's drop-offset pose) can
    # know where the FULL object centroid is, not just the subpart's.
    # We use the best camera only (no union across cameras) — the parent
    # is large enough that a single-view cloud lands its centroid close
    # to the true parent centroid; multi-view union would be marginal
    # gain at the cost of two extra tool calls per camera.
    parent_obb = _empty_obb()
    parent_cloud = _empty_cloud()
    try:
        # Use the same camera that produced the best subpart detection
        # — its intrinsics/depth/pose are the ones whose parent_mask we
        # have. mask_to_world_points' kwargs mirror the existing call
        # above: ``mask, depth, intrinsics, camera_pose`` (NOT ``rgb``,
        # NOT ``extrinsics`` — those names will be rejected).
        if best.cam is not None:
            p_cloud = ctx.tool(
                "geometry.mask_to_world_points",
                mask=best.parent_mask, depth=best.cam["depth"],
                intrinsics=best.cam["intrinsics"],
                camera_pose=best.cam["pose"],
            )["points"]
            if len(p_cloud["points"]) >= min_points:
                parent_cloud = p_cloud
                p_filtered = ctx.tool(
                    "geometry.filter_noise",
                    points=p_cloud,
                )["points"]
                p_obb_src = (
                    p_filtered
                    if len(p_filtered["points"]) >= min_points
                    else p_cloud
                )
                parent_obb = ctx.tool(
                    "geometry.compute_obb",
                    points=p_obb_src,
                )["obb"]
    except Exception as e:
        # Non-fatal — parent OBB is optional; downstream that needs it
        # will see an empty OBB and degrade gracefully (drop_offset_pose
        # becomes a no-op when parent and held OBB are equal/empty).
        logger.warning(
            "perceive_subpart: failed to compute parent OBB (%s); "
            "returning empty parent_obb/parent_cloud", e,
        )

    # NOTE: `mask` returned here is the PARENT object mask (whole pan),
    # not the subpart. Downstream world-building uses this to isolate
    # the WHOLE parent body in the collision world via
    # ``geometry.build_world_config``'s ``object_masks`` entry, then the
    # grasp planner ignores that obstacle. If we returned only the
    # subpart mask, the rest of the parent body (the pan minus its
    # handle) would remain in the collision world and block every
    # approach trajectory. The subpart OBB/cloud still drive grasp pose
    # generation; only the collision isolation needs to encompass the
    # whole parent.
    return {
        "found": True,
        "obb": out_obb,
        "mask": best.parent_mask,
        "cloud": out_cloud,
        "subpart_mask": best.mask,
        "score": best.score,
        "parent_obb": parent_obb,
        "parent_cloud": parent_cloud,
    }
