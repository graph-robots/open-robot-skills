"""One-shot DINO+VLM perception: set-of-marks pick, no tournament.

Canonical script for the ``perceiving-objects-oneshot`` skill bundle.
Pipeline:

1. ``grounding-dino.detect`` with the broad ``object.`` prompt — N boxes
   (GroundingDINO's own score/IoU NMS is the only box suppression; no
   skill-level containment filter).
2. Draw letter-labeled overlay (A, B, C, …) on every detection.
3. ONE ``vlm.query`` asking which letter bounds the WHOLE target
   object, or ``none``.
4. ``none`` (or unparseable / empty detections) → ``found: False`` so
   the subgraph exits ``not_found`` cleanly — this is the loop-exit
   signal for clean-all-items workflows.
5. Otherwise: ``sam3.segment_box`` on the chosen box, then
   ``geometry.mask_to_world_points`` for the world-frame cloud.

The VLM prompt template lives in ``prompts/vlm_one_shot.md``.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap.skills import load_prompt
from gap_core.types import BoundingBox2D, CameraFrame, Mask, PointCloud

logger = logging.getLogger(__name__)

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
# Set-of-marks rendering uses a fixed colour palette so the VLM can refer
# to "the red box" if it disagrees with our letters (we never read that
# back — letters are the contract — but consistent colours help the VLM
# anchor on each detection).
_COLORS = [
    (255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 200, 0),
    (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255),
]


def _empty_cloud() -> PointCloud:
    return {"points": np.zeros((0, 3), dtype=np.float32)}


def _empty_mask() -> Mask:
    return np.zeros((0, 0), dtype=np.uint8)


def _box_xyxy(box: BoundingBox2D) -> tuple[int, int, int, int]:
    return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])


def _draw_set_of_marks(
    rgb: np.ndarray, detections: list,
) -> np.ndarray:
    """Render letter-labeled boxes onto a copy of ``rgb``.

    Letters are placed top-left of each box on a coloured background tile
    so they remain readable regardless of underlying texture. Only the
    first ``len(_LABELS)`` detections are labeled; trailing ones are
    drawn boxless because the VLM can't address them anyway.
    """
    import cv2

    out = rgb.copy()
    n = min(len(detections), len(_LABELS))
    for i in range(n):
        det = detections[i]
        x1, y1, x2, y2 = _box_xyxy(det["box"])
        color = _COLORS[i % len(_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Letter tag — solid coloured rectangle with the letter in white.
        label = _LABELS[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.8, 2
        (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
        tx1, ty1 = x1, max(0, y1 - th - 6)
        tx2, ty2 = x1 + tw + 8, max(0, y1)
        cv2.rectangle(out, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            out, label, (tx1 + 4, ty2 - 4),
            font, scale, (255, 255, 255), thick, cv2.LINE_AA,
        )
    return out


def _parse_pick(text: str, n: int) -> int | None:
    """Return the index of the picked letter, or ``None`` if the VLM
    replied "none" / unparseable. Only labels in ``_LABELS[:n]`` count.
    """
    s = text.strip().lower()
    if "none" in s:
        return None
    # Prefer a single-letter response.
    up = s.upper()
    if len(up) == 1 and up in _LABELS[:n]:
        return _LABELS.index(up)
    # Otherwise pick the LAST capital letter mentioned (the typical
    # "Answer: B" pattern ends with the letter).
    last = None
    for m in re.finditer(r"\b([A-H])\b", text):
        idx = _LABELS.index(m.group(1)) if m.group(1) in _LABELS[:n] else -1
        if idx >= 0:
            last = idx
    return last  # may be None


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
    min_score: float = 0.0,
    min_points: int = 100,
    use_multiview: bool = False,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    dino_prompt: str = "object.",
    object_description: str = "",
) -> Output:
    """One-shot DINO+VLM perception with clean ``not_found`` exit.

    Multi-view note: this skill picks the FIRST non-wrist (i.e. exterior)
    camera and runs perception on that single view. The wrist camera is
    ignored — single-view is sufficient for the use cases this skill is
    intended for (clean-all-items loops on the LIBERO floor scene). If
    you need multi-view fusion or wrist fallback, use ``perceiving-objects``.

    Returns ``found: False`` (with empty cloud / mask) when the VLM
    replies "none", when DINO returns no detections, or when the
    selected box has too few segmented points — all of these are loop
    terminator signals for the workflow's ``not_found`` exit.
    """
    if not cameras:
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    # Pick the first exterior camera (skip wrist). All cameras have
    # ``name`` per the observation contract.
    cam = next(
        (c for c in cameras if c["name"] and "eye_in_hand" not in c["name"]),
        cameras[0],
    )

    try:
        gd = ctx.tool(
            "grounding-dino.detect",
            image=cam["rgb"], query=dino_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold,
        )
        detections = list(gd["detections"])
    except Exception as e:
        logger.warning("GDINO detect failed: %s", e)
        detections = []

    if not detections:
        logger.info("perceiving-objects-oneshot: no DINO detections -> not_found")
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    # All DINO detections are kept as candidates — GroundingDINO already
    # applies its own score/IoU NMS. No skill-level containment filter:
    # the VLM is told (see prompts/vlm_one_shot.md) to pick the box that
    # bounds the WHOLE object, so a nested logo/label box simply loses
    # the set-of-marks comparison instead of being pre-filtered.
    detections = detections[: len(_LABELS)]  # cap at 8 (label budget)

    rgb_np = cam["rgb"]
    overlay = _draw_set_of_marks(rgb_np, detections)
    prompt = load_prompt(
        __package__, "vlm_one_shot",
        object_name=object_name,
        object_description=object_description,
    )

    try:
        resp = ctx.tool("vlm.query", prompt=prompt, image=overlay)
        pick = _parse_pick(resp["text"], len(detections))
    except Exception as e:
        logger.warning("VLM query failed: %s — treating as not_found", e)
        pick = None

    if pick is None:
        logger.info(
            "perceiving-objects-oneshot: VLM picked 'none' (or unparseable) "
            "for '%s' from %d detections -> not_found",
            object_name, len(detections),
        )
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    selected = detections[pick]
    logger.info(
        "perceiving-objects-oneshot: VLM picked box %s for '%s'",
        _LABELS[pick], object_name,
    )

    try:
        seg = ctx.tool(
            "sam3.segment_box",
            image=cam["rgb"], box=selected["box"],
        )
    except Exception as e:
        logger.warning("SAM3 segment_box failed: %s", e)
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    if not seg["masks"] or not seg["scores"]:
        logger.info("perceiving-objects-oneshot: SAM3 returned no mask -> not_found")
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    seg_mask = seg["masks"][0]
    seg_score = float(seg["scores"][0])

    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=seg_mask, depth=cam["depth"],
        intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
    )["points"]
    num_points = len(cloud["points"])
    if num_points < min_points:
        logger.info(
            "perceiving-objects-oneshot: cloud too small (%d < %d) -> not_found",
            num_points, min_points,
        )
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}

    return {
        "found": True,
        "cloud": cloud,
        "mask": seg_mask,
        "score": seg_score,
    }
