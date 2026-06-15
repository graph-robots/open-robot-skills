"""Ground a free-form placement description into a 3D ``placement_zone_obb``.

Given a natural-language phrase describing WHERE the held object should be
placed (e.g. ``"the left compartment of the caddy"``, ``"the inside of
the top drawer of the cabinet"``, ``"on top of the cabinet shelf"``),
locate the corresponding region in the scene and return its OBB. The OBB
is consumed by ``compute_drop_pose.py`` via the existing
``container_interior_obb`` parameter (already plumbed; the zone branch of
that script handles it correctly).

Two-path pipeline (single ``type: script`` state, mirrors
``perceiving-objects``'s picking pattern):

1. ``robot.get_observation`` — fresh RGBD frames. Pick the
   agentview camera (skip ``eye_in_hand`` — the wrist view doesn't help
   locate above-workspace zones).

2. **Path A — DINO + VLM-letter** (primary):
   - ``grounding-dino.detect`` with a sub-region-rich prompt
     (``"rectangular hole. dark hole. opening. drawer. ..."``).
   - Render the top up-to-8 detections labeled ``A..H`` on the image.
   - ``vlm.query`` with ``prompts/vlm_select_zone.md`` — the VLM picks a
     single letter (or ``"none"``). No coordinate output, so no
     pixel-coordinate hallucination is possible.
   - ``sam3.segment_box`` on the chosen detection.

3. **Path B — sam3.segment_text** (fallback when DINO returns no boxes,
   the VLM replies ``"none"``, or the SAM3 score is too low):
   - ``sam3.segment_text(image, placement_description)`` and a
     leading-article-stripped variant.

4. **No mask** → return ``{placement_zone_obb: None}``;
   ``compute_drop_pose`` falls back to the bare-container path
   (preserves the legacy 3-state behavior).

5. ``geometry.mask_to_world_points`` → cloud (drop if too few points).
   ``geometry.filter_and_compute_obb`` → OBB. Sanity-gate by Z extent
   and a workspace AABB; return ``None`` on any gate failure.

When ``placement_description`` is empty/None, return ``None``
immediately with no tool calls.

Why no VLM pixel-pointing: empirically the small open-source VLMs
reason correctly about which region is meant but hallucinate pixel
coordinates by hundreds of pixels, so a projected 3D point lands far
from the intended zone (verified in iteration 1, regression to 1/18
pass rate). Reducing the VLM's job to single-letter selection over
DINO-anchored boxes eliminates that failure mode entirely.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap.skills import load_prompt
from gap_core.types import Mask, OrientedBoundingBox

logger = logging.getLogger(__name__)


# -- Helpers mirrored from the perceiving-objects canonical script.
# -- Cross-bundle imports of canonical scripts are fragile in the
# -- synthetic-package layout, so we keep a local copy. If the upstream
# -- helpers drift, update both copies in lockstep.

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _render_boxes_on_image(image: np.ndarray, detections: list) -> np.ndarray:
    """Side-by-side composite: original frame + frame with labeled boxes."""
    import cv2

    img_np = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

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

    return np.concatenate([original, annotated], axis=1)


def _parse_letter(text: str, n: int) -> int:
    """Return the index of the chosen letter, or -1 for none/unparseable."""
    text = text.strip().upper()
    if len(text) == 1 and text in _LABELS[:n]:
        return _LABELS.index(text)

    # If the model replied with "NONE" (or a sentence containing it
    # before any letter), defer to the fallback path.
    if "NONE" in text and not re.search(r"\b([A-H])\b", text):
        return -1

    last_idx = -1
    for match in re.finditer(r"\b([A-H])\b", text):
        letter = match.group(1)
        idx = _LABELS.index(letter) if letter in _LABELS[:n] else -1
        if idx >= 0:
            last_idx = idx

    return last_idx


# -- End of mirrored helpers


# Workspace bounds for sanity-checking the zone OBB. Generous box around
# the LIBERO table; rejects depth-NaN points that project to absurd
# coordinates.
_WORKSPACE_X = (-0.20, 1.20)
_WORKSPACE_Y = (-0.80, 0.80)
_WORKSPACE_Z = (-0.05, 1.20)

# DINO prompt tuned empirically on a cached LIBERO caddy frame: visual-feature
# descriptors (``rectangular hole``, ``dark hole``) reliably surface each of
# the four caddy compartments as individual boxes, while the conceptual word
# ``compartment`` collapses onto the WHOLE caddy because that's "the object
# with compartments". Drawers and shelves also work with the same prompt
# because their open cavities and surfaces are both visually distinct dark
# rectangles or flat panels.
_DEFAULT_DINO_PROMPT = (
    "rectangular hole. dark hole. opening. drawer. shelf. compartment. cubby."
)


class Output(TypedDict):
    placement_zone_obb: OrientedBoundingBox | None


def _box_iou(a, b) -> float:
    """IoU of two 2D bounding boxes."""
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _box_center_in_mask(box, mask: Mask) -> bool:
    """Return True if the integer center pixel of ``box`` lies on ``mask``."""
    arr = np.asarray(mask)
    h, w = arr.shape[:2]
    cx = int((box["x1"] + box["x2"]) / 2)
    cy = int((box["y1"] + box["y2"]) / 2)
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return False
    return bool(arr[cy, cx] > 0)


def _filter_to_container(detections, container_mask: Mask | None) -> list:
    """Keep detections whose box center falls inside ``container_mask``.

    DINO surfaces both the visible compartment openings (small boxes
    centered on the caddy) and noise boxes elsewhere in the image (dark
    patches in the wall, table edges). The noise boxes have similar
    scores to the real compartments, so without spatial filtering they
    push real compartments out of the letter-labeled top-8. The caddy
    mask comes "for free" from the upstream container perception
    subgraph; centering the filter on it is a clean spatial prior.
    """
    if container_mask is None or np.asarray(container_mask).size == 0:
        return list(detections)
    return [d for d in detections if _box_center_in_mask(d["box"], container_mask)]


def _nms_dedupe(detections, iou_threshold: float = 0.7) -> list:
    """Keep the highest-score detection for each cluster of overlapping boxes.

    DINO often returns the same region with multiple labels (e.g. one
    box labeled both ``"drawer"`` and ``"shelf compartment"``). Without
    NMS, two near-identical whole-container detections can squat at the
    top of the list and push individual sub-region boxes out of the
    letter-labeled top-8. NMS by IoU keeps one representative per
    cluster, freeing slots for the compartment-sized detections that
    the VLM actually needs to choose between.
    """
    sorted_dets = sorted(detections, key=lambda d: -d["score"])
    kept = []
    for det in sorted_dets:
        if any(_box_iou(det["box"], k["box"]) > iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def _try_dino_vlm_letter(
    ctx: Any,
    cam: Any,
    placement_description: str,
    dino_prompt: str,
    box_threshold: float,
    text_threshold: float,
    min_score: float,
    container_mask: Mask | None,
) -> tuple[Mask | None, float]:
    """Path A: DINO over-detect → mask filter → NMS dedupe → VLM picks letter → sam3.segment_box."""
    try:
        gdino_resp = ctx.tool(
            "grounding-dino.detect",
            image=cam["rgb"], query=dino_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold,
        )
    except Exception as e:
        logger.warning("GDINO detect failed: %s", e)
        return None, 0.0

    raw_detections = list(gdino_resp["detections"])
    if not raw_detections:
        logger.info("GDINO returned 0 detections for prompt %r", dino_prompt)
        return None, 0.0

    in_container = _filter_to_container(raw_detections, container_mask)
    detections = _nms_dedupe(in_container, iou_threshold=0.7)
    logger.info(
        "GDINO: %d raw → %d in container_mask → %d after NMS (prompt=%r)",
        len(raw_detections), len(in_container), len(detections), dino_prompt,
    )
    if not detections:
        return None, 0.0

    n = min(len(detections), len(_LABELS))
    annotated_image = _render_boxes_on_image(cam["rgb"], detections[:n])

    vlm_prompt = load_prompt(
        __package__, "vlm_select_zone",
        n=n,
        label_list=", ".join(_LABELS[:n]),
        placement_description=placement_description,
    )

    try:
        vlm_resp = ctx.tool(
            "vlm.query",
            prompt=vlm_prompt, image=annotated_image,
        )
    except Exception as e:
        logger.warning("VLM query failed for placement zone: %s", e)
        return None, 0.0

    vlm_text = vlm_resp["text"]
    selected_idx = _parse_letter(vlm_text, n)
    logger.info(
        "VLM placement-zone letter=%r idx=%d (text=%r) for %r from %d detections",
        vlm_text.strip()[:40], selected_idx, vlm_text[:80],
        placement_description, n,
    )
    if selected_idx < 0:
        return None, 0.0

    selected_det = detections[selected_idx]
    try:
        seg_resp = ctx.tool(
            "sam3.segment_box",
            image=cam["rgb"], box=selected_det["box"],
        )
    except Exception as e:
        logger.warning("SAM3 segment_box failed for placement zone: %s", e)
        return None, 0.0

    if not seg_resp["masks"] or not seg_resp["scores"]:
        return None, 0.0
    if seg_resp["scores"][0] < min_score:
        logger.info(
            "SAM3 segment_box score %.2f below threshold %.2f",
            seg_resp["scores"][0], min_score,
        )
        return None, 0.0
    return seg_resp["masks"][0], seg_resp["scores"][0]


_INTERIOR_KEYWORDS = ("drawer", "cabinet", "cubby", "compartment", "basket", "tray")


def _try_segment_text(
    ctx: Any,
    cam: Any,
    placement_description: str,
    min_score: float,
) -> list[tuple[str, Mask, float]]:
    """Path B: sam3.segment_text with the description and a few rephrasings.

    Variants tried in order:
      1. The description verbatim.
      2. With leading article stripped.
      3. If the description mentions a container that has an interior
         (drawer/cabinet/etc.) without already qualifying it with
         ``inside`` / ``interior`` / ``inside of`` / ``open``, prepend
         ``"the inside of "`` — empirically the LLM coordinator
         sometimes drops the "inside of" qualifier and then the bare
         "the bottom drawer" mask covers the drawer FACE, not the
         cavity. This recovers task 26 across description variants.
    """
    desc_lower = placement_description.lower()
    has_interior_qualifier = any(
        k in desc_lower for k in ("inside", "interior", "open ")
    )
    has_interior_target = any(k in desc_lower for k in _INTERIOR_KEYWORDS)
    # Only auto-add "inside of" when the description is shaped like a
    # bare container noun phrase ("the bottom drawer", "the left
    # compartment of the caddy"), not when it leads with a positional
    # preposition ("on top of …", "above …", "below …", "left of …").
    starts_positional = bool(
        re.match(
            r"^(on(\s+top)?\s+of|above|below|under|next\s+to|left\s+of|right\s+of|in\s+front\s+of|behind)\b",
            desc_lower,
        )
    )

    raw_variants: list[str] = [
        placement_description,
        re.sub(r"^(the|a|an)\s+", "", placement_description.strip(), flags=re.IGNORECASE),
    ]
    if has_interior_target and not has_interior_qualifier and not starts_positional:
        s_clean = re.sub(r"^(the|a|an)\s+", "", placement_description.strip(), flags=re.IGNORECASE)
        raw_variants.append(f"the inside of the {s_clean}")
        raw_variants.append(f"open {s_clean}")

    variants: list[str] = []
    seen: set[str] = set()
    for s in raw_variants:
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            variants.append(s)

    out: list[tuple[str, Mask, float]] = []
    for prompt in variants:
        try:
            seg_resp = ctx.tool(
                "sam3.segment_text",
                image=cam["rgb"], query=prompt,
            )
        except Exception as e:
            logger.warning("SAM3 segment_text failed for %r: %s", prompt, e)
            continue
        if not seg_resp["masks"] or not seg_resp["scores"]:
            continue
        score = seg_resp["scores"][0]
        if score >= min_score:
            logger.info("SAM3 segment_text prompt=%r score=%.3f (kept)", prompt, score)
            out.append((prompt, seg_resp["masks"][0], score))
        else:
            logger.info(
                "SAM3 segment_text prompt=%r score=%.3f below threshold",
                prompt, score,
            )
    return out


def _obb_in_workspace(obb: OrientedBoundingBox) -> bool:
    cx, cy, cz = obb["center"]["x"], obb["center"]["y"], obb["center"]["z"]
    return (
        _WORKSPACE_X[0] <= cx <= _WORKSPACE_X[1]
        and _WORKSPACE_Y[0] <= cy <= _WORKSPACE_Y[1]
        and _WORKSPACE_Z[0] <= cz <= _WORKSPACE_Z[1]
    )


def _xy_offset_from_container(
    zone: OrientedBoundingBox, container: OrientedBoundingBox | None,
) -> float:
    """Return the XY distance from zone center to container center.

    Used to pick between Path A and Path B candidates: the placement
    zone should be at or inside the container's XY footprint, so the
    candidate with the smaller XY offset is the more plausible one.
    Returns 0.0 when no container is supplied (no spatial prior).
    """
    if container is None:
        return 0.0
    dx = zone["center"]["x"] - container["center"]["x"]
    dy = zone["center"]["y"] - container["center"]["y"]
    return (dx * dx + dy * dy) ** 0.5


def _zone_within_container(
    zone: OrientedBoundingBox,
    container: OrientedBoundingBox | None,
    margin_factor: float = 1.5,
) -> bool:
    """True when the zone center is within the container's XY footprint
    (plus a generous margin) regardless of OBB orientation.

    The container OBB's ``extent.x`` / ``extent.y`` are expressed in the
    OBB's *local* frame, so under rotation (e.g. the LIBERO caddy is
    quat(w=0.71, z=0.71), a 90° world-Z rotation), a world-Y offset
    must be compared against the caddy's *wide* local axis, not its
    narrow one. Use the OBB's diameter (max of extent.x/extent.y) as a
    rotation-invariant upper bound on the container footprint, then
    require the zone's world XY offset to fit within
    ``margin_factor * diameter``. Conservative on rectangular
    containers but eliminates false rejects on rotated caddies/shelves.
    """
    if container is None:
        return True
    dx = zone["center"]["x"] - container["center"]["x"]
    dy = zone["center"]["y"] - container["center"]["y"]
    xy_offset = (dx * dx + dy * dy) ** 0.5
    diameter = max(container["extent"]["x"], container["extent"]["y"])
    return xy_offset <= margin_factor * diameter + 0.02


def _mask_to_zone_obb(
    ctx: Any,
    cam: Any,
    mask: Mask,
    min_points: int,
    max_zone_z_extent: float,
    container_obb: OrientedBoundingBox | None,
    label: str,
) -> OrientedBoundingBox | None:
    """Project a 2D mask to a 3D zone OBB and apply sanity gates.

    Gates (any failure → return ``None``):
      * cloud must have ≥ ``min_points`` points (filters depth-NaN masks);
      * OBB Z-extent ≤ ``max_zone_z_extent`` (rejects vertical walls);
      * OBB center inside the workspace AABB (rejects nonsense projections);
      * if ``container_obb`` is supplied, OBB XY center within
        ``1.5 × container_extent`` of container center (rejects masks
        that segment a 2D region inside the caddy footprint but project
        through bad depth to background).
    """
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask, depth=cam["depth"],
        intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
    )
    points = cloud["points"]
    num_points = int(len(points["points"]))
    if num_points < min_points:
        logger.info(
            "[%s] cloud too sparse: %d < %d points",
            label, num_points, min_points,
        )
        return None

    zone = ctx.tool("geometry.filter_and_compute_obb", points=points)["obb"]

    if zone["extent"]["z"] > max_zone_z_extent:
        logger.info(
            "[%s] Z-extent %.3f m > %.3f m — likely a wall, rejecting",
            label, zone["extent"]["z"], max_zone_z_extent,
        )
        return None

    if not _obb_in_workspace(zone):
        logger.info(
            "[%s] OBB center (%.3f, %.3f, %.3f) outside workspace — rejecting",
            label, zone["center"]["x"], zone["center"]["y"], zone["center"]["z"],
        )
        return None

    if not _zone_within_container(zone, container_obb):
        logger.info(
            "[%s] OBB center (%.3f, %.3f) too far from container center "
            "(%.3f, %.3f) given extent (%.3f, %.3f) — rejecting",
            label, zone["center"]["x"], zone["center"]["y"],
            container_obb["center"]["x"] if container_obb else 0.0,
            container_obb["center"]["y"] if container_obb else 0.0,
            container_obb["extent"]["x"] if container_obb else 0.0,
            container_obb["extent"]["y"] if container_obb else 0.0,
        )
        return None

    # Container-top consistency gate. A placement zone you set an object
    # ONTO must sit AT its container's top surface, not float above it.
    # A noisy SAM3 text mask (e.g. "the surface of the stove" grabbing
    # raised burner grates / the back panel) plus depth error projects
    # to an OBB whose center floats well above the container top; trusting
    # it pushes the drop pose high and the object misses the surface
    # (observed: zone center z=0.111 vs stove top ~0; goal_predicate
    # failure). The threshold is RELATIVE to the perceived container, so
    # this generalizes (a high "on top of the cabinet" zone is fine —
    # its container OBB top is correspondingly high). On rejection,
    # compute_drop_pose falls back to the clean container-top path driven
    # by the cleanly-perceived container OBB.
    if container_obb is not None:
        container_top = container_obb["center"]["z"] + container_obb["extent"]["z"]
        if zone["center"]["z"] > container_top + 0.05:
            logger.info(
                "[%s] zone center z=%.3f floats >5 cm above container top "
                "z=%.3f — mis-projected mask, rejecting (fall back to "
                "container-top drop)",
                label, zone["center"]["z"], container_top,
            )
            return None

    return zone


def run(
    ctx: NodeContext,
    placement_description: str | None = None,
    container_obb: OrientedBoundingBox | None = None,
    container_mask: Mask | None = None,
    dino_prompt: str | None = None,
    min_score: float = 0.3,
    min_points: int = 30,
    max_zone_z_extent: float = 0.20,
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
) -> Output:
    if not placement_description:
        return {"placement_zone_obb": None}

    obs = ctx.tool("robot.get_observation")
    cameras = list(obs["cameras"])

    cam = next(
        (c for c in cameras if c.get("name") and "eye_in_hand" not in c["name"]),
        cameras[0] if cameras else None,
    )
    if cam is None:
        logger.warning("No cameras available for placement zone perception")
        return {"placement_zone_obb": None}

    # Run BOTH paths, build a candidate list of (label, zone_obb,
    # mask_score, xy_offset_from_container). The candidate with the
    # smaller XY offset wins — the placement zone is a sub-region of
    # (or adjacent to) the container, so candidates that drift far in
    # XY are almost certainly mis-projections.
    candidates: list[tuple[str, OrientedBoundingBox, float, float]] = []

    a_mask, a_score = _try_dino_vlm_letter(
        ctx, cam, placement_description,
        dino_prompt or _DEFAULT_DINO_PROMPT,
        box_threshold, text_threshold, min_score,
        container_mask,
    )
    if a_mask is not None:
        a_obb = _mask_to_zone_obb(
            ctx, cam, a_mask, min_points, max_zone_z_extent,
            container_obb, label="dino+vlm-letter",
        )
        if a_obb is not None:
            candidates.append((
                "dino+vlm-letter", a_obb, a_score,
                _xy_offset_from_container(a_obb, container_obb),
            ))

    text_candidates = _try_segment_text(
        ctx, cam, placement_description, min_score,
    )
    for prompt, b_mask, b_score in text_candidates:
        b_obb = _mask_to_zone_obb(
            ctx, cam, b_mask, min_points, max_zone_z_extent,
            container_obb, label=f"sam3-text:{prompt!r}",
        )
        if b_obb is not None:
            candidates.append((
                f"sam3-text:{prompt}", b_obb, b_score,
                _xy_offset_from_container(b_obb, container_obb),
            ))

    if not candidates:
        logger.warning(
            "No acceptable zone for placement description %r — falling back to bare container",
            placement_description,
        )
        return {"placement_zone_obb": None}

    # Pick the best candidate. Sorting purely by XY offset is wrong when
    # two candidates are within perception noise of each other in XY but
    # differ a lot in mask confidence: e.g. observed
    #   sam3-text 'the surface of the stove'  score=0.551  xy=0.084
    #   sam3-text 'surface of the stove'      score=0.793  xy=0.087
    # — a 3 mm XY difference should NOT pick the much noisier 0.551 mask
    # (its OBB then floats above the surface and the drop misses). Bucket
    # XY offset into ~5 cm bins (below that is projection/segmentation
    # noise), then prefer the higher mask_score within a bin, then the
    # smaller offset. When no container_obb was supplied all offsets are
    # 0.0 → one bucket → highest mask_score wins (Path A still first on a
    # true tie by construction).
    candidates.sort(key=lambda c: (round(c[3] / 0.05), -c[2], c[3]))
    chosen_path, zone_obb, zone_score, xy_offset = candidates[0]

    logger.info(
        "Placement zone OBB chosen path=%s (xy_offset=%.3f m, mask_score=%.3f, "
        "desc=%r) center=(%.3f, %.3f, %.3f) extent=(%.3f, %.3f, %.3f); "
        "candidates=%s",
        chosen_path, xy_offset, zone_score, placement_description,
        zone_obb["center"]["x"], zone_obb["center"]["y"], zone_obb["center"]["z"],
        zone_obb["extent"]["x"], zone_obb["extent"]["y"], zone_obb["extent"]["z"],
        [(c[0], f"{c[3]:.3f}") for c in candidates],
    )
    return {"placement_zone_obb": zone_obb}
