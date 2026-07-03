"""DINO+VLM multiview perception: GDINO broad detect → crop+tournament → SAM3 → 3D fusion.

Canonical script for the ``perceiving-objects`` skill bundle. The generated
subgraph references this as a ``type: script`` state. Internally it calls:

- ``grounding-dino.detect`` (broad ``object.`` text prompt)
- ``vlm.query`` in a single-elimination **pairwise tournament** to pick
  which DINO box matches the target
- ``sam3.segment_box`` to segment the chosen box
- ``geometry.mask_to_world_points`` to fuse depth into a world-frame cloud

Why a tournament instead of a one-shot box pick: the targets here are
small (20-40 px in an 800x512 frame), so a Set-of-Marks overlay asking
the VLM for a single letter out of N is unreliable — it cannot resolve
the boxes. Instead each detection is cropped and upscaled, and the
target is found via binary "A or B?" comparisons of upscaled crop pairs.
VLMs are dramatically more reliable 2-way than N-way (cf. CropVLM,
arXiv:2511.19820). On the LIBERO-PosVar object-ID study this lifted
accuracy from ~30% (Set-of-Marks) to 97%.

The VLM prompt template lives in ``prompts/vlm_pairwise.md`` and is
loaded at call time via :func:`gap.skills.load_prompt`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from gap import NodeContext
from gap.skills import load_prompt
from gap_core.types import BoundingBox2D, CameraFrame, Mask, PointCloud

logger = logging.getLogger(__name__)

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_INTERSECTION_DIST = 0.01
_MIN_INTERSECTION = 1

# --- Skill-level cache ------------------------------------------------------
# Caches the full perceive output (found/cloud/mask/score) keyed on the
# input camera frames + args. A hit short-circuits the entire
# DINO+tournament+SAM path. Bump _CACHE_VERSION when the algorithm changes
# meaningfully.
_CACHE_VERSION = "9"  # v9: strict (binding-description) verify phrasing in reject_unverified mode
_CACHE_ENABLED = os.environ.get("GAP_PERCEPTION_CACHE", "1") == "1"
# Default to checkout-local .llm_cache/perceiving-objects/ so cache is
# per-checkout and easy to wipe. Override with GAP_PERCEPTION_CACHE_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = Path(os.environ.get(
    "GAP_PERCEPTION_CACHE_DIR",
    str(_REPO_ROOT / ".llm_cache" / "perceiving-objects"),
))


def _hash_camera(cam: CameraFrame) -> bytes:
    h = hashlib.sha256()
    h.update((cam["name"] or "").encode("utf-8"))
    h.update(b"|rgb|");   h.update(np.ascontiguousarray(cam["rgb"]).tobytes())
    h.update(b"|depth|"); h.update(np.ascontiguousarray(cam["depth"]).tobytes())
    h.update(b"|intr|");  h.update(np.ascontiguousarray(cam["intrinsics"]).tobytes())
    h.update(b"|pose|");  h.update(repr(cam["pose"]).encode("utf-8"))
    return h.digest()


def _make_cache_key(cameras: list[CameraFrame], args: dict) -> str:
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}".encode("utf-8"))
    h.update(b"|model|")
    # The vlm bundle's provider/model env config is part of the key so
    # swapping models doesn't return stale cached picks.
    h.update(os.environ.get("GAP_VLM_PROVIDER", "").encode("utf-8"))
    h.update(b"/")
    h.update(os.environ.get("GAP_VLM_MODEL", "").encode("utf-8"))
    for cam in cameras:
        h.update(b"|cam|")
        h.update(_hash_camera(cam))
    h.update(b"|args|")
    h.update(repr(sorted(args.items())).encode("utf-8"))
    return h.hexdigest()


def _cache_load(key: str) -> "Output | None":
    p = _CACHE_DIR / f"{key}.pkl"
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            d = pickle.load(f)
        return {"found": d["found"], "cloud": d["cloud"],
                "mask": d["mask"], "score": d["score"]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("perceiving-objects cache: load failed for %s: %s",
                       key[:12], exc)
        return None


def _cache_store(key: str, out: "Output") -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        blob = pickle.dumps(
            {
                "found": out["found"],
                "cloud": out["cloud"],
                "mask":  out["mask"],
                "score": out["score"],
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        with tempfile.NamedTemporaryFile(
            dir=str(_CACHE_DIR), delete=False, suffix=".tmp",
        ) as tf:
            tf.write(blob)
            tmp = tf.name
        os.replace(tmp, _CACHE_DIR / f"{key}.pkl")
    except Exception as exc:  # noqa: BLE001
        logger.warning("perceiving-objects cache: store failed for %s: %s",
                       key[:12], exc)


def _empty_cloud() -> PointCloud:
    return {"points": np.zeros((0, 3), dtype=np.float32)}


def _empty_mask() -> Mask:
    return np.zeros((0, 0), dtype=np.uint8)


_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255),
]


def _box_xyxy(box: BoundingBox2D) -> tuple[int, int, int, int]:
    return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])


# Size guard for containment-NMS. A genuine sub-region fragment (logo,
# label, illustration) is small in absolute pixels; a legitimate object
# box that merely happens to sit inside a larger over-capture box is
# not. Only a contained box whose longest side is below this many pixels
# is dropped -- so a tight basket box contained in a basket+table
# over-capture box is KEPT. Validated by the 800-frame perception_study
# sweep (2026-05-21): the size-bounded variant `tourn_nmsz` matched the
# unbounded one on the orange-juice partial-mask fix (10/40 -> 0) AND
# eliminated the posvar/all basket backfire (7 degraded container masks
# -> 0). See [[project_perception_mask_quality_study]].
_NMS_FRAGMENT_PX = 80


def _drop_contained_detections(
    detections: list, containment_thresh: float = 0.7,
    max_small_side: int = _NMS_FRAGMENT_PX,
) -> list:
    """Size-bounded containment / Intersection-over-Smaller NMS.

    GroundingDINO with a generic prompt like ``"object."`` routinely
    emits *both* a whole-object box and a sub-region box (an iconic
    label, logo, or part of the object). The downstream VLM tournament
    sees the sub-region crop as a higher-signal "looks like X" than the
    whole-object crop -- and picks it -- which produces a tiny mask and
    a perceived 5cm-tall object instead of the real 14cm carton.

    Drop detection ``A`` whenever a *larger* detection ``B`` exists with
    ``inter(A, B) / area(A) >= containment_thresh`` AND ``A`` is itself
    small (longest side < ``max_small_side`` px) -- i.e. a genuine
    logo/label fragment. A legitimate large object box contained inside
    an over-capture box is KEPT: the earlier unbounded variant dropped
    the tight basket box in favour of a basket+table box, which skewed
    the container OBB and the place pose. Standard IoU NMS doesn't catch
    the fragment case (a small nested box has low IoU with its big
    parent); IoS / containment is the appropriate metric.
    """
    def _area(b: BoundingBox2D) -> float:
        return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

    def _inter(a: BoundingBox2D, b: BoundingBox2D) -> float:
        x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    kept: list = []
    dropped: list[tuple[int, int]] = []
    for i, di in enumerate(detections):
        ai = _area(di["box"])
        if ai <= 0:
            continue
        contained_in_j = -1
        for j, dj in enumerate(detections):
            if j == i:
                continue
            aj = _area(dj["box"])
            if aj <= ai:
                continue
            if _inter(di["box"], dj["box"]) / ai >= containment_thresh:
                contained_in_j = j
                break
        # Size guard: only drop a *small* contained box (a true fragment).
        # A large legitimate box is kept even when it is contained.
        if contained_in_j >= 0:
            longest = max(di["box"]["x2"] - di["box"]["x1"],
                          di["box"]["y2"] - di["box"]["y1"])
            if longest >= max_small_side:
                contained_in_j = -1
        if contained_in_j < 0:
            kept.append(di)
        else:
            dropped.append((i, contained_in_j))
    if dropped:
        logger.info(
            "containment-NMS dropped %d sub-region boxes (kept %d/%d): %s",
            len(dropped), len(kept), len(detections),
            ", ".join(f"{i}<<{j}" for i, j in dropped),
        )
    return kept


def _crop_region(rgb: np.ndarray, box: BoundingBox2D, pad: float = 0.30):
    """Crop `box` from `rgb` with `pad` fraction of box size on each side,
    clamped to the image. Returns the crop (or None if degenerate)."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = _box_xyxy(box)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    px, py = int(pad * bw), int(pad * bh)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return rgb[cy1:cy2, cx1:cx2].copy()


def _upscale_letterbox(crop: np.ndarray, size: int = 384) -> np.ndarray:
    """Resize so the longest side == size, letterbox onto neutral gray.
    Upscaling the tiny detections is the whole point."""
    import cv2

    h, w = crop.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    r = cv2.resize(crop, (nw, nh), interpolation=interp)
    canvas = np.full((size, size, 3), 128, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas


def _make_pair_sheet(crop_a: np.ndarray, crop_b: np.ndarray,
                     tile: int = 384) -> np.ndarray:
    """Two upscaled crops side by side with A / B colored headers."""
    import cv2

    pad = 6
    cell = tile + 2 * pad
    sheet = np.full((cell, 2 * cell, 3), 255, np.uint8)
    for i, cr in enumerate((crop_a, crop_b)):
        t = _upscale_letterbox(cr, tile)
        cv2.rectangle(t, (0, 0), (tile - 1, 34), _COLORS[i], -1)
        cv2.putText(t, _LABELS[i], (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 3, cv2.LINE_AA)
        x = i * cell + pad
        sheet[pad:pad + tile, x:x + tile] = t
    return sheet


def _tournament(
    ctx: Any,
    rgb_np: np.ndarray,
    detections: list,
    object_name: str,
    object_description: str,
) -> int:
    """Single-elimination pairwise binary discrimination over detection
    crops. Returns the winning index into `detections`, or -1."""
    crops: list[np.ndarray] = []
    keep: list[int] = []
    for i, det in enumerate(detections[:len(_LABELS)]):
        cr = _crop_region(rgb_np, det["box"])
        if cr is not None:
            crops.append(cr)
            keep.append(i)
    if not crops:
        return -1
    if len(crops) == 1:
        return keep[0]

    bracket = list(range(len(crops)))
    while len(bracket) > 1:
        nxt: list[int] = []
        for j in range(0, len(bracket), 2):
            if j + 1 >= len(bracket):
                nxt.append(bracket[j])           # bye
                continue
            a, b = bracket[j], bracket[j + 1]
            sheet = _make_pair_sheet(crops[a], crops[b])
            prompt = load_prompt(
                __package__, "vlm_pairwise",
                object_name=object_name,
                object_description=object_description,
            )
            try:
                resp = ctx.tool("vlm.query", prompt=prompt, image=sheet)
                win = a if _parse_letter(resp["text"], 2) == 0 else b
            except Exception as e:
                logger.warning(
                    "VLM pairwise compare failed (perceiving-objects): %s", e)
                win = a
            nxt.append(win)
        bracket = nxt
    return keep[bracket[0]]


@dataclass
class _CamResult:
    cloud: Any
    mask: Any
    score: float
    box: Any = None          # selected GDINO box (None on segment_text path)


def _verify_pick(
    ctx: Any,
    rgb_np: np.ndarray,
    box: Any,
    object_name: str,
    object_description: str,
    default: bool,
    strict: bool = False,
) -> bool:
    """Focused yes/no on the chosen crop — 'is this close-up actually a
    <object_name>?'. The gate signal for the `safe` wrist-fallback
    policy. `default` is returned on a missing box / infra error:
      - exterior pick -> default True  (verify error must NOT trigger a
        spurious wrist fallback; keep the exterior pick = zero-regression)
      - wrist pick     -> default False (never accept an unverified wrist)

    ``strict`` (used with reject_unverified — restricted item sets): the
    description is BINDING, exclusions included. The default phrasing
    ("It should look like: …") is advisory — a milk-carton crop passes
    'Is this a "object"? … except the milk' because the VLM answers the
    object-ness question and ignores the subordinate exception clause.
    """
    if box is None:
        return default
    crop = _crop_region(rgb_np, box, pad=0.35)
    if crop is None:
        return default
    img = _upscale_letterbox(crop)
    if strict and object_description:
        q = (
            f'Does the main object in this close-up satisfy ALL of this '
            f'description: "{object_description}"? If the object is one of '
            f"the things the description excludes, answer NO."
        )
    else:
        q = f'Is the main object in this close-up a "{object_name}"?'
        if object_description:
            q += f" It should look like: {object_description}."
    try:
        resp = ctx.tool("vlm.query_yes_no", prompt=q, image=img)
        return bool(resp["answer"])
    except Exception:
        try:
            resp = ctx.tool("vlm.query",
                            prompt=q + " Answer YES or NO.", image=img)
            return resp["text"].strip().upper().startswith("Y")
        except Exception as e:
            logger.warning(
                "verify_pick failed (perceiving-objects): %s", e)
            return default


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


def _is_wrist(cam: CameraFrame) -> bool:
    return bool(cam["name"]) and "eye_in_hand" in cam["name"]


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
    run_identify: bool | None = None,
) -> _CamResult | None:
    is_wrist = _is_wrist(cam)
    # When unspecified, preserve the historical behavior (identify on
    # every non-wrist cam, segment_text-only on the wrist). The `safe`
    # gate in run() passes this explicitly.
    do_identify = (not is_wrist) if run_identify is None else run_identify

    seg_mask = None
    seg_score = 0.0
    sel_box = None

    if do_identify:
        gdino_detections: list = []
        try:
            gdino_resp = ctx.tool(
                "grounding-dino.detect",
                image=cam["rgb"], query=dino_prompt,
                box_threshold=box_threshold, text_threshold=text_threshold,
            )
            gdino_detections = list(gdino_resp["detections"])
        except Exception as e:
            logger.warning("GDINO detect failed (perceiving-objects): %s", e)

        # Containment-NMS: when DINO emits both a whole-object box and a
        # sub-region (logo / label / illustration) for the same object,
        # the VLM tournament tends to pick the higher-signal sub-region
        # crop. Drop sub-regions before the tournament so the fragment
        # can't win. See `_drop_contained_detections` for the rationale.
        if gdino_detections:
            gdino_detections = _drop_contained_detections(gdino_detections)

        if gdino_detections:
            rgb_np = cam["rgb"]
            try:
                selected_idx = _tournament(
                    ctx, rgb_np, gdino_detections,
                    object_name, object_description,
                )
                logger.info(
                    "VLM tournament selected box %d for '%s' from %d detections",
                    selected_idx, object_name,
                    min(len(gdino_detections), len(_LABELS)),
                )

                if 0 <= selected_idx < len(gdino_detections):
                    selected_det = gdino_detections[selected_idx]
                    sel_box = selected_det["box"]
                    seg_resp = ctx.tool(
                        "sam3.segment_box",
                        image=cam["rgb"], box=selected_det["box"],
                    )
                    if seg_resp["masks"] and seg_resp["scores"]:
                        seg_mask = seg_resp["masks"][0]
                        seg_score = seg_resp["scores"][0]
            except Exception as e:
                logger.warning(
                    "VLM tournament selection failed (perceiving-objects): %s", e)

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

    cloud = _extend_cloud_to_support(ctx, cloud, seg_mask, cam)

    return _CamResult(cloud=cloud, mask=seg_mask, score=seg_score,
                      box=sel_box)


# ---------------------------------------------------------------------------
# Static-container anchoring (loop reperception hygiene)
# ---------------------------------------------------------------------------
# In a pack-all loop the container is re-perceived every pass, and the
# items already placed in/on it bleed into its segmentation: the basket
# OBB center drifted +2 cm then +4 cm over one episode, dragging every
# subsequent drop toward the rim. The container is STATIC, so the first
# clean sighting's XY envelope is authoritative: later sightings drop
# cloud points outside it (+ margin). Keyed per-process like the
# decide_next_item progress guard (one episode == one PID).
_CONTAINER_WORDS = (
    "basket", "bin", "box", "container", "tote", "crate", "bowl", "tray",
    "plate",
)
_CONTAINER_ENV_MARGIN_M = 0.015


def _static_container_anchor(object_name: str, out):
    if not out.get("found"):
        return out
    name = (object_name or "").lower()
    if not any(w in name for w in _CONTAINER_WORDS):
        return out
    try:
        pts = np.asarray(out["cloud"]["points"], dtype=np.float64).reshape(-1, 3)
        if len(pts) < 30:
            return out
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:40]
        path = f"/tmp/.gap_container_cloud_{os.getpid()}_{safe}.npz"
        centroid = pts.mean(axis=0)
        cached = None
        try:
            d = np.load(path)
            cached = d["points"]
        except Exception:
            pass
        if cached is not None:
            drift = float(np.linalg.norm(
                centroid[:2] - np.asarray(cached, dtype=np.float64).mean(axis=0)[:2]
            ))
            if drift < 0.06:
                # Same (static) container: reuse the FIRST clean sighting's
                # cloud verbatim. Contents placed in/on it bleed into later
                # masks — the downstream cluster filter then locks onto a
                # wall+pile SUBSET and the OBB center walks toward the pile
                # (+2 then +4 cm over one episode), dragging every drop
                # toward the rim. Point-level filtering cannot fix a subset
                # bias; cloud reuse can, and the 6 cm drift gate still
                # refreshes if the container was actually moved.
                anchored = dict(out)
                cloud = dict(out["cloud"])
                cloud["points"] = np.asarray(cached, dtype=np.float32)
                cloud.pop("colors", None)
                anchored["cloud"] = cloud
                logger.info(
                    "static-container anchor '%s': reusing first-sighting "
                    "cloud (new sighting drifted %.0f mm)",
                    object_name, drift * 1000,
                )
                return anchored
            logger.info(
                "static-container anchor '%s': %.0f mm drift exceeds the "
                "moved-container gate; refreshing the cached cloud",
                object_name, drift * 1000,
            )
        try:
            np.savez_compressed(path, points=pts.astype(np.float32))
        except Exception:
            pass
        return out
    except Exception as exc:
        logger.warning("static-container anchor skipped: %s", exc)
        return out


# ---------------------------------------------------------------------------
# Amodal bottom completion (low-profile objects)
# ---------------------------------------------------------------------------
# A top-down view of a low object (the 2 cm cream-cheese / butter boxes)
# yields a top-face-only cloud whose OBB has ~zero vertical extent; the
# grasp then closes ABOVE the object and grabs air. When the object cloud
# floats above the LOCAL support plane -- estimated from a dilated ring of
# non-object pixels around the mask -- mirror the footprint down to the
# support height so the OBB regains the object's true vertical extent.
# Tall objects whose sides are observed are a no-op (their cloud already
# reaches the support plane).
_SUPPORT_GAP_MIN_M = 0.012
_SUPPORT_RING_PX = 12
_SUPPORT_MAX_SYNTH = 2000


def _extend_cloud_to_support(ctx, cloud, seg_mask, cam):
    try:
        pts = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        if len(pts) < 20:
            return cloud
        from scipy import ndimage
        m = np.asarray(seg_mask) > 0
        ring = ndimage.binary_dilation(m, iterations=_SUPPORT_RING_PX) & ~m
        ring_resp = ctx.tool(
            "geometry.mask_to_world_points",
            mask=(ring.astype(np.uint8) * 255), depth=cam["depth"],
            intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
        )["points"]
        ring_pts = np.asarray(
            ring_resp["points"], dtype=np.float64
        ).reshape(-1, 3)
        if len(ring_pts) < 50:
            return cloud
        z = pts[:, 2]
        z_hi = float(np.percentile(z, 98.0))
        z_lo = float(z.min())
        # TOP SLAB: the one surface a tabletop camera sees completely, so
        # its footprint is an unbiased estimate of the object's
        # cross-section. A one-sided view biases the FULL cloud's centroid
        # toward the camera (a milk carton read +7 mm off-center with half
        # its true depth); the slab centroid stays true.
        slab_band = max(0.012, 0.15 * (z_hi - z_lo))
        slab = pts[z > z_hi - slab_band]
        if len(slab) < 15:
            return cloud
        centroid_bias = float(
            np.linalg.norm(slab[:, :2].mean(axis=0) - pts[:, :2].mean(axis=0))
        )
        # Complete column + unbiased -> nothing to fix, skip the ring call.
        if centroid_bias < 0.004 and (z_hi - z_lo) > 0.03:
            return cloud
        from scipy import ndimage
        m = np.asarray(seg_mask) > 0
        ring = ndimage.binary_dilation(m, iterations=_SUPPORT_RING_PX) & ~m
        ring_resp = ctx.tool(
            "geometry.mask_to_world_points",
            mask=(ring.astype(np.uint8) * 255), depth=cam["depth"],
            intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
        )["points"]
        ring_pts = np.asarray(
            ring_resp["points"], dtype=np.float64
        ).reshape(-1, 3)
        support_z = (
            float(np.percentile(ring_pts[:, 2], 20.0))
            if len(ring_pts) >= 50 else z_lo
        )
        floating = (float(np.median(z)) - support_z) >= _SUPPORT_GAP_MIN_M
        if centroid_bias < 0.004 and not floating:
            return cloud
        sel_idx = np.arange(len(slab))
        if len(sel_idx) > _SUPPORT_MAX_SYNTH:
            sel_idx = np.linspace(
                0, len(slab) - 1, _SUPPORT_MAX_SYNTH
            ).astype(int)
        base = slab[sel_idx]
        lo_lv = support_z + 0.002
        hi_lv = max(z_hi - 0.004, lo_lv + 0.001)
        n_lv = max(2, min(12, int((hi_lv - lo_lv) / 0.005)))
        # Extrude the slab footprint from the support plane up to the slab:
        # a connected prism (DBSCAN keeps it) that can only GROW the OBB
        # toward the true shape — complete clouds gain nothing.
        layers = [pts]
        for lv in np.linspace(lo_lv, hi_lv, n_lv):
            layer = base.copy()
            layer[:, 2] = lv
            layers.append(layer)
        out = dict(cloud)
        out["points"] = np.concatenate(layers, axis=0).astype(np.float32)
        if out.get("colors") is not None and len(out["colors"]) == len(pts):
            colors = np.asarray(out["colors"], dtype=np.float32).reshape(-1, 3)
            out["colors"] = np.concatenate(
                [colors] + [np.full((len(base), 3), colors.mean(axis=0),
                                    dtype=np.float32)] * n_lv,
                axis=0,
            )
        logger.info(
            "amodal completion: slab-centroid bias %.1f mm, floating=%s -> "
            "extruded %d slab points across %d levels (support_z=%.3f)",
            centroid_bias * 1000, floating, len(base), n_lv, support_z,
        )
        return out
    except Exception as exc:
        logger.warning("support-plane completion skipped: %s", exc)
        return cloud


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
    reject_unverified: bool = False,
) -> Output:
    """Execute DINO+VLM perception with the `safe` wrist-fallback gate.

    Default to the exterior (e.g. agentview) identification; consult the
    wrist (eye-in-hand) view ONLY when the exterior pick fails its own
    close-up verify AND the wrist pick passes its own. On the
    LIBERO-PosVar object-ID study (4 PosVar suites, 200 frames,
    `scripts/analyze_wrist_regression.py`) this `safe` gate was the only
    zero-regression policy: +2.5% net accuracy (94.5% -> 97.0%), 0/189
    regressions on the frames the exterior already got right — whereas
    "always verify->wrist" (-3%), "always fuse both" (-1%) and
    "wrist-only" (-20%) all regressed. The double gate is essential:
    never abandon a self-consistent exterior pick, never jump onto an
    unverified wrist.

    Skill-level caching is **on by default**: identical perceive calls
    (same cameras + args) short-circuit with the previously computed
    Output. Set ``GAP_PERCEPTION_CACHE=0`` to disable. Cache dir
    overridable with ``GAP_PERCEPTION_CACHE_DIR`` (default
    ``<open-robot-skills checkout>/.llm_cache/perceiving-objects``).
    """
    cache_key: str | None = None
    if _CACHE_ENABLED:
        cache_key = _make_cache_key(cameras, {
            "object_name": object_name,
            "text_prompts": sorted(text_prompts) if text_prompts else None,
            "min_points": min_points,
            "min_score": min_score,
            "use_multiview": use_multiview,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "dino_prompt": dino_prompt,
            "object_description": object_description,
            "reject_unverified": reject_unverified,
        })
        hit = _cache_load(cache_key)
        if hit is not None:
            logger.info("perceiving-objects cache HIT key=%s found=%s score=%.3f",
                        cache_key[:12], hit["found"], hit["score"])
            return _static_container_anchor(object_name, hit)
        logger.info("perceiving-objects cache MISS key=%s", cache_key[:12])

    out = _run_uncached(
        ctx, cameras, object_name, text_prompts, min_points, min_score,
        use_multiview, box_threshold, text_threshold, dino_prompt,
        object_description, reject_unverified,
    )
    if cache_key is not None and out["found"]:
        _cache_store(cache_key, out)
    return _static_container_anchor(object_name, out)


def _run_uncached(
    ctx: NodeContext,
    cameras: list[CameraFrame],
    object_name: str,
    text_prompts: list[str] | None,
    min_points: int,
    min_score: float,
    use_multiview: bool,
    box_threshold: float,
    text_threshold: float,
    dino_prompt: str,
    object_description: str,
    reject_unverified: bool = False,
) -> Output:
    """Existing perceive body — kept callable directly for cache-bypass paths."""
    if text_prompts is None:
        text_prompts = [object_name]

    def _collect(cams, run_identify):
        out: list[tuple[Any, _CamResult]] = []
        for cam in cams:
            r = _perceive_single_camera(
                ctx, cam, object_name, text_prompts, min_score, min_points,
                box_threshold, text_threshold, dino_prompt,
                object_description, run_identify=run_identify,
            )
            if r is not None:
                out.append((cam, r))
        return out

    def _finish(results: list[_CamResult]) -> Output:
        if not results:
            return {"found": False, "cloud": _empty_cloud(),
                    "mask": _empty_mask(), "score": 0.0}
        cloud = (_merge_multiview(results)
                 if use_multiview and len(results) > 1
                 else results[0].cloud)
        best = max(results, key=lambda r: r.score)
        return {"found": True, "cloud": cloud,
                "mask": best.mask, "score": best.score}

    wrist_cams = [c for c in cameras if _is_wrist(c)]
    ext_cams = [c for c in cameras if not _is_wrist(c)]

    # Non-LIBERO / single-view platforms: keep the historical per-camera
    # behavior unchanged (identify on non-wrist, segment_text on wrist).
    if not wrist_cams or not ext_cams:
        results = _collect(cameras, None)
        # Terminal-reject gate (opt-in via reject_unverified). Run the chosen
        # pick through the close-up verify using the same object_description
        # that excludes the basket ("never the wicker basket..."). Once every
        # item is packed and teleported away, the only candidate left is the
        # basket itself — it fails the verify, so we report found=False and
        # the caller's loop exits cleanly on "none" instead of grasping the
        # basket until the iteration cap. default=True keeps a real pick on
        # any VLM/infra error, so this never causes a spurious early stop.
        if reject_unverified and results:
            bc, br = max(results, key=lambda cr: cr[1].score)
            if not _verify_pick(ctx, bc["rgb"], br.box,
                                object_name, object_description, default=True,
                                strict=True):
                logger.info(
                    "single-view terminal-reject: pick failed '%s' verify "
                    "-> found=False (table clear)", object_name)
                return {"found": False, "cloud": _empty_cloud(),
                        "mask": _empty_mask(), "score": 0.0}
        return _finish([r for _, r in results])

    # --- safe gate ---
    ext = _collect(ext_cams, True)
    if ext:
        bc, br = max(ext, key=lambda cr: cr[1].score)
        if _verify_pick(ctx, bc["rgb"], br.box,
                        object_name, object_description, default=True,
                        strict=reject_unverified):
            logger.info(
                "safe gate: exterior pick verified for '%s' -> exterior",
                object_name)
            return _finish([r for _, r in ext])

    # Exterior pick rejected (or none) -> gated wrist fallback.
    logger.info(
        "safe gate: exterior pick rejected for '%s' -> trying wrist",
        object_name)
    wr = _collect(wrist_cams, True)
    if wr:
        wc, wrr = max(wr, key=lambda cr: cr[1].score)
        if _verify_pick(ctx, wc["rgb"], wrr.box,
                        object_name, object_description, default=False,
                        strict=reject_unverified):
            logger.info("safe gate: wrist pick verified -> wrist")
            return _finish([wrr])

    # Neither side verified. Two regimes:
    #   * broad queries ("grocery item" — any remaining item is a valid
    #     target): favor RECALL — keep the exterior result even unverified
    #     (never fuse an unverified wrist — that is the regressing
    #     behavior). A verify false-reject would otherwise end a pack-all
    #     loop with items still on the table.
    #   * narrow queries (reject_unverified=True — the loop targets specific
    #     item names): favor PRECISION — report found=False. Keeping an
    #     unverified pick here makes the loop grasp non-target items (a
    #     "pack the milk and the tomato sauce" loop was observed packing
    #     all six objects through this fallback).
    if reject_unverified:
        logger.info(
            "safe gate: no verified pick and reject_unverified=True "
            "-> found=False (no target item present)")
        return {"found": False, "cloud": _empty_cloud(),
                "mask": _empty_mask(), "score": 0.0}
    fallback = [r for _, r in ext] or [r for _, r in _collect(ext_cams, None)]
    logger.info("safe gate: no verified pick; using conservative exterior "
                "fallback (%d result(s))", len(fallback))
    return _finish(fallback)
