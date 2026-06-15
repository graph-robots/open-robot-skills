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
from gap_core.types import BoundingBox2D, CameraFrame, Mask, PointCloud, pose_to_matrix

logger = logging.getLogger(__name__)

_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
_INTERSECTION_DIST = 0.01
_MIN_INTERSECTION = 1

# --- Skill-level cache ------------------------------------------------------
# Caches the full perceive output (found/cloud/mask/score) keyed on the
# input camera frames + args. A hit short-circuits the entire
# DINO+tournament+SAM path. Bump _CACHE_VERSION when the algorithm changes
# meaningfully.
# v4: the vlm bundle moved to deterministic temperature-0 decoding and an
# explicit YES/NO-first verify elicitation — v3 entries hold picks made
# under nondeterministic sampling and a verify gate that mislabeled
# affirmative prose as "no".
# v5: robot-point exclusion in the cloud path.
_CACHE_VERSION = "6"  # v6: wrist-support cloud fusion on the verified path
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
    h.update(b"|rgb|")
    h.update(np.ascontiguousarray(cam["rgb"]).tobytes())
    h.update(b"|depth|")
    h.update(np.ascontiguousarray(cam["depth"]).tobytes())
    h.update(b"|intr|")
    h.update(np.ascontiguousarray(cam["intrinsics"]).tobytes())
    h.update(b"|pose|")
    h.update(repr(cam["pose"]).encode("utf-8"))
    return h.digest()


def _make_cache_key(cameras: list[CameraFrame], args: dict) -> str:
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}".encode())
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


def _cache_load(key: str) -> Output | None:
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


def _cache_store(key: str, out: Output) -> None:
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
        x1 = max(a["x1"], b["x1"])
        y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"])
        y2 = min(a["y2"], b["y2"])
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
) -> bool:
    """Focused yes/no on the chosen crop — 'is this close-up actually a
    <object_name>?'. The gate signal for the `safe` wrist-fallback
    policy. `default` is returned on a missing box / infra error:
      - exterior pick -> default True  (verify error must NOT trigger a
        spurious wrist fallback; keep the exterior pick = zero-regression)
      - wrist pick     -> default False (never accept an unverified wrist)
    """
    if box is None:
        return default
    crop = _crop_region(rgb_np, box, pad=0.35)
    if crop is None:
        return default
    img = _upscale_letterbox(crop)
    q = f'Is the main object in this close-up a "{object_name}"?'
    if object_description:
        # Anchor the judgment on the caller-provided appearance hints —
        # without the explicit "judge by shape/colors" instruction the VLM
        # falls back to its semantic prior for the category name and
        # rejects correct picks whose upscaled low-res crop reads as a
        # generic object (e.g. the LIBERO cream-cheese box scored "a small
        # book" and the false NO forced the degraded wrist fallback).
        q += (f" It should look like: {object_description}."
              " Judge by the described shape and colors; printed text may"
              " be illegible at this resolution.")
    try:
        resp = ctx.tool("vlm.query_yes_no", prompt=q, image=img)
        logger.info(
            "verify_pick('%s'): answer=%s text=%r",
            object_name, resp["answer"], str(resp["text"])[:200])
        return bool(resp["answer"])
    except Exception as first_exc:
        logger.warning(
            "verify_pick: vlm.query_yes_no failed (perceiving-objects), "
            "falling back to vlm.query: %s", first_exc)
        try:
            resp = ctx.tool("vlm.query",
                            prompt=q + " Answer YES or NO.", image=img)
            logger.info(
                "verify_pick('%s') [query fallback]: text=%r",
                object_name, str(resp["text"])[:200])
            return resp["text"].strip().upper().startswith("Y")
        except Exception as e:
            logger.warning(
                "verify_pick failed on BOTH vlm.query_yes_no and vlm.query "
                "(perceiving-objects) — returning default=%s, the safe gate "
                "will NOT get a real verification signal: %s", default, e)
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

    # Strip robot-body points: when the target sits against the robot base
    # the segmentation mask bleeds onto robot pixels and the merged cloud
    # yields a wildly oversized OBB (e.g. basket + robot column fused into
    # a half-metre blob centred nowhere). FK-sphere exclusion removes them;
    # non-7-DOF arms pass through unchanged inside the tool.
    try:
        obs = ctx.tool("robot.get_observation")
        joints = obs["arms"][0]["joint_state"]
        before = len(cloud["points"])
        cloud = ctx.tool(
            "geometry.exclude_robot_points",
            points=cloud, joint_positions=joints,
        )["points"]
        removed = before - len(cloud["points"])
        if removed:
            logger.info(
                "perceive '%s' cam '%s': excluded %d robot-near points "
                "(%d remain)", object_name, cam["name"], removed,
                len(cloud["points"]),
            )
    except Exception as exc:
        logger.warning("robot-point exclusion skipped: %s", exc)

    num_points = len(cloud["points"])
    if num_points < min_points:
        logger.debug(
            "Too few points (%d < %d) for '%s' in cam '%s'",
            num_points, min_points, object_name, cam["name"],
        )
        return None

    return _CamResult(cloud=cloud, mask=seg_mask, score=seg_score,
                      box=sel_box)


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


def _cloud_pts(cloud: PointCloud) -> np.ndarray:
    pts = np.asarray(cloud["points"], dtype=np.float64)
    return pts.reshape(-1, 3) if pts.size else np.zeros((0, 3))


def _clouds_intersect(anchor_pts: np.ndarray, cloud: PointCloud) -> bool:
    """Dev-era multiview fusion guard: the candidate cloud must share at
    least ``_MIN_INTERSECTION`` points within ``_INTERSECTION_DIST`` of the
    anchor cloud — i.e. both views observed the same physical surface."""
    from scipy.spatial import cKDTree

    pts = _cloud_pts(cloud)
    if len(anchor_pts) == 0 or len(pts) == 0:
        return False
    dists, _ = cKDTree(anchor_pts).query(pts)
    n = int(np.sum(dists < _INTERSECTION_DIST))
    logger.info("wrist-support intersection: %d points within %.0fmm",
                n, _INTERSECTION_DIST * 1000)
    return n >= _MIN_INTERSECTION


def _segment_wrist_by_projection(
    ctx: Any,
    cam: CameraFrame,
    anchor_pts: np.ndarray,
    min_score: float,
    min_points: int,
) -> _CamResult | None:
    """Cross-view segmentation: seed ``sam3.segment_box`` on the wrist
    frame with the 2-98 percentile bbox of the verified exterior cloud
    projected into the wrist camera.

    Geometry-guided (no extra VLM call): the exterior cloud IS the verified
    object, so its wrist-frame projection bounds the object's visible wrist
    pixels; SAM grows the seed box to the full visible extent (the top face
    the exterior view cannot see)."""
    if len(anchor_pts) < 10:
        return None
    try:
        K = np.asarray(cam["intrinsics"], dtype=np.float64)
        T_world_to_cam = np.linalg.inv(pose_to_matrix(cam["pose"]))
        ph = np.hstack([anchor_pts, np.ones((len(anchor_pts), 1))])
        pc = (T_world_to_cam @ ph.T).T[:, :3]
        z = pc[:, 2]
        ok = z > 1e-3
        if int(ok.sum()) < 10:
            return None
        u = K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]
        v = K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]
        H, W = np.asarray(cam["depth"]).shape
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if int(inb.sum()) < 10:
            logger.info("wrist-support: object not visible in '%s' "
                        "(%d projected px in bounds)", cam["name"], inb.sum())
            return None
        u, v = u[inb], v[inb]
        x1, x2 = np.percentile(u, [2, 98])
        y1, y2 = np.percentile(v, [2, 98])
        pad = 0.15 * max(x2 - x1, y2 - y1)
        box: BoundingBox2D = {
            "x1": float(max(0.0, x1 - pad)), "y1": float(max(0.0, y1 - pad)),
            "x2": float(min(W - 1.0, x2 + pad)), "y2": float(min(H - 1.0, y2 + pad)),
        }
        seg = ctx.tool("sam3.segment_box", image=cam["rgb"], box=box)
        if not seg["masks"] or not seg["scores"] or seg["scores"][0] < min_score:
            return None
        mask, score = seg["masks"][0], seg["scores"][0]
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=mask, depth=cam["depth"],
            intrinsics=cam["intrinsics"], camera_pose=cam["pose"],
        )["points"]
        try:
            obs = ctx.tool("robot.get_observation")
            cloud = ctx.tool(
                "geometry.exclude_robot_points",
                points=cloud, joint_positions=obs["arms"][0]["joint_state"],
            )["points"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("wrist-support robot-point exclusion skipped: %s", exc)
        if len(cloud["points"]) < min_points:
            return None
        return _CamResult(cloud=cloud, mask=mask, score=score)
    except Exception as exc:  # noqa: BLE001
        logger.warning("wrist-support projection seeding failed: %s", exc)
        return None


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

    On the verified-exterior path the wrist views still contribute CLOUD
    GEOMETRY (never identity): each wrist cloud of the same object —
    validated by the dev-era multiview intersection guard — is fused into
    the output cloud so the downstream OBB sees the top face / far side a
    single front view misses. Without it, tall (13-15 cm) bottles/cartons
    perceive as a thin front-face sliver whose centre is biased toward
    the camera by half the object depth (measured 12-13 mm on the LIBERO
    grocery suite), and the resulting off-centre pinch slips during
    transport. The returned mask/score remain the exterior pick's.

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
        })
        hit = _cache_load(cache_key)
        if hit is not None:
            logger.info("perceiving-objects cache HIT key=%s found=%s score=%.3f",
                        cache_key[:12], hit["found"], hit["score"])
            return hit
        logger.info("perceiving-objects cache MISS key=%s", cache_key[:12])

    out = _run_uncached(
        ctx, cameras, object_name, text_prompts, min_points, min_score,
        use_multiview, box_threshold, text_threshold, dino_prompt,
        object_description,
    )
    if cache_key is not None and out["found"]:
        _cache_store(cache_key, out)
    return out


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

    def _finish(results: list[_CamResult],
                anchor: _CamResult | None = None) -> Output:
        if not results:
            return {"found": False, "cloud": _empty_cloud(),
                    "mask": _empty_mask(), "score": 0.0}
        cloud = (_merge_multiview(results)
                 if use_multiview and len(results) > 1
                 else results[0].cloud)
        # When an anchor result is given (the verified exterior pick), its
        # mask/score stay authoritative — wrist masks live in a moving
        # camera frame and must never reach downstream consumers that
        # project masks through the static exterior camera (build_world).
        best = anchor if anchor is not None else max(results,
                                                     key=lambda r: r.score)
        return {"found": True, "cloud": cloud,
                "mask": best.mask, "score": best.score}

    def _collect_wrist_support(
        wrist_cams_, ext_results: list[_CamResult],
    ) -> list[_CamResult]:
        """Dev-era multiview fusion, anchored on the verified exterior
        pick: gather wrist-view clouds of the SAME object so the fused
        cloud covers the top face / far side the exterior view cannot see
        (a single front view yields a sliver OBB whose centre is biased
        toward the camera by half the object depth — measured 12-13 mm on
        LIBERO tall bottles/cartons, enough to make the fingers pinch the
        edge and slip). A wrist result is fused ONLY when its cloud
        intersects the exterior cloud (the dev multiview guard), so a
        mis-segmented wrist view can never displace the verified pick."""
        anchor_pts = np.concatenate(
            [_cloud_pts(r.cloud) for r in ext_results], axis=0,
        ) if ext_results else np.zeros((0, 3))
        support: list[_CamResult] = []
        for wcam in wrist_cams_:
            try:
                # Historical wrist behavior first: segment_text on the frame.
                r = None
                try:
                    r = _perceive_single_camera(
                        ctx, wcam, object_name, text_prompts, min_score,
                        min_points, box_threshold, text_threshold,
                        dino_prompt, object_description, run_identify=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "wrist-support: segment_text path failed on '%s': %s",
                        wcam.get("name"), exc)
                if r is None or not _clouds_intersect(anchor_pts, r.cloud):
                    # Geometry-guided fallback: seed SAM with the exterior
                    # cloud's projection into the wrist camera.
                    r = _segment_wrist_by_projection(
                        ctx, wcam, anchor_pts, min_score, min_points)
                if r is None:
                    continue
                if _clouds_intersect(anchor_pts, r.cloud):
                    logger.info(
                        "wrist-support: fusing wrist view '%s' "
                        "(%d points, score=%.3f)", wcam["name"],
                        len(r.cloud["points"]), r.score)
                    support.append(r)
            except Exception as exc:  # noqa: BLE001 — support is best-effort
                logger.warning(
                    "wrist-support: skipping wrist view '%s': %s",
                    wcam.get("name"), exc)
        return support

    wrist_cams = [c for c in cameras if _is_wrist(c)]
    ext_cams = [c for c in cameras if not _is_wrist(c)]

    # Non-LIBERO / single-view platforms: keep the historical per-camera
    # behavior unchanged (identify on non-wrist, segment_text on wrist).
    if not wrist_cams or not ext_cams:
        return _finish([r for _, r in _collect(cameras, None)])

    # --- safe gate ---
    ext = _collect(ext_cams, True)
    if ext:
        bc, br = max(ext, key=lambda cr: cr[1].score)
        if _verify_pick(ctx, bc["rgb"], br.box,
                        object_name, object_description, default=True):
            logger.info(
                "safe gate: exterior pick verified for '%s' -> exterior",
                object_name)
            ext_results = [r for _, r in ext]
            support = _collect_wrist_support(wrist_cams, ext_results)
            return _finish(ext_results + support, anchor=br)

    # Exterior pick rejected (or none) -> gated wrist fallback. Loud on
    # purpose: a wrist-only result is a single top-down view whose cloud
    # covers just the visible top face, so the downstream OBB loses its
    # height — if this fires on objects the exterior view identified
    # correctly, pass `object_description` (shape/appearance hints) so the
    # close-up verify can recognize the rendered asset.
    logger.warning(
        "safe gate: exterior pick rejected by close-up verify for '%s' "
        "(object_description=%r) -> trying wrist fallback",
        object_name, object_description)
    wr = _collect(wrist_cams, True)
    if wr:
        wc, wrr = max(wr, key=lambda cr: cr[1].score)
        if _verify_pick(ctx, wc["rgb"], wrr.box,
                        object_name, object_description, default=False):
            logger.info("safe gate: wrist pick verified -> wrist")
            return _finish([wrr])

    # Neither side verified: conservatively keep the exterior result if
    # any, else fall back to the legacy exterior segment_text net (never
    # fuse an unverified wrist — that is the regressing behavior).
    fallback = [r for _, r in ext] or [r for _, r in _collect(ext_cams, None)]
    logger.info("safe gate: no verified pick; using conservative exterior "
                "fallback (%d result(s))", len(fallback))
    return _finish(fallback)
