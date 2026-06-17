"""sam3 tool bundle — text/point/box segmentation + stateful video tracking.

In-process SAM3 + SAM3-tracker model calls (autocast contexts, prompt
handling, mask post-processing, score sorting, drift detection);
functions take/return numpy arrays and :mod:`gap.types` TypedDicts.

Models load lazily on first call (module-level singletons), so importing
this module never pulls torch / the upstream ``sam3`` package. Device comes
from the ``GAP_SAM3_DEVICE`` env var (default ``cuda``).

Tracker sessions are module-level state addressed by an opaque
``tracker_id``. Stale sessions are evicted lazily on tracker calls after
a TTL.
"""

from __future__ import annotations

import collections
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D, Mask

logger = logging.getLogger(__name__)

#: Torch device for both the image model and the video tracker.
_DEVICE = os.environ.get("GAP_SAM3_DEVICE", "cuda")

#: Hugging Face repo id for SAM3 weights (image + video predictors share
#: the same checkpoint).  Must match upstream sam3/model_builder.py
#: ``SAM3_MODEL_ID``; kept here so ``prefetch()`` doesn't have to import
#: torch / sam3 just to read the constant.
_SAM3_HF_REPO = "facebook/sam3"


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check``.

    Mirrors grounding-dino's check: looks for the SAM3 ``config.json``
    in the HF cache without importing torch or sam3. Returns ``None``
    when ``huggingface_hub`` isn't available ("unknown").
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        result = try_to_load_from_cache(_SAM3_HF_REPO, "config.json")
    except Exception:
        return None
    return isinstance(result, str)


def prefetch() -> None:
    """Snapshot-download the SAM3 weights into the HF cache.

    Called by ``gap skills check --download``. The SAM3 checkpoint is a
    gated repo on Hugging Face — users must accept the model card terms
    AND have a valid ``HF_TOKEN`` (or be logged in via ``huggingface-cli
    login``). If either is missing, this raises with the same 401/403
    HfHubHTTPError the lazy first-call path would raise, so the user
    finds out at install time instead of mid-run.

    Idempotent: a re-run against an already-cached snapshot is a
    near-no-op (HF revision check + symlink refresh).
    """
    from huggingface_hub import snapshot_download

    logger.info("[sam3] prefetching weights for %s ...", _SAM3_HF_REPO)
    snapshot_download(repo_id=_SAM3_HF_REPO, repo_type="model")
    logger.info("[sam3] prefetch complete (cached at HF default)")

# Cap how many detections segment_text returns by default.  The model emits
# one mask per detected instance — on cluttered scenes that can be ~200, each
# ~1 MB at 1280x720 (uint8 0/255).  Most callers consume only ``masks[0]``.
_SEGMENT_TEXT_TOP_K = 5

# Tracker session TTL (seconds since last touch) before lazy eviction.
_DEFAULT_TTL_S = 120.0

# Drift detection. A mask whose area jumps > _AREA_JUMP_RATIO times
# the running median, or arrives with confidence < _DRIFT_LOW_SCORE, is
# treated as drift. The session keeps its previous good mask and reports
# object_present=True with confidence=0 (so the caller knows it's drifting).
# After _DRIFT_MAX_CONSECUTIVE such hits in a row, object_present=False
# (caller treats as lost and re-inits).
_AREA_HISTORY_LEN = 10
_AREA_JUMP_RATIO = 1.5
_DRIFT_LOW_SCORE = 0.30
_DRIFT_MAX_CONSECUTIVE = 5


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class SegmentTextResult(TypedDict):
    masks: list[Mask]              # uint8 [H, W], 0 background / 255 foreground
    scores: list[float]            # confidence per mask, best-first
    boxes: list[BoundingBox2D]     # tight boxes (text prompt only)


class SegmentResult(TypedDict):
    masks: list[Mask]
    scores: list[float]


class TrackerInitResult(TypedDict):
    tracker_id: str                # opaque session id; "" when nothing detected
    initial_mask: Mask | None
    initial_box: BoundingBox2D | None
    score: float
    object_present: bool


class TrackerUpdateResult(TypedDict):
    mask: Mask | None
    box: BoundingBox2D | None
    confidence: float
    object_present: bool


class TrackerCloseResult(TypedDict):
    closed: bool


# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_image_lock = threading.Lock()
_image_model: Any = None
_image_processor: Any = None

_tracker_lock = threading.Lock()
_tracker_predictor: Any = None


def _get_model(device: str | None = None) -> tuple[Any, Any]:
    """Load the SAM3 image model + processor once (module-level singleton)."""
    global _image_model, _image_processor
    with _image_lock:
        if _image_model is None:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            dev = device or _DEVICE
            logger.info("Loading SAM3 model on %s ...", dev)
            model = build_sam3_image_model(enable_inst_interactivity=True)
            model = model.to(dev)
            _image_model = model
            _image_processor = Sam3Processor(model, confidence_threshold=0.0)
            logger.info("SAM3 model loaded on %s.", dev)
        return _image_model, _image_processor


def _get_tracker_predictor() -> Any:
    """Load the SAM3 video predictor once (single-GPU, streaming mode)."""
    global _tracker_predictor
    with _tracker_lock:
        if _tracker_predictor is None:
            _ensure_cc_compiler()
            import torch
            from sam3.model_builder import build_sam3_video_predictor

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Single-GPU mode: gpus_to_use of size 1 means world_size=1 and no
            # worker processes are spawned, so we can call predictor.model
            # directly from this process.
            gpu_id = (
                torch.cuda.current_device() if torch.cuda.is_available() else 0
            )
            logger.info("Loading SAM3 video predictor (single GPU, dev=%d) ...", gpu_id)
            # Disable temporal disambiguation. The default build uses
            # hotstart_delay=15 + hotstart_unmatch_thresh=8 + masklet
            # confirmation, which expects a fresh text-grounded re-detection
            # of the object on every frame. In streaming mode we add one new
            # frame per tracker_update with no fresh prompt; the hotstart
            # logic interprets that as "tracklet unmatched by detector" and
            # quietly removes the masklet on frame ~3, surfacing as
            # `object_present=False` for every subsequent tracker_update.
            # Setting apply_temporal_disambiguation=False switches to the
            # ablation build (hotstart_delay=0, no removal heuristics) which
            # is the right behavior for a single-target visual-prompt stream.
            _tracker_predictor = build_sam3_video_predictor(
                gpus_to_use=[gpu_id],
                apply_temporal_disambiguation=False,
            )
            logger.info("SAM3 video predictor loaded.")
        return _tracker_predictor


def _ensure_cc_compiler() -> None:
    """Ensure os.environ['CC'] points to an existing compiler.

    Triton (triton/runtime/build.py:26) uses CC to JIT-compile the SAM3
    video predictor's NMS kernels. If CC is unset or stale (Ray Serve envs
    have been seen to inject /usr/bin/gcc-13 which doesn't exist on the
    host), force it to a real compiler. Idempotent — no-op if already valid.
    """
    cc = os.environ.get("CC")
    if cc and os.path.exists(cc):
        return
    for cand in (
        "/usr/bin/gcc-12",
        "/usr/bin/gcc-11",
        "/usr/bin/gcc",
        shutil.which("gcc") or "",
        shutil.which("clang") or "",
    ):
        if cand and os.path.exists(cand):
            os.environ["CC"] = cand
            return


# ---------------------------------------------------------------------------
# Image / mask / box helpers (former proto encode-decode layer, numpy-fied)
# ---------------------------------------------------------------------------


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert a tensor-like object to a numpy array, handling bfloat16."""
    import torch

    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "cpu"):
        t = tensor.cpu()
        if hasattr(t, "dtype") and t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def _to_pil(image: np.ndarray):
    """RGB uint8 [H, W, 3] numpy image -> PIL Image."""
    from PIL import Image

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError("sam3", f"expected RGB uint8 [H, W, 3] image, got shape {arr.shape}")
    return Image.fromarray(arr, "RGB")


def _autocast(device: str):
    import torch

    return (
        torch.autocast(device, dtype=torch.bfloat16)
        if "cuda" in device
        else torch.autocast("cpu")
    )


def _mask_u8(mask_bool: np.ndarray) -> Mask:
    """Boolean mask -> gap Mask (uint8, 0 bg / 255 fg), squeezed to (H, W)."""
    if mask_bool.ndim > 2:
        mask_bool = mask_bool.reshape(mask_bool.shape[-2], mask_bool.shape[-1])
    return mask_bool.astype(np.uint8) * 255


def _box_dict(box_pixels) -> BoundingBox2D:
    return {
        "x1": float(box_pixels[0]),
        "y1": float(box_pixels[1]),
        "x2": float(box_pixels[2]),
        "y2": float(box_pixels[3]),
    }


def _xyxy_to_normalized_xywh(
    box_xyxy: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> list[float]:
    x1, y1, x2, y2 = box_xyxy
    x1n = max(0.0, x1) / image_w
    y1n = max(0.0, y1) / image_h
    x2n = min(float(image_w), x2) / image_w
    y2n = min(float(image_h), y2) / image_h
    return [x1n, y1n, max(0.0, x2n - x1n), max(0.0, y2n - y1n)]


# ---------------------------------------------------------------------------
# Image segmentation tools
# ---------------------------------------------------------------------------


@tool(
    name="sam3.segment_text",
    summary="Segment all instances matching a natural-language description; returns masks, scores and boxes best-first.",
    tags=("perception",),
)
def segment_text(
    image: np.ndarray,
    query: str,
    max_results: int = _SEGMENT_TEXT_TOP_K,
) -> SegmentTextResult:
    """Text-prompted segmentation (one mask per detected instance).

    ``max_results`` caps the returned detections (score-sorted descending);
    cluttered scenes can yield 100+ instances and most callers consume only
    ``masks[0]``. Pass ``max_results <= 0`` to return everything.
    """
    model, processor = _get_model()
    pil_image = _to_pil(image)

    try:
        with _autocast(_DEVICE):
            inference_state = processor.set_image(pil_image)
            output = processor.set_text_prompt(
                state=inference_state, prompt=query
            )
    except Exception as e:
        raise PerceptionFailed(f"SAM3 inference failed: {e}") from e

    masks_tensor = output.get("masks")
    boxes_tensor = output.get("boxes")
    scores_tensor = output.get("scores")

    if masks_tensor is None or boxes_tensor is None:
        return {"masks": [], "scores": [], "boxes": []}

    masks_np = _to_numpy(masks_tensor)
    boxes_np = _to_numpy(boxes_tensor)
    scores_np = _to_numpy(scores_tensor)

    # Squeeze masks if needed: (N, 1, H, W) -> (N, H, W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    num_preds = len(scores_np)

    # Build parallel lists of (mask, score, box) and sort by score desc,
    # capped to top-K.
    indices = list(range(num_preds))
    indices.sort(key=lambda i: float(scores_np[i]), reverse=True)
    if max_results > 0:
        indices = indices[:max_results]

    masks_out: list[Mask] = []
    scores_out: list[float] = []
    boxes_out: list[BoundingBox2D] = []

    for i in indices:
        bool_mask = masks_np[i] > 0
        masks_out.append(_mask_u8(bool_mask))
        scores_out.append(float(scores_np[i]))
        boxes_out.append(_box_dict(boxes_np[i]))

    return {"masks": masks_out, "scores": scores_out, "boxes": boxes_out}


@tool(
    name="sam3.segment_point",
    summary="Segment the object at a pixel coordinate; returns candidate masks with scores best-first.",
    tags=("perception",),
)
def segment_point(
    image: np.ndarray,
    pixel_x: float,
    pixel_y: float,
) -> SegmentResult:
    """Point-prompted segmentation (foreground point, multimask output)."""
    model, processor = _get_model()
    pil_image = _to_pil(image)

    if getattr(model, "inst_interactive_predictor", None) is None:
        raise ToolError(
            "sam3.segment_point",
            "Instance interactivity not enabled on SAM3 model",
        )

    try:
        with _autocast(_DEVICE):
            inference_state = processor.set_image(pil_image)
            point_coords = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int64)  # foreground
            masks, scores, _ = model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
    except Exception as e:
        raise PerceptionFailed(f"SAM3 point prompt inference failed: {e}") from e

    return _sorted_segment_result(masks, scores)


@tool(
    name="sam3.segment_box",
    summary="Segment within a bounding box, optionally refined by a foreground point; returns masks with scores best-first.",
    tags=("perception",),
)
def segment_box(
    image: np.ndarray,
    box: BoundingBox2D,
    pixel_x: float = 0.0,
    pixel_y: float = 0.0,
    use_point: bool = False,
) -> SegmentResult:
    """Box-prompted segmentation. With ``use_point=True`` the pixel becomes an
    additional foreground point (higher quality when a pointing model such as
    Molmo supplies one)."""
    model, processor = _get_model()
    pil_image = _to_pil(image)

    if getattr(model, "inst_interactive_predictor", None) is None:
        raise ToolError(
            "sam3.segment_box",
            "Instance interactivity not enabled on SAM3 model",
        )

    try:
        with _autocast(_DEVICE):
            inference_state = processor.set_image(pil_image)

            box_np = np.array(
                [box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float32
            )
            kwargs: dict[str, Any] = {"box": box_np, "multimask_output": True}

            if use_point:
                kwargs["point_coords"] = np.array(
                    [[pixel_x, pixel_y]], dtype=np.float32
                )
                kwargs["point_labels"] = np.array([1], dtype=np.int64)

            masks, scores, _ = model.predict_inst(inference_state, **kwargs)
    except Exception as e:
        raise PerceptionFailed(f"SAM3 box prompt inference failed: {e}") from e

    return _sorted_segment_result(masks, scores)


def _sorted_segment_result(masks, scores) -> SegmentResult:
    """Shared mask post-processing for the point/box prompt paths."""
    masks_np = np.asarray(masks)
    scores_np = np.asarray(scores)

    if masks_np.size == 0 or scores_np.size == 0:
        return {"masks": [], "scores": []}

    # Sort by score descending
    sort_idx = np.argsort(scores_np)[::-1]
    masks_np = masks_np[sort_idx]
    scores_np = scores_np[sort_idx]

    masks_out: list[Mask] = []
    scores_out: list[float] = []

    for i in range(len(scores_np)):
        bool_mask = masks_np[i] > 0
        h, w = bool_mask.shape[-2], bool_mask.shape[-1]
        # Handle potential extra dimensions: squeeze to (H, W)
        if bool_mask.ndim > 2:
            bool_mask = bool_mask.reshape(h, w)
        masks_out.append(_mask_u8(bool_mask))
        scores_out.append(float(scores_np[i]))

    return {"masks": masks_out, "scores": scores_out}


# ---------------------------------------------------------------------------
# Stateful video tracker (ported session semantics)
# ---------------------------------------------------------------------------


@dataclass
class _TrackerSession:
    object_name: str
    video_session_id: str
    last_mask: np.ndarray | None
    last_box_pixels: tuple[float, float, float, float] | None
    last_score: float
    target_text: str
    area_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=_AREA_HISTORY_LEN),
    )
    consecutive_drift: int = 0
    last_touched: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, _TrackerSession] = {}
_table_lock = threading.Lock()


def _close_video_session(predictor: Any, video_session_id: str) -> None:
    try:
        predictor.handle_request({
            "type": "close_session",
            "session_id": video_session_id,
        })
    except Exception:
        logger.debug("close_session failed", exc_info=True)


def _reap_stale_sessions(predictor: Any, ttl_s: float = _DEFAULT_TTL_S) -> None:
    """Evict sessions idle longer than ``ttl_s`` (lazy replacement for the
    servicer's background reaper thread)."""
    now = time.monotonic()
    with _table_lock:
        stale = [
            tid for tid, s in _sessions.items() if (now - s.last_touched) > ttl_s
        ]
        for tid in stale:
            sess = _sessions.pop(tid, None)
            if sess is not None:
                _close_video_session(predictor, sess.video_session_id)
    if stale:
        logger.info("Reaped %d stale tracker sessions", len(stale))


@tool(
    name="sam3.tracker_init",
    summary="Open a SAM3 video-tracker session seeded with one frame and a text, box, or point prompt; returns a tracker_id and the initial mask.",
    tags=("perception",),
)
def tracker_init(
    image: np.ndarray,
    text: str = "",
    box: BoundingBox2D | None = None,
    pixel_x: float = 0.0,
    pixel_y: float = 0.0,
    use_point: bool = False,
    object_name: str = "",
) -> TrackerInitResult:
    """Exactly one of ``box``, point (``use_point=True``), ``text`` is
    consumed; precedence is box > point > text. When the initial detection
    finds nothing, the session is closed and ``object_present=False`` with an
    empty ``tracker_id`` is returned (no exception — mirror of the servicer).
    """
    predictor = _get_tracker_predictor()
    _ensure_cc_compiler()
    _reap_stale_sessions(predictor)

    pil = _to_pil(image)
    image_w, image_h = pil.size

    # Resolve prompt — contract is box > point > text.
    prompt_text: str | None = None
    bbox_norm_xywh: list[float] | None = None
    prompt_kind = "none"

    if box is not None and box["x2"] > box["x1"] and box["y2"] > box["y1"]:
        bbox_norm_xywh = _xyxy_to_normalized_xywh(
            (box["x1"], box["y1"], box["x2"], box["y2"]), image_w, image_h,
        )
        prompt_kind = "box"
    elif use_point:
        # Convert a point to a small bbox around it. The video predictor's
        # bbox prompt path is more reliable for init than a single point.
        px = float(pixel_x) / image_w
        py = float(pixel_y) / image_h
        half = 0.05
        bbox_norm_xywh = [
            max(0.0, px - half), max(0.0, py - half),
            min(1.0, 2 * half), min(1.0, 2 * half),
        ]
        prompt_kind = "point"
    elif text:
        prompt_text = text
        prompt_kind = "text"
    else:
        raise ToolError(
            "sam3.tracker_init",
            "requires one of: box, point (use_point=True), text",
        )

    # Open a SAM3 video session seeded with the first PIL image.
    sess = predictor.handle_request({
        "type": "start_session",
        "resource_path": [pil],
    })
    video_session_id = sess["session_id"]

    # Flip is_image_only to False so the video tracker's visual-prompt +
    # memory-bank machinery engages on the init frame. is_image_type([pil])
    # sets it to True for length-1 lists; we override because we *will*
    # add new frames via _streaming.add_new_frame and need the tracker
    # initialized correctly from frame 0.
    try:
        inference_states = predictor._ALL_INFERENCE_STATES
        sess_obj = inference_states.get(video_session_id)
        if sess_obj is not None:
            sess_obj["state"]["is_image_only"] = False
    except Exception:
        logger.debug("could not flip is_image_only", exc_info=True)

    logger.debug(
        "tracker_init: prompt_kind=%s bbox_norm_xywh=%s text=%r image=%dx%d",
        prompt_kind, bbox_norm_xywh, prompt_text, image_w, image_h,
    )
    try:
        if bbox_norm_xywh is not None:
            prompt_resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": video_session_id,
                "frame_index": 0,
                "bounding_boxes": [bbox_norm_xywh],
                "bounding_box_labels": [1],
            })
        else:
            prompt_resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": video_session_id,
                "frame_index": 0,
                "text": prompt_text,
            })
    except Exception as e:
        logger.warning("add_prompt failed on init: %s", e, exc_info=True)
        _close_video_session(predictor, video_session_id)
        return {
            "tracker_id": "",
            "initial_mask": None,
            "initial_box": None,
            "score": 0.0,
            "object_present": False,
        }

    from gap_skills.tools.sam3 import _streaming

    outputs = prompt_resp.get("outputs", {})
    mask, box_pixels, score, present = _streaming.first_object_from_outputs(outputs)

    if not present or mask is None or box_pixels is None:
        _close_video_session(predictor, video_session_id)
        logger.warning(
            "tracker_init: no object detected (prompt_kind=%s text=%r bbox=%s)",
            prompt_kind, prompt_text, bbox_norm_xywh,
        )
        return {
            "tracker_id": "",
            "initial_mask": None,
            "initial_box": None,
            "score": 0.0,
            "object_present": False,
        }

    tracker_id = uuid.uuid4().hex
    session = _TrackerSession(
        object_name=object_name,
        video_session_id=video_session_id,
        last_mask=mask,
        last_box_pixels=box_pixels,
        last_score=score,
        target_text=prompt_text or object_name or "",
        last_touched=time.monotonic(),
    )
    session.area_history.append(int(mask.sum()))
    with _table_lock:
        _sessions[tracker_id] = session

    logger.info(
        "tracker_init created %s prompt=%s name=%r score=%.3f",
        tracker_id, prompt_kind, object_name, score,
    )
    return {
        "tracker_id": tracker_id,
        "initial_mask": _mask_u8(mask),
        "initial_box": _box_dict(box_pixels),
        "score": float(score),
        "object_present": True,
    }


@tool(
    name="sam3.tracker_update",
    summary="Advance a SAM3 tracker session by one frame; returns the tracked mask, box and confidence.",
    tags=("perception",),
)
def tracker_update(tracker_id: str, image: np.ndarray) -> TrackerUpdateResult:
    """Pushes the new frame into the session's SAM3 memory bank and
    propagates one step. Drift heuristics (mask-area jump vs running median,
    low confidence) hold the last good mask out with ``confidence=0``; after
    5 consecutive drift hits, ``object_present=False`` so the caller can
    re-init. Raises ToolError for an unknown ``tracker_id`` (never created,
    closed, or TTL-evicted)."""
    predictor = _get_tracker_predictor()
    _ensure_cc_compiler()
    _reap_stale_sessions(predictor)

    with _table_lock:
        session = _sessions.get(tracker_id)
    if session is None:
        raise ToolError("sam3.tracker_update", f"unknown tracker_id {tracker_id!r}")

    pil = _to_pil(image)

    from gap_skills.tools.sam3 import _streaming

    with session.lock:
        inference_states = predictor._ALL_INFERENCE_STATES
        sess_state_obj = inference_states.get(session.video_session_id)
        if sess_state_obj is None:
            # SAM3-side session vanished (rare — predictor reset).
            session.last_touched = time.monotonic()
            return {"mask": None, "box": None, "confidence": 0.0, "object_present": False}
        inference_state = sess_state_obj["state"]

        new_frame_idx = _streaming.add_new_frame(
            predictor.model, inference_state, pil,
        )

        outputs: dict | None = None
        for resp in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session.video_session_id,
            "propagation_direction": "forward",
            "start_frame_index": new_frame_idx,
            "max_frame_num_to_track": 1,
        }):
            outputs = resp.get("outputs")
            break

        session.last_touched = time.monotonic()

        if outputs is None:
            return {"mask": None, "box": None, "confidence": 0.0, "object_present": False}

        mask, box_pixels, score, present = _streaming.first_object_from_outputs(outputs)
        if not present or mask is None or box_pixels is None:
            return {"mask": None, "box": None, "confidence": 0.0, "object_present": False}

        new_area = int(mask.sum())
        drift_signals: list[str] = []
        if len(session.area_history) >= 3:
            running_median = float(np.median(list(session.area_history)))
            if running_median > 0 and new_area > _AREA_JUMP_RATIO * running_median:
                drift_signals.append(f"area_jump:{new_area}/{running_median:.0f}")
        if score < _DRIFT_LOW_SCORE:
            drift_signals.append(f"low_score:{score:.2f}")

        if drift_signals:
            session.consecutive_drift += 1
            logger.info(
                "drift on tracker %s frame=%d signals=[%s] consec=%d/%d",
                tracker_id, new_frame_idx,
                ",".join(drift_signals),
                session.consecutive_drift, _DRIFT_MAX_CONSECUTIVE,
            )
            if session.consecutive_drift >= _DRIFT_MAX_CONSECUTIVE:
                return {"mask": None, "box": None, "confidence": 0.0, "object_present": False}
            # Hold the previous good mask out so the caller doesn't act on
            # the drifted one. Confidence=0 signals the caller to skip.
            return {
                "mask": _mask_u8(session.last_mask),
                "box": _box_dict(session.last_box_pixels),
                "confidence": 0.0,
                "object_present": True,
            }

        session.consecutive_drift = 0
        session.area_history.append(new_area)
        session.last_mask = mask
        session.last_box_pixels = box_pixels
        session.last_score = score

        return {
            "mask": _mask_u8(mask),
            "box": _box_dict(box_pixels),
            "confidence": float(score),
            "object_present": True,
        }


@tool(
    name="sam3.tracker_close",
    summary="Free a SAM3 tracker session. Idempotent: closing an unknown id is not an error.",
    tags=("perception",),
)
def tracker_close(tracker_id: str) -> TrackerCloseResult:
    with _table_lock:
        session = _sessions.pop(tracker_id, None)
    if session is not None and _tracker_predictor is not None:
        _close_video_session(_tracker_predictor, session.video_session_id)
    return {"closed": session is not None}
