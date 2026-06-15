"""grounding-dino tool bundle — zero-shot object detection.

Extracted from the original Grounding DINO gRPC servicer in the dev
tree. The transformers pipeline
(processor → model → post_process_grounded_object_detection) is verbatim;
the proto byte decode/encode is replaced by numpy arrays + gap.types dicts.

The model loads lazily on first call (module-level singleton); importing
this module never pulls torch/transformers. Knobs via env:

- ``GAP_DINO_DEVICE`` — torch device (default ``cuda``).
- ``GAP_DINO_MODEL``  — HuggingFace model name
  (default ``IDEA-Research/grounding-dino-base``).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PerceptionFailed, ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "IDEA-Research/grounding-dino-base"
_DEFAULT_BOX_THRESHOLD = 0.20
_DEFAULT_TEXT_THRESHOLD = 0.20

_DEVICE = os.environ.get("GAP_DINO_DEVICE", "cuda")
_MODEL_NAME = os.environ.get("GAP_DINO_MODEL", _DEFAULT_MODEL_NAME)

_load_lock = threading.Lock()
_model: Any = None
_processor: Any = None


class Detection(TypedDict):
    box: BoundingBox2D    # [x1, y1, x2, y2] in pixels
    label: str            # matched text label
    score: float          # confidence [0, 1]


class DetectResult(TypedDict):
    detections: list[Detection]


def weights_cached() -> bool | None:
    """Filesystem-only weight-cache probe for ``gap check``.

    Checks the Hugging Face cache for the configured model's config.json
    (the canonical presence marker) — never downloads, never imports
    torch. ``None`` when huggingface_hub is unavailable ("unknown").
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        result = try_to_load_from_cache(_MODEL_NAME, "config.json")
    except Exception:
        return None
    return isinstance(result, str)


def _get_model() -> tuple[Any, Any]:
    """Load the Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor
    with _load_lock:
        if _model is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            logger.info(
                "Loading Grounding DINO model: %s on %s ...", _MODEL_NAME, _DEVICE
            )
            _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_NAME)
            model = model.to(_DEVICE)
            model.eval()
            _model = model
            logger.info("Grounding DINO model loaded successfully on %s.", _DEVICE)
        return _model, _processor


@tool(
    name="grounding-dino.detect",
    summary="Zero-shot object detection from a text prompt; returns labeled boxes with confidence scores.",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str,
    box_threshold: float = _DEFAULT_BOX_THRESHOLD,
    text_threshold: float = _DEFAULT_TEXT_THRESHOLD,
) -> DetectResult:
    """Detect objects matching ``query`` in an RGB uint8 [H, W, 3] image.

    Grounding DINO expects period-separated phrases (``"red cube. green
    cube."``); a missing trailing period is appended automatically. Returns
    an empty detections list when nothing clears the thresholds — select the
    best box downstream (e.g. closest to a pointing-model pixel) and feed it
    to ``sam3.segment_box`` for a pixel-accurate mask.
    """
    import torch
    from PIL import Image

    model, processor = _get_model()

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ToolError(
            "grounding-dino.detect",
            f"expected RGB uint8 [H, W, 3] image, got shape {arr.shape}",
        )
    pil_image = Image.fromarray(arr, "RGB")

    # GDINO requires period-terminated phrases
    text_prompt = query
    if not text_prompt.endswith("."):
        text_prompt = text_prompt + "."

    # Defaults if zero/unset (mirrors the servicer's proto-default handling)
    box_threshold = box_threshold if box_threshold > 0 else _DEFAULT_BOX_THRESHOLD
    text_threshold = text_threshold if text_threshold > 0 else _DEFAULT_TEXT_THRESHOLD

    try:
        inputs = processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(_DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
    except Exception as e:
        raise PerceptionFailed(f"Grounding DINO detection failed: {e}") from e

    detections: list[Detection] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    for box, score, label in zip(boxes, scores, labels, strict=False):
        detections.append({
            "box": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            },
            "label": str(label),
            "score": float(score),
        })

    logger.info("grounding-dino.detect returning %d detections.", len(detections))
    return {"detections": detections}
