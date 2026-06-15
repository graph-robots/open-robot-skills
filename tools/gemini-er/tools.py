"""Gemini Robotics-ER 2D detection tool bundle (google-genai SDK).

New implementation: the dev tree's workflows call a ``gemini_er.v1.GeminiER``
service whose implementation was never committed. The contract here is
reconstructed from the consumer
(``examples/libero_pro/full_popcorn_cycle_cartesian/scripts/perceive_gemini_er.py``):
``Detect(image, prompt)`` returns a ``detections`` list whose entries carry a
pixel-space ``box`` (:class:`gap.types.BoundingBox2D`), a ``label`` string,
and a ``score`` float the caller maxes over to pick the best detection.

The model is prompted for the Gemini 2D-detection JSON convention — entries
with ``"box_2d": [ymin, xmin, ymax, xmax]`` normalized to 0–1000 — which is
converted to pixel coordinates here. Parsing is deliberately forgiving:
no detection (or unparseable output) yields an empty list, never an error.

Config:

- ``GAP_GEMINI_ER_MODEL`` — model name (default :data:`DEFAULT_MODEL`).
- API key via the SDK's default resolution (``GOOGLE_API_KEY`` /
  ``GEMINI_API_KEY``).

``google.genai`` is imported lazily at call time — install the bundle extra:
``pip install -e "open-robot-skills[gemini-er]"``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import TypedDict

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import tool
from gap_core.types import BoundingBox2D
from PIL import Image

logger = logging.getLogger(__name__)

#: Default Gemini Robotics-ER model. Override per-call with ``model=`` or
#: globally with the ``GAP_GEMINI_ER_MODEL`` env var.
DEFAULT_MODEL = "gemini-robotics-er-1.5-preview"

_DETECT_PROMPT = (
    'Detect the 2D bounding boxes of "{query}" in the image. '
    "Return ONLY a JSON array (no prose, no markdown). Each entry must be an "
    'object with the keys "box_2d" — [ymin, xmin, ymax, xmax] as integers '
    'normalized to 0-1000 — and "label" — a short descriptive string. '
    "Return [] if nothing matches."
)


class Detection(TypedDict):
    box: BoundingBox2D
    label: str
    score: float


class DetectResult(TypedDict):
    detections: list[Detection]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_genai():
    """Lazy-import the google-genai SDK (heavy; optional bundle dep)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ToolError(
            "gemini-er.detect",
            "google-genai is not installed; install the bundle extra: "
            'pip install -e "open-robot-skills[gemini-er]"',
        ) from exc
    return genai, types


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"gemini-er expects a uint8 [H, W, 3] RGB array, got "
            f"dtype={arr.dtype} shape={arr.shape}"
        )
    return arr


def _png_bytes(image: np.ndarray) -> bytes:
    pil_image = Image.fromarray(image, "RGB")
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        return buf.getvalue()


def _extract_json_array(text: str) -> list:
    """Pull a JSON array out of model output. Never raises — [] on failure."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_detections(
    text: str, *, width: int, height: int, query: str
) -> list[Detection]:
    """Convert normalized 0-1000 ``box_2d`` entries to pixel detections."""
    detections: list[Detection] = []
    for entry in _extract_json_array(text):
        if not isinstance(entry, dict):
            continue
        box_2d = entry.get("box_2d")
        if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in box_2d)
        except (TypeError, ValueError):
            continue
        box: BoundingBox2D = {
            "x1": _clamp(xmin / 1000.0 * width, 0.0, float(width)),
            "y1": _clamp(ymin / 1000.0 * height, 0.0, float(height)),
            "x2": _clamp(xmax / 1000.0 * width, 0.0, float(width)),
            "y2": _clamp(ymax / 1000.0 * height, 0.0, float(height)),
        }
        try:
            score = float(entry.get("score", entry.get("confidence", 1.0)))
        except (TypeError, ValueError):
            score = 1.0
        detections.append({
            "box": box,
            "label": str(entry.get("label") or query),
            "score": score,
        })
    return detections


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(
    name="gemini-er.detect",
    summary="Detect pixel-space 2D bounding boxes for a text query via Gemini Robotics-ER.",
    tags=("perception",),
)
def detect(
    image: np.ndarray,
    query: str,
    model: str | None = None,
) -> DetectResult:
    """Detect 2D bounding boxes for *query* in *image*.

    Args:
        image: uint8 [H, W, 3] RGB array.
        query: Open-vocabulary object description (e.g. "red mug handle").
        model: Per-call model override (else ``GAP_GEMINI_ER_MODEL``,
            else :data:`DEFAULT_MODEL`).

    Returns:
        ``{"detections": [{"box": BoundingBox2D, "label": str, "score": float}]}``
        — boxes in pixel coordinates, best-detection selection is
        ``max(detections, key=lambda d: d["score"])``. Empty list when the
        model finds nothing (never an error).
    """
    arr = _validate_image(image)
    height, width = arr.shape[:2]

    genai, genai_types = _import_genai()
    client = genai.Client()  # API key via GOOGLE_API_KEY / GEMINI_API_KEY
    image_part = genai_types.Part.from_bytes(
        data=_png_bytes(arr), mime_type="image/png",
    )
    model_name = model or os.environ.get("GAP_GEMINI_ER_MODEL") or DEFAULT_MODEL
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, _DETECT_PROMPT.format(query=query)],
        )
    except Exception as exc:
        raise ToolError("gemini-er.detect", f"generate_content failed: {exc}") from exc

    text = response.text or ""
    detections = _parse_detections(text, width=width, height=height, query=query)
    if not detections:
        logger.debug("gemini-er.detect: no detections for query=%r (raw=%r)",
                     query, text[:200])
    return {"detections": detections}
