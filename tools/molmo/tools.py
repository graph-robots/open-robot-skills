"""Molmo tool bundle — visual pointing and Q&A against a self-hosted vLLM.

Ported from the dev tree's Molmo gRPC servicer
into in-process ``@tool`` functions. Molmo has no hosted API: the backend is
an OpenAI-compatible vLLM endpoint you run yourself (see SKILL.md for the
serving recipe). ``molmo.point_prompt`` is the signature capability — a text
query resolved to a single pixel coordinate.

The point parser (:func:`_parse_points`) is ported verbatim and supports all
four Molmo output formats:

1. Molmo2 ``<points coords="type idx x y ...">label</points>`` (0-1000)
2. Molmo1 ``<point x="X" y="Y">`` tags (0-100)
3. Legacy ``<points x1=".." y1=".." ...>`` tags (0-100)
4. Fallback plain ``x, y`` pairs anywhere in the text (0-100)

Config:

- ``GAP_MOLMO_BASE_URL`` — base URL of the vLLM OpenAI-compatible API
  (required; e.g. ``http://127.0.0.1:8122/v1``).
- ``GAP_MOLMO_MODEL`` — model name served by vLLM
  (default :data:`DEFAULT_MODEL`).

Deviation from the source: after exhausting retries the source servicer
returned an empty string; here the failure raises :class:`gap.errors.ToolError`
so workflows can route ``on_error`` instead of silently "not finding" objects.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import time
from typing import TypedDict

import httpx
import numpy as np
from gap.errors import ToolError
from gap.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)

#: Default model name served by the vLLM backend. Override per-call with
#: ``model=`` or globally with the ``GAP_MOLMO_MODEL`` env var.
DEFAULT_MODEL = "allenai/Molmo2-8B"

_MAX_TOKENS = 1024  # ported from the source servicer
_MAX_RETRIES = 3
_BACKOFF_S = 1.0


class PointResult(TypedDict):
    pixel_x: float
    pixel_y: float
    found: bool


class QueryResult(TypedDict):
    text: str


class YesNoResult(TypedDict):
    answer: bool
    text: str


# ---------------------------------------------------------------------------
# Point parsing (ported verbatim from the source server.py:36-114, itself
# ported from HyRL/hyrl/integrations/molmo.py)
# ---------------------------------------------------------------------------


def _parse_points(text: str) -> tuple[list[tuple[float, float]], float]:
    """Parse point coordinates from model text output.

    Supports multiple formats:
    - Molmo2: <points coords="obj_idx x y ...">label</points> (normalized 0-1000)
    - Molmo1: <point x="X" y="Y"> (normalized 0-100)
    - Legacy: <points x1=".." y1=".."> (normalized 0-100)
    - Fallback: plain "x, y" pairs (normalized 0-100)

    Args:
        text: Generated text potentially containing point tags.

    Returns:
        Tuple of (points, norm_scale) where points is a list of (x, y) tuples
        and norm_scale is the normalization scale (100.0 for Molmo1, 1000.0 for Molmo2).
    """
    # 1) Molmo2 format: <points coords="type obj_idx x y ...">label</points>
    coords_match = re.search(
        r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE
    )
    if coords_match:
        nums = [float(n) for n in coords_match.group(1).split()]
        points: list[tuple[float, float]] = []
        # Skip first number (type indicator), then parse triplets: (obj_idx, x, y)
        i = 1
        while i + 2 <= len(nums):
            x, y = nums[i + 1], nums[i + 2]
            points.append((x, y))
            i += 3
        return points, 1000.0

    # 2) Parse one or more <point x=".." y=".."> tags (Molmo1 format)
    point_tags = re.findall(r"<point\b[^>]*>", text, flags=re.IGNORECASE)
    points = []
    for tag in point_tags:
        mx = re.search(
            r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE
        )
        my = re.search(
            r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE
        )
        if mx and my:
            points.append((float(mx.group(1)), float(my.group(1))))
    if points:
        return [
            (x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0
        ], 100.0

    # 3) Legacy <points x1=".." y1=".." ...> format
    tag_match = re.search(r"<points\b[^>]*>", text, flags=re.IGNORECASE)
    if tag_match:
        source = tag_match.group(0)
        xs = {
            int(i): float(v)
            for i, v in re.findall(
                r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source
            )
        }
        ys = {
            int(i): float(v)
            for i, v in re.findall(
                r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source
            )
        }
        idxs = sorted(set(xs) & set(ys))
        points = [(xs[i], ys[i]) for i in idxs]
        if points:
            return [
                (x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0
            ], 100.0

    # 4) Fallback: parse plain "x, y" pairs anywhere in text
    pairs = re.findall(r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text)
    points = [(float(x), float(y)) for x, y in pairs]
    return [
        (x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0
    ], 100.0


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"molmo expects a uint8 [H, W, 3] RGB array, got dtype={arr.dtype} "
            f"shape={arr.shape}"
        )
    return arr


def _image_to_data_url(image: np.ndarray) -> str:
    """Encode a uint8 [H, W, 3] RGB array as a PNG base64 data URL."""
    pil_image = Image.fromarray(image, "RGB")
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _base_url(tool_name: str) -> str:
    base_url = os.environ.get("GAP_MOLMO_BASE_URL", "")
    if not base_url:
        raise ToolError(
            tool_name,
            "GAP_MOLMO_BASE_URL is not set. Molmo has no hosted API — this "
            "bundle queries a self-hosted vLLM server exposing the "
            "OpenAI-compatible API (see the molmo bundle's SKILL.md for the "
            "vllm serve recipe), e.g. GAP_MOLMO_BASE_URL=http://127.0.0.1:8122/v1. "
            "For an API-only (zero-GPU) alternative, use the gemini-er bundle "
            "(gemini-er.detect).",
        )
    return base_url


def _http_client() -> httpx.Client:
    """Build the HTTP client for the vLLM backend (test seam)."""
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def _query_model(tool_name: str, prompt: str, image: np.ndarray | None) -> str:
    """Send a chat completion request to the vLLM backend (payload verbatim)."""
    chat_url = f"{_base_url(tool_name).rstrip('/')}/chat/completions"
    model = os.environ.get("GAP_MOLMO_MODEL", "") or DEFAULT_MODEL

    content: list[dict] = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(image)},
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0.0,
        "stop": ["<|endoftext|>"],
    }

    last_exc: Exception | None = None
    with _http_client() as client:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.post(chat_url, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "vLLM request failed (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S * (2 ** attempt))

    raise ToolError(
        tool_name,
        f"vLLM backend unavailable after {_MAX_RETRIES} attempts: "
        f"chat_url={chat_url}, error={last_exc}",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="molmo.point_prompt",
    summary="Point at a named object in an image; returns pixel coordinates.",
    tags=("perception",),
)
def point_prompt(image: np.ndarray, query: str) -> PointResult:
    """Resolve *query* to a single pixel coordinate in *image*.

    Args:
        image: uint8 [H, W, 3] RGB array.
        query: What to point at (e.g. "red button").

    Returns:
        ``{"pixel_x": float, "pixel_y": float, "found": bool}`` — the first
        detected point in pixel coordinates; ``found=False`` (with zeros)
        when the model produced no parseable point.
    """
    arr = _validate_image(image)
    height, width = arr.shape[:2]

    generated_text = _query_model("molmo.point_prompt", f"Point at {query}", arr)
    points, norm_scale = _parse_points(generated_text)
    if not points:
        logger.info("molmo.point_prompt: no points parsed for %r", query)
        return {"pixel_x": 0.0, "pixel_y": 0.0, "found": False}

    # Convert the first detected point to pixel coordinates
    nx, ny = points[0]
    return {
        "pixel_x": nx / norm_scale * width,
        "pixel_y": ny / norm_scale * height,
        "found": True,
    }


@tool(
    name="molmo.query",
    summary="Free-form visual question answering via Molmo.",
    tags=("perception",),
)
def query(prompt: str, image: np.ndarray | None = None) -> QueryResult:
    """Ask Molmo a free-form question, optionally about an image.

    Returns:
        ``{"text": <model response>}``.
    """
    arr = _validate_image(image) if image is not None else None
    return {"text": _query_model("molmo.query", prompt, arr)}


@tool(
    name="molmo.query_yes_no",
    summary="Yes/no visual question answering via Molmo; coerces the reply to a bool.",
    tags=("perception",),
)
def query_yes_no(prompt: str, image: np.ndarray | None = None) -> YesNoResult:
    """Ask Molmo a yes/no question, optionally about an image.

    The coercion is ported verbatim from the source servicer: ``answer`` is
    true iff the literal substring ``"yes"`` appears in the lowercased reply.

    Returns:
        ``{"answer": <bool>, "text": <raw model response>}``.
    """
    arr = _validate_image(image) if image is not None else None
    text = _query_model("molmo.query_yes_no", prompt, arr)
    return {"answer": "yes" in text.lower(), "text": text}
