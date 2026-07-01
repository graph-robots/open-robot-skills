"""VLM tool bundle — hosted vision-language Q&A behind a provider switch.

In-process ``@tool`` functions: ``Query`` / ``QueryYesNo`` semantics with a
free-form prompt and one or more images (``images=`` carries several
context frames in one request). The tool signatures deliberately expose no
system prompt and no temperature knob — prompts are self-contained and
sampling is pinned for determinism.

Providers — selected by ``GAP_VLM_PROVIDER`` (default ``"openrouter"``); a
per-call ``provider=`` kwarg overrides the env. Every ``GAP_VLM_*`` knob
inherits from the matching ``GAP_LLM_*`` / google-SDK env var when unset
(see :func:`_resolve_provider`, :func:`_resolve_model`,
:func:`_resolve_vertex_project`, :func:`_resolve_vertex_region`) — so a
user who configures the agent's LLM doesn't have to re-configure the VLM
bundle separately. Set ``GAP_VLM_*`` explicitly only to route the VLM to
a different provider/model than the agent.

- ``openrouter`` (default) — OpenRouter's OpenAI-compatible
  chat-completions API (data-URL image blocks, ``temperature: 0.0``, 3
  retries with exponential backoff). Base URL defaults to
  ``https://openrouter.ai/api/v1`` (override with ``GAP_VLM_BASE_URL`` for
  any other OpenAI-compatible server, e.g. a local vLLM). Key from
  ``GAP_VLM_API_KEY`` (else ``OPENROUTER_API_KEY``); model from
  ``GAP_VLM_MODEL`` (else ``GAP_LLM_MODEL`` else :data:`DEFAULT_MODEL`).
- ``vertex`` — Vertex AI via ``google-genai`` (Gemini models). Lazy
  import; install the vertex extra
  (``pip install "graph-as-policy[vertex]"``). Config:
  ``GAP_VLM_MODEL`` (else ``GAP_LLM_MODEL`` else :data:`DEFAULT_MODEL`) +
  ``GAP_VLM_PROJECT_ID`` (else ``GOOGLE_CLOUD_PROJECT``) +
  ``GAP_VLM_REGION`` (else ``GOOGLE_CLOUD_REGION`` else
  ``GOOGLE_CLOUD_LOCATION`` else ``"global"``).

Generation config: perception callers (the pairwise tournament, the
yes/no verify gate) are binary judgments that depend on deterministic
decoding, so both providers pin ``temperature: 0.0`` + ``max_tokens:
1024`` with 3 retries + exponential backoff.

All functions are synchronous — the gap runtime is threaded, not async.
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
from gap_core.errors import ToolError
from gap_core.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)

#: Default model when none is resolved (used by both providers). Override
#: per-call with ``model=`` or globally with ``GAP_VLM_MODEL`` /
#: ``GAP_LLM_MODEL``. On ``openrouter`` the slug may need a ``google/``
#: prefix depending on the account.
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

#: Provider used when neither ``provider=`` nor ``GAP_VLM_PROVIDER`` nor
#: ``GAP_LLM_PROVIDER`` is set.
DEFAULT_PROVIDER = "openrouter"


def _envstr(name: str) -> str:
    """``os.environ.get(name, "").strip()`` — empty string if unset/blank."""
    return os.environ.get(name, "").strip()


def _resolve_provider(provider: str | None) -> str:
    """Per-call override > ``GAP_VLM_PROVIDER`` > ``GAP_LLM_PROVIDER`` >
    :data:`DEFAULT_PROVIDER`. The ``GAP_LLM_*`` inheritance lets a user
    who's already configured the agent's LLM run the VLM bundle through
    the same provider without re-exporting a parallel set of env vars
    (the silent ``GAP_VLM_*`` defaults caused the dev-era milk-vs-soup
    mispick: missing creds → tournament fell back to "box 0 wins")."""
    return (
        (provider or "").strip().lower()
        or _envstr("GAP_VLM_PROVIDER").lower()
        or _envstr("GAP_LLM_PROVIDER").lower()
        or DEFAULT_PROVIDER
    )


def _resolve_model(model: str | None) -> str:
    """Per-call override > ``GAP_VLM_MODEL`` > ``GAP_LLM_MODEL`` >
    :data:`DEFAULT_MODEL`."""
    return (
        (model or "").strip()
        or _envstr("GAP_VLM_MODEL")
        or _envstr("GAP_LLM_MODEL")
        or DEFAULT_MODEL
    )


def _resolve_vertex_project() -> str:
    """``GAP_VLM_PROJECT_ID`` > ``GOOGLE_CLOUD_PROJECT`` (the documented
    google-genai knob). Empty string when unset — the caller raises with
    the install hint."""
    return (
        _envstr("GAP_VLM_PROJECT_ID")
        or _envstr("GOOGLE_CLOUD_PROJECT")
    )


def _resolve_vertex_region() -> str:
    """``GAP_VLM_REGION`` > ``GOOGLE_CLOUD_REGION`` > ``GOOGLE_CLOUD_LOCATION``
    > ``"global"`` (the documented Vertex default)."""
    return (
        _envstr("GAP_VLM_REGION")
        or _envstr("GOOGLE_CLOUD_REGION")
        or _envstr("GOOGLE_CLOUD_LOCATION")
        or "global"
    )

_MAX_TOKENS = 1024  # ported from the source servicer
_MAX_RETRIES = 3
_BACKOFF_S = 1.0
#: Deterministic decoding for the binary perception judgments (tournament
#: A/B picks, yes/no verify). The dev servicer's proxy path always sent
#: ``"temperature": 0.0``; this port applies it to every provider.
_TEMPERATURE = 0.0


class QueryResult(TypedDict):
    text: str


class YesNoResult(TypedDict):
    answer: bool
    text: str


# ---------------------------------------------------------------------------
# Image helpers (numpy-first: gap images are uint8 [H, W, 3], no byte packing)
# ---------------------------------------------------------------------------


def _validate_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"vlm expects a uint8 [H, W, 3] RGB array, got dtype={arr.dtype} "
            f"shape={arr.shape}"
        )
    return arr


def _png_b64(image: np.ndarray) -> str:
    """Encode a uint8 [H, W, 3] RGB array as a raw base64 PNG string."""
    arr = _validate_image(image)
    pil_image = Image.fromarray(arr, "RGB")
    with io.BytesIO() as buf:
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gather_images(
    image: np.ndarray | None, images: list | None
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    if image is not None:
        arrays.append(image)
    if images:
        arrays.extend(images)
    return [_validate_image(a) for a in arrays]


# ---------------------------------------------------------------------------
# Provider: openrouter (OpenRouter's OpenAI-compatible chat-completions API)
# ---------------------------------------------------------------------------


def _http_client() -> httpx.Client:
    """Build the HTTP client for the OpenAI-compatible path (test seam)."""
    return httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))


def _query_openrouter(prompt: str, images: list[np.ndarray], model: str | None) -> str:
    base_url = _envstr("GAP_VLM_BASE_URL") or "https://openrouter.ai/api/v1"
    model = _resolve_model(model)
    api_key = _envstr("GAP_VLM_API_KEY") or _envstr("OPENROUTER_API_KEY")
    chat_url = f"{base_url.rstrip('/')}/chat/completions"

    content: list[dict] = [{"type": "text", "text": prompt}]
    for arr in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png_b64(arr)}"},
        })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _MAX_TOKENS,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    last_exc: Exception | None = None
    with _http_client() as client:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.post(chat_url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "VLM request failed (attempt %d/%d, chat_url=%s): %s",
                    attempt + 1, _MAX_RETRIES, chat_url, exc,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S * (2 ** attempt))

    raise ToolError(
        "vlm",
        f"backend unavailable after {_MAX_RETRIES} attempts: "
        f"chat_url={chat_url}, error={last_exc}",
    )


# ---------------------------------------------------------------------------
# Provider: vertex (google-genai; Gemini models only)
# ---------------------------------------------------------------------------


def _is_claude_model(model: str) -> bool:
    """Check if a model name refers to a Claude model."""
    return "claude" in model.lower()


def _query_vertex(prompt: str, images: list[np.ndarray], model: str | None) -> str:
    model = _resolve_model(model)
    if _is_claude_model(model):
        raise ToolError(
            "vlm",
            f"vertex serves Gemini models only (got {model!r}); "
            "Claude-on-Vertex was removed with the anthropic dependency. "
            "Use a gemini-* model, or route Claude via the openrouter provider.",
        )
    project_id = _resolve_vertex_project()
    region = _resolve_vertex_region()

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ToolError(
            "vlm",
            "google-genai is not installed in the vlm bundle's venv "
            "(needed to route gemini-* via Vertex). Re-sync the bundle: "
            "`uv sync --project open-robot-skills/tools/vlm` "
            "(google-genai is declared in tools/vlm/pyproject.toml).",
        ) from exc

    client = genai.Client(vertexai=True, project=project_id, location=region)
    parts: list = [prompt]
    for arr in images:
        parts.append(types.Part.from_bytes(
            data=base64.b64decode(_png_b64(arr)), mime_type="image/png",
        ))
    config = types.GenerateContentConfig(
        temperature=_TEMPERATURE, max_output_tokens=_MAX_TOKENS,
    )
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=parts, config=config,
            )
            return response.text or ""
        except Exception as exc:  # noqa: BLE001 — transient API errors
            last_exc = exc
            logger.warning(
                "VLM vertex request failed (attempt %d/%d, model=%s): %s",
                attempt + 1, _MAX_RETRIES, model, exc,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_S * (2 ** attempt))
    raise ToolError(
        "vlm",
        f"vertex backend unavailable after {_MAX_RETRIES} attempts: "
        f"model={model}, error={last_exc}",
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_PROVIDERS = {
    "openrouter": _query_openrouter,
    "vertex": _query_vertex,
}


def _query(
    prompt: str,
    image: np.ndarray | None,
    images: list | None,
    provider: str | None,
    model: str | None,
) -> str:
    name = _resolve_provider(provider)
    fn = _PROVIDERS.get(name)
    if fn is None:
        raise ToolError(
            "vlm",
            f"unknown provider {name!r} (valid: {sorted(_PROVIDERS)}); set "
            f"GAP_VLM_PROVIDER (or GAP_LLM_PROVIDER — VLM inherits from "
            f"LLM when unset) or pass provider=",
        )
    return fn(prompt, _gather_images(image, images), model)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="vlm.query",
    summary="Free-form visual question answering via a hosted VLM.",
    tags=("perception",),
)
def query(
    prompt: str,
    image: np.ndarray | None = None,
    images: list | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> QueryResult:
    """Ask the configured VLM a free-form question, optionally about images.

    Args:
        prompt: Free-form question.
        image: Optional uint8 [H, W, 3] RGB context image.
        images: Optional additional context images (same dtype/shape).
        provider: Per-call provider override (``openrouter``/``vertex``).
        model: Per-call model override.

    Returns:
        ``{"text": <model response>}``.
    """
    return {"text": _query(prompt, image, images, provider, model)}


#: Appended to every ``query_yes_no`` prompt so the reply is machine-checkable.
#: The dev servicer relied on temperature-0 replies leading with "Yes,"/"No,"
#: and coerced with a bare ``"yes" in text.lower()``; without an explicit
#: instruction, models sometimes answer affirmatively in prose that contains
#: no literal "yes" ("...it appears to be a match.") which the substring
#: check silently mislabels as False. In the perceiving-objects safe gate
#: such a false "No" rejects a correct exterior pick and forces a degraded
#: single-view wrist fallback — the G1 cream-cheese failure mode.
_YES_NO_INSTRUCTION = (
    " Answer with the single word YES or NO first, then one short "
    "sentence of justification."
)

_YES_NO_WORD = re.compile(r"\b(yes|no)\b")


def _coerce_yes_no(text: str) -> bool:
    """First standalone yes/no word wins; legacy substring check as fallback."""
    m = _YES_NO_WORD.search(text.lower())
    if m:
        return m.group(1) == "yes"
    return "yes" in text.lower()


@tool(
    name="vlm.query_yes_no",
    summary="Yes/no visual question answering; coerces the model reply to a bool.",
    tags=("perception",),
)
def query_yes_no(
    prompt: str,
    image: np.ndarray | None = None,
    images: list | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> YesNoResult:
    """Ask the configured VLM a yes/no question, optionally about images.

    The prompt is suffixed with an explicit "answer YES or NO first"
    instruction (see :data:`_YES_NO_INSTRUCTION`) and ``answer`` is the
    first standalone ``yes``/``no`` word in the lowercased reply, falling
    back to the source servicer's verbatim ``"yes" in text.lower()``
    substring check when neither word appears.

    Returns:
        ``{"answer": <bool>, "text": <raw model response>}``.
    """
    text = _query(prompt + _YES_NO_INSTRUCTION, image, images, provider, model)
    return {"answer": _coerce_yes_no(text), "text": text}
