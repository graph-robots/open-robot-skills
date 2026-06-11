"""Tests for the vlm tool bundle — per-provider request shaping, all mocked.

No network, no GPU: the anthropic provider is exercised by monkeypatching the
SDK client class, the openai provider through ``httpx.MockTransport``, and
the vertex provider behind import guards (skipped when the [vertex] extra is
absent).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from gap.errors import ToolError
from gap.skills import load_skills
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

try:
    from anthropic import AnthropicVertex  # noqa: F401

    HAS_ANTHROPIC_VERTEX = True
except Exception:  # pragma: no cover - depends on installed extras
    HAS_ANTHROPIC_VERTEX = False


@pytest.fixture(scope="module")
def vlm():
    """The vlm bundle's tools module, loaded through the engine's loader."""
    reg = load_skills(ROOT, only=["vlm"])
    return reg.get("vlm").tools_module


@pytest.fixture()
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in ("GAP_VLM_PROVIDER", "GAP_VLM_MODEL", "GAP_VLM_BASE_URL",
                "GAP_VLM_API_KEY", "GAP_VLM_PROJECT_ID", "GAP_VLM_REGION"):
        monkeypatch.delenv(var, raising=False)


def _fake_anthropic_factory(calls: list[dict], text: str = "a red mug"):
    """A stand-in for ``anthropic.Anthropic`` capturing messages.create kwargs."""

    class _FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
            )

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = _FakeMessages()

    return _FakeAnthropic


def _decode_image_block(block: dict) -> np.ndarray:
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    raw = base64.b64decode(block["source"]["data"])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


# ---------------------------------------------------------------------------
# anthropic provider (default)
# ---------------------------------------------------------------------------


def test_anthropic_is_default_provider_with_base64_png_image(vlm, image, monkeypatch):
    import anthropic

    calls: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_factory(calls))

    out = vlm.query(prompt="What is on the table?", image=image)
    assert out == {"text": "a red mug"}

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model"] == vlm.DEFAULT_ANTHROPIC_MODEL == "claude-opus-4-8"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.0

    (message,) = kwargs["messages"]
    assert message["role"] == "user"
    image_block, text_block = message["content"]
    np.testing.assert_array_equal(_decode_image_block(image_block), image)
    assert text_block == {"type": "text", "text": "What is on the table?"}


def test_anthropic_model_env_override_and_multiple_images(vlm, image, monkeypatch):
    import anthropic

    calls: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_factory(calls))
    monkeypatch.setenv("GAP_VLM_MODEL", "claude-haiku-4-5")

    second = np.zeros((2, 3, 3), dtype=np.uint8)
    vlm.query(prompt="compare", image=image, images=[second])

    kwargs = calls[0]
    assert kwargs["model"] == "claude-haiku-4-5"
    content = kwargs["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]
    np.testing.assert_array_equal(_decode_image_block(content[0]), image)
    np.testing.assert_array_equal(_decode_image_block(content[1]), second)


def test_anthropic_text_only_query(vlm, monkeypatch):
    import anthropic

    calls: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_factory(calls))

    vlm.query(prompt="hello")
    content = calls[0]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "hello"}]


# ---------------------------------------------------------------------------
# openai provider (OpenAI-compatible chat completions via httpx)
# ---------------------------------------------------------------------------


def _mock_openai(vlm, monkeypatch, reply: str):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": reply}}]},
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))
    return captured


def test_openai_provider_request_shaping(vlm, image, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openai")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://vlm.test/v1")
    monkeypatch.setenv("GAP_VLM_API_KEY", "sk-test")
    monkeypatch.setenv("GAP_VLM_MODEL", "gcp/google/gemini-3-flash-preview")
    captured = _mock_openai(vlm, monkeypatch, reply="two cups")

    out = vlm.query(prompt="What objects are on the table?", image=image)
    assert out == {"text": "two cups"}

    assert captured["url"] == "http://vlm.test/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"

    payload = captured["payload"]
    assert payload["model"] == "gcp/google/gemini-3-flash-preview"
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.0

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What objects are on the table?"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")), image,
    )


def test_openai_provider_requires_base_url_and_model(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openai")
    with pytest.raises(ToolError, match="GAP_VLM_BASE_URL"):
        vlm.query(prompt="q")

    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://vlm.test/v1")
    with pytest.raises(ToolError, match="GAP_VLM_MODEL"):
        vlm.query(prompt="q")


def test_openai_provider_backend_failure_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openai")
    monkeypatch.setenv("GAP_VLM_BASE_URL", "http://vlm.test/v1")
    monkeypatch.setenv("GAP_VLM_MODEL", "m")
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)

    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, json={"error": "overloaded"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(vlm, "_http_client", lambda: httpx.Client(transport=transport))

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="q")
    assert len(attempts) == 3


# ---------------------------------------------------------------------------
# vertex provider (import-guarded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANTHROPIC_VERTEX, reason="anthropic [vertex] extra absent")
def test_vertex_provider_routes_claude_models_to_anthropic_vertex(
    vlm, image, monkeypatch,
):
    import anthropic

    calls: list[dict] = []
    init_kwargs: dict = {}

    class _FakeVertex:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="vertex says hi")],
            )

    monkeypatch.setattr(anthropic, "AnthropicVertex", _FakeVertex)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")
    monkeypatch.setenv("GAP_VLM_REGION", "us-central1")

    out = vlm.query(prompt="hi", image=image)
    assert out == {"text": "vertex says hi"}
    assert init_kwargs == {"project_id": "test-project", "region": "us-central1"}
    assert calls[0]["model"] == "claude-opus-4-8"
    content = calls[0]["messages"][0]["content"]
    np.testing.assert_array_equal(_decode_image_block(content[0]), image)
    assert content[1] == {"type": "text", "text": "hi"}


def test_vertex_provider_routes_gemini_models_to_genai(vlm, image, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, *, model, contents, config=None):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(text="gemini says hi")

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    out = vlm.query(prompt="hi", image=image)
    assert out == {"text": "gemini says hi"}
    assert captured["client_kwargs"] == {
        "vertexai": True, "project": "test-project", "location": "global",
    }
    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["contents"][0] == "hi"
    # Deterministic decoding — parity with the dev servicer's production
    # path (temperature 0.0, max_tokens 1024).
    assert captured["config"].temperature == 0.0
    assert captured["config"].max_output_tokens == 1024


def test_vertex_gemini_retries_transient_failures(vlm, image, monkeypatch):
    """The vertex Gemini path retries like the openai path (3 attempts)."""
    pytest.importorskip("google.genai")
    from google import genai

    attempts: list[int] = []

    class _FlakyClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, *, model, contents, config=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("503 transient")
            return SimpleNamespace(text="third time lucky")

    monkeypatch.setattr(genai, "Client", _FlakyClient)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    out = vlm.query(prompt="hi")
    assert out == {"text": "third time lucky"}
    assert len(attempts) == 3


def test_vertex_gemini_exhausted_retries_raise_tool_error(vlm, monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    class _DeadClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._generate)

        def _generate(self, **kwargs):
            raise RuntimeError("permanently overloaded")

    monkeypatch.setattr(genai, "Client", _DeadClient)
    monkeypatch.setattr(vlm, "_BACKOFF_S", 0.0)
    monkeypatch.setenv("GAP_VLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_VLM_MODEL", "gemini-3-flash-preview")
    monkeypatch.setenv("GAP_VLM_PROJECT_ID", "test-project")

    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        vlm.query(prompt="hi")


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_provider_kwarg_overrides_env(vlm, monkeypatch):
    import anthropic

    calls: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_factory(calls))
    # Env says openai (and is unconfigured, so it would fail) — the kwarg wins.
    monkeypatch.setenv("GAP_VLM_PROVIDER", "openai")

    out = vlm.query(prompt="q", provider="anthropic")
    assert out == {"text": "a red mug"}
    assert len(calls) == 1


def test_unknown_provider_raises_tool_error(vlm, monkeypatch):
    monkeypatch.setenv("GAP_VLM_PROVIDER", "bedrock")
    with pytest.raises(ToolError, match="unknown provider 'bedrock'"):
        vlm.query(prompt="q")


def test_invalid_image_rejected(vlm):
    bad = np.zeros((4, 6), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError, match=r"uint8 \[H, W, 3\]"):
        vlm.query(prompt="q", image=bad)


# ---------------------------------------------------------------------------
# Yes/no coercion (first standalone yes/no word; legacy substring fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", True),
        ("yes.", True),
        ("YES — clearly visible.", True),
        ("The answer is yes", True),
        ("Yes, the object matches the description.", True),
        ("Eyes on the table", True),  # legacy substring fallback quirk
        ("No", False),
        ("no.", False),
        ("Absolutely not", False),
        ("I cannot tell", False),
        ("", False),
        # First standalone word wins — the legacy substring check would
        # mislabel both of these (the G1 verify-gate failure mode):
        ("No, although the label literally says YES on it.", False),
        ("NO. The item appears to be a small book.", False),
    ],
)
def test_query_yes_no_coercion(vlm, monkeypatch, text, expected):
    import anthropic

    calls: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_factory(calls, text=text))

    out = vlm.query_yes_no(prompt="Is the sauce in the basket?")
    assert out == {"answer": expected, "text": text}


def test_query_yes_no_appends_explicit_instruction(vlm, monkeypatch):
    """query_yes_no must elicit a parseable YES/NO-first reply.

    Without the instruction (and at temperature > 0) models answer
    affirmatively in prose with no literal "yes" — which the coercion
    mislabels as False. A false "No" from the perceiving-objects verify
    gate rejects a correct exterior pick and forces the degraded
    single-view wrist fallback.
    """
    import anthropic

    calls: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic_factory(calls, text="Yes."))

    vlm.query_yes_no(prompt="Is this a cream cheese box?")

    (message,) = calls[0]["messages"]
    (text_block,) = message["content"]
    assert text_block["text"].startswith("Is this a cream cheese box?")
    assert "YES or NO first" in text_block["text"]
