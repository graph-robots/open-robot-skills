"""Tests for the molmo tool bundle — mocked vLLM backend, no network.

Covers the four point-output formats ported verbatim from the source
servicer, point→pixel conversion, the unset-base-URL error message, and the
yes/no coercion.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from gap.errors import ToolError
from gap.skills import load_skills
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def molmo():
    """The molmo bundle's tools module, loaded through the engine."""
    reg = load_skills(ROOT, only=["molmo"])
    return reg.get("molmo").tools_module


@pytest.fixture()
def image() -> np.ndarray:
    # 100 high x 200 wide — pixel-conversion assertions depend on this.
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GAP_MOLMO_BASE_URL", raising=False)
    monkeypatch.delenv("GAP_MOLMO_MODEL", raising=False)


def _mock_backend(molmo, monkeypatch, reply: str) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": reply}}]},
        )

    monkeypatch.setenv("GAP_MOLMO_BASE_URL", "http://molmo.test/v1")
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        molmo, "_http_client", lambda: httpx.Client(transport=transport),
    )
    return captured


# ---------------------------------------------------------------------------
# The four point-output formats (regexes ported verbatim)
# ---------------------------------------------------------------------------


def test_parse_points_molmo2_coords_format(molmo):
    # <points coords="type idx x y idx x y ...">: triplets after the type
    # indicator, normalized 0-1000.
    text = '<points coords="0 1 500 600 2 100 200">two cups</points>'
    points, scale = molmo._parse_points(text)
    assert points == [(500.0, 600.0), (100.0, 200.0)]
    assert scale == 1000.0


def test_parse_points_molmo1_point_tags(molmo):
    text = 'Sure! <point x="45.2" y="60.1" alt="red cup">red cup</point>'
    points, scale = molmo._parse_points(text)
    assert points == [(45.2, 60.1)]
    assert scale == 100.0


def test_parse_points_molmo1_multiple_tags_and_range_filter(molmo):
    text = '<point x="10" y="20"> and <point x="500" y="20">'  # 500 > 100 dropped
    points, scale = molmo._parse_points(text)
    assert points == [(10.0, 20.0)]
    assert scale == 100.0


def test_parse_points_legacy_points_tag(molmo):
    text = '<points x1="10.5" y1="20.5" x2="30" y2="40" alt="cups">cups</points>'
    points, scale = molmo._parse_points(text)
    assert points == [(10.5, 20.5), (30.0, 40.0)]
    assert scale == 100.0


def test_parse_points_fallback_plain_pairs(molmo):
    text = "The object is located at 55.5, 70.2 in the image."
    points, scale = molmo._parse_points(text)
    assert points == [(55.5, 70.2)]
    assert scale == 100.0


def test_parse_points_nothing_parseable(molmo):
    points, scale = molmo._parse_points("I cannot find that object.")
    assert points == []
    assert scale == 100.0


# ---------------------------------------------------------------------------
# point_prompt: prompt shaping + pixel conversion
# ---------------------------------------------------------------------------


def test_point_prompt_converts_normalized_point_to_pixels(molmo, image, monkeypatch):
    captured = _mock_backend(
        molmo, monkeypatch, '<point x="50" y="25">red button</point>',
    )

    out = molmo.point_prompt(image=image, query="red button")
    # 200 wide, 100 high; Molmo1 scale is 100.
    assert out == {"pixel_x": 100.0, "pixel_y": 25.0, "found": True}

    assert captured["url"] == "http://molmo.test/v1/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == molmo.DEFAULT_MODEL == "allenai/Molmo2-8B"
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.0
    assert payload["stop"] == ["<|endoftext|>"]

    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Point at red button"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")), image,
    )


def test_point_prompt_molmo2_scale_conversion(molmo, image, monkeypatch):
    _mock_backend(molmo, monkeypatch, '<points coords="0 1 500 600">cup</points>')
    out = molmo.point_prompt(image=image, query="cup")
    # Molmo2 scale is 1000: x = 500/1000*200, y = 600/1000*100.
    assert out == {"pixel_x": 100.0, "pixel_y": 60.0, "found": True}


def test_point_prompt_not_found(molmo, image, monkeypatch):
    _mock_backend(molmo, monkeypatch, "I cannot find that object.")
    out = molmo.point_prompt(image=image, query="unicorn")
    assert out == {"pixel_x": 0.0, "pixel_y": 0.0, "found": False}


def test_model_env_override(molmo, image, monkeypatch):
    captured = _mock_backend(molmo, monkeypatch, '<point x="1" y="1">')
    monkeypatch.setenv("GAP_MOLMO_MODEL", "allenai/Molmo-7B-D-0924")
    molmo.point_prompt(image=image, query="cup")
    assert captured["payload"]["model"] == "allenai/Molmo-7B-D-0924"


# ---------------------------------------------------------------------------
# Unset base URL: self-hosting requirement + the API alternative
# ---------------------------------------------------------------------------


def test_unset_base_url_error_explains_self_hosting(molmo, image):
    with pytest.raises(ToolError) as excinfo:
        molmo.point_prompt(image=image, query="cup")
    message = str(excinfo.value)
    assert "GAP_MOLMO_BASE_URL" in message
    assert "self-hosted vLLM" in message
    assert "gemini-er" in message  # points at the zero-GPU alternative


def test_backend_failure_raises_tool_error(molmo, image, monkeypatch):
    monkeypatch.setenv("GAP_MOLMO_BASE_URL", "http://molmo.test/v1")
    monkeypatch.setattr(molmo, "_BACKOFF_S", 0.0)
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    monkeypatch.setattr(
        molmo, "_http_client", lambda: httpx.Client(transport=transport),
    )
    with pytest.raises(ToolError, match="unavailable after 3 attempts"):
        molmo.query(prompt="q")


def test_invalid_image_rejected(molmo, monkeypatch):
    monkeypatch.setenv("GAP_MOLMO_BASE_URL", "http://molmo.test/v1")
    with pytest.raises(ValueError, match=r"uint8 \[H, W, 3\]"):
        molmo.point_prompt(image=np.zeros((4, 6, 4), dtype=np.uint8), query="cup")


# ---------------------------------------------------------------------------
# query / query_yes_no
# ---------------------------------------------------------------------------


def test_query_returns_text_without_image(molmo, monkeypatch):
    captured = _mock_backend(molmo, monkeypatch, "A tidy kitchen scene.")
    out = molmo.query(prompt="Describe the scene")
    assert out == {"text": "A tidy kitchen scene."}
    content = captured["payload"]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "Describe the scene"}]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Yes", True),
        ("yes, it is inside the basket.", True),
        ("Eyes are visible", True),  # substring quirk, ported verbatim
        ("No", False),
        ("It is not.", False),
        ("", False),
    ],
)
def test_query_yes_no_coercion(molmo, image, monkeypatch, reply, expected):
    _mock_backend(molmo, monkeypatch, reply)
    out = molmo.query_yes_no(prompt="Is the sauce in the basket?", image=image)
    assert out == {"answer": expected, "text": reply}
