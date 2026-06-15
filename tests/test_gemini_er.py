"""Tests for the gemini-er tool bundle — mocked google-genai, no network.

The SDK is faked through the bundle's ``_import_genai`` seam, so these tests
run whether or not google-genai is installed (it stays a lazy import).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gap_core.errors import ToolError
from gap.skills import load_skills

ROOT = Path(__file__).resolve().parents[1]


def _has_genai() -> bool:
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        return False


@pytest.fixture(scope="module")
def ger():
    """The gemini-er bundle's tools module, loaded through the engine."""
    reg = load_skills(ROOT, only=["gemini-er"])
    return reg.get("gemini-er").tools_module


@pytest.fixture()
def image() -> np.ndarray:
    # 480 high x 640 wide — pixel-conversion assertions depend on this.
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _fake_genai(response_text: str, captured: dict):
    """Build fake (genai, types) modules returning a canned response."""

    class _Models:
        def generate_content(self, *, model, contents):
            captured["model"] = model
            captured["contents"] = contents
            return SimpleNamespace(text=response_text)

    class _Client:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = _Models()

    def _from_bytes(*, data, mime_type):
        part = SimpleNamespace(data=data, mime_type=mime_type)
        captured.setdefault("parts", []).append(part)
        return part

    genai = SimpleNamespace(Client=_Client)
    types_mod = SimpleNamespace(Part=SimpleNamespace(from_bytes=_from_bytes))
    return genai, types_mod


def _patch(ger, monkeypatch, response_text: str) -> dict:
    captured: dict = {}
    monkeypatch.setattr(
        ger, "_import_genai", lambda: _fake_genai(response_text, captured),
    )
    return captured


# ---------------------------------------------------------------------------
# Detection parsing + pixel conversion
# ---------------------------------------------------------------------------


def test_detect_converts_box_2d_to_pixels(ger, image, monkeypatch):
    captured = _patch(ger, monkeypatch, (
        "```json\n"
        '[{"box_2d": [100, 200, 500, 600], "label": "cup"}]\n'
        "```"
    ))

    out = ger.detect(image=image, query="cup")

    (det,) = out["detections"]
    # [ymin, xmin, ymax, xmax] = [100, 200, 500, 600] on a 640x480 frame:
    assert det["box"] == {
        "x1": pytest.approx(200 / 1000 * 640),  # 128.0
        "y1": pytest.approx(100 / 1000 * 480),  # 48.0
        "x2": pytest.approx(600 / 1000 * 640),  # 384.0
        "y2": pytest.approx(500 / 1000 * 480),  # 240.0
    }
    assert det["label"] == "cup"
    assert det["score"] == 1.0  # default when the model reports no score

    # Request shaping: PNG part + detection prompt, default model.
    assert captured["model"] == ger.DEFAULT_MODEL == "gemini-robotics-er-1.5-preview"
    part, prompt = captured["contents"]
    assert part.mime_type == "image/png"
    assert part.data.startswith(b"\x89PNG")
    assert '"cup"' in prompt and "box_2d" in prompt


def test_detect_multiple_boxes_scores_and_clamping(ger, image, monkeypatch):
    _patch(ger, monkeypatch, (
        '[{"box_2d": [0, 0, 1200, 1300], "label": "table", "score": 0.4},'
        ' {"box_2d": [250, 250, 750, 750], "label": "mug", "score": 0.9}]'
    ))

    out = ger.detect(image=image, query="mug")
    dets = out["detections"]
    assert [d["label"] for d in dets] == ["table", "mug"]
    assert [d["score"] for d in dets] == [0.4, 0.9]

    # Out-of-range normalized coords clamp to the frame.
    assert dets[0]["box"]["x2"] == 640.0
    assert dets[0]["box"]["y2"] == 480.0

    # The consumer contract: best detection by max score.
    best = max(dets, key=lambda d: d["score"])
    assert best["label"] == "mug"


def test_detect_model_env_override(ger, image, monkeypatch):
    captured = _patch(ger, monkeypatch, "[]")
    monkeypatch.setenv("GAP_GEMINI_ER_MODEL", "gemini-robotics-er-2.0")
    ger.detect(image=image, query="cup")
    assert captured["model"] == "gemini-robotics-er-2.0"


# ---------------------------------------------------------------------------
# No-detection and malformed output never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response_text",
    [
        "[]",
        "",
        "I do not see any such object in the image.",
        "```json\n[]\n```",
    ],
)
def test_detect_no_detection_returns_empty_list(ger, image, monkeypatch, response_text):
    _patch(ger, monkeypatch, response_text)
    assert ger.detect(image=image, query="unicorn") == {"detections": []}


@pytest.mark.parametrize(
    "response_text",
    [
        '[{"box_2d": [100, 200, 500',          # truncated JSON
        '{"box_2d": [1, 2, 3, 4]}',            # object, not array
        '["not-a-dict", 42]',                  # wrong entry types
        '[{"label": "cup"}]',                  # missing box_2d
        '[{"box_2d": [1, 2, 3], "label": "cup"}]',        # wrong arity
        '[{"box_2d": ["a", "b", "c", "d"], "label": "x"}]',  # non-numeric
    ],
)
def test_detect_tolerates_malformed_json(ger, image, monkeypatch, response_text):
    _patch(ger, monkeypatch, response_text)
    assert ger.detect(image=image, query="cup") == {"detections": []}


def test_detect_skips_bad_entries_keeps_good_ones(ger, image, monkeypatch):
    _patch(ger, monkeypatch, (
        '[{"label": "no box"}, {"box_2d": [0, 0, 500, 500]}]'
    ))
    out = ger.detect(image=image, query="cup")
    (det,) = out["detections"]
    assert det["label"] == "cup"  # falls back to the query when label missing


# ---------------------------------------------------------------------------
# Input validation + dependency hint
# ---------------------------------------------------------------------------


def test_detect_rejects_non_rgb_image(ger, monkeypatch):
    _patch(ger, monkeypatch, "[]")
    with pytest.raises(ValueError, match=r"uint8 \[H, W, 3\]"):
        ger.detect(image=np.zeros((4, 6), dtype=np.float32), query="cup")


@pytest.mark.skipif(_has_genai(), reason="google-genai installed; import cannot fail")
def test_missing_genai_dependency_hint(ger, image):
    with pytest.raises(ToolError, match=r"open-robot-skills\[gemini-er\]"):
        ger.detect(image=image, query="cup")


def test_genai_errors_wrapped_as_tool_error(ger, image, monkeypatch):
    class _Boom:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self._raise)

        def _raise(self, **kwargs):
            raise RuntimeError("quota exceeded")

    genai = SimpleNamespace(Client=_Boom)
    types_mod = SimpleNamespace(
        Part=SimpleNamespace(from_bytes=lambda **kw: object()),
    )
    monkeypatch.setattr(ger, "_import_genai", lambda: (genai, types_mod))

    with pytest.raises(ToolError, match="quota exceeded"):
        ger.detect(image=image, query="cup")
