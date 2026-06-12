"""grounding-dino bundle: signature/schema units + GPU smoke."""

from __future__ import annotations

import numpy as np
import pytest


def test_detect_registered(tool_registry):
    assert "grounding-dino.detect" in tool_registry
    desc = tool_registry.get("grounding-dino.detect")
    assert desc.tags == ("perception",)


def test_detect_schema(tool_registry):
    schema = tool_registry.get("grounding-dino.detect").schema
    assert set(schema.inputs) == {"image", "query", "box_threshold", "text_threshold"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["box_threshold"].default == pytest.approx(0.20)
    assert schema.inputs["text_threshold"].default == pytest.approx(0.20)
    assert set(schema.outputs) == {"detections"}


@pytest.mark.gpu
def test_detect_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    img[40:120, 40:120] = (200, 30, 30)
    out = tool_registry.invoke(
        "grounding-dino.detect", image=img, query="red square"
    )
    assert set(out) == {"detections"}
    for det in out["detections"]:
        assert set(det) == {"box", "label", "score"}
        assert 0.0 <= det["score"] <= 1.0
        box = det["box"]
        assert box["x2"] >= box["x1"] and box["y2"] >= box["y1"]
