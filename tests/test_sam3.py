"""sam3 bundle: signature/schema units + GPU smoke.

Collection must work without torch/sam3 installed — all heavy imports stay
behind ``pytest.importorskip`` inside the gpu-marked tests.
"""

from __future__ import annotations

import numpy as np
import pytest

EXPECTED_TOOLS = {
    "sam3.segment_text",
    "sam3.segment_point",
    "sam3.segment_box",
    "sam3.tracker_init",
    "sam3.tracker_update",
    "sam3.tracker_close",
}


def test_all_tools_registered(tool_registry):
    for name in EXPECTED_TOOLS:
        assert name in tool_registry


def test_segment_text_schema(tool_registry):
    schema = tool_registry.get("sam3.segment_text").schema
    assert set(schema.inputs) == {"image", "query", "max_results"}
    assert schema.inputs["image"].required
    assert schema.inputs["query"].required
    assert schema.inputs["max_results"].required is False
    assert schema.inputs["max_results"].default == 5
    assert set(schema.outputs) == {"masks", "scores", "boxes"}


def test_segment_box_schema(tool_registry):
    schema = tool_registry.get("sam3.segment_box").schema
    assert set(schema.inputs) == {"image", "box", "pixel_x", "pixel_y", "use_point"}
    assert schema.inputs["use_point"].default is False
    assert set(schema.outputs) == {"masks", "scores"}


def test_tracker_schemas(tool_registry):
    init = tool_registry.get("sam3.tracker_init").schema
    assert set(init.inputs) == {
        "image", "text", "box", "pixel_x", "pixel_y", "use_point", "object_name",
    }
    assert set(init.outputs) == {
        "tracker_id", "initial_mask", "initial_box", "score", "object_present",
    }

    update = tool_registry.get("sam3.tracker_update").schema
    assert set(update.inputs) == {"tracker_id", "image"}
    assert set(update.outputs) == {"mask", "box", "confidence", "object_present"}

    close = tool_registry.get("sam3.tracker_close").schema
    assert set(close.inputs) == {"tracker_id"}
    assert set(close.outputs) == {"closed"}


def test_invalid_image_shape_rejected_before_model_load(tool_registry, skills_registry):
    """The RGB shape check fires in _to_pil — but only after the model
    singleton loads, so exercise the helper directly (no GPU needed)."""
    from gap.errors import ToolError

    mod = skills_registry.get("sam3").tools_module
    with pytest.raises(ToolError):
        mod._to_pil(np.zeros((4, 4), dtype=np.uint8))
    pil = mod._to_pil(np.zeros((4, 4, 3), dtype=np.uint8))
    assert pil.size == (4, 4)


def _smoke_image() -> np.ndarray:
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    img[32:96, 32:96] = (220, 30, 30)  # red square on white
    return img


@pytest.mark.gpu
def test_segment_text_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    out = tool_registry.invoke(
        "sam3.segment_text", image=_smoke_image(), query="red square"
    )
    assert set(out) == {"masks", "scores", "boxes"}
    assert len(out["masks"]) == len(out["scores"]) == len(out["boxes"])
    for mask in out["masks"]:
        assert mask.dtype == np.uint8
        assert mask.shape == (128, 128)
        assert set(np.unique(mask)) <= {0, 255}
    # Scores are sorted best-first.
    assert out["scores"] == sorted(out["scores"], reverse=True)


@pytest.mark.gpu
def test_segment_box_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    out = tool_registry.invoke(
        "sam3.segment_box",
        image=_smoke_image(),
        box={"x1": 28.0, "y1": 28.0, "x2": 100.0, "y2": 100.0},
    )
    assert out["masks"], "box prompt on a clear square should segment something"
    assert out["masks"][0].shape == (128, 128)


@pytest.mark.gpu
def test_tracker_roundtrip_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sam3")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    frame0 = _smoke_image()
    init = tool_registry.invoke(
        "sam3.tracker_init",
        image=frame0,
        box={"x1": 28.0, "y1": 28.0, "x2": 100.0, "y2": 100.0},
        object_name="red_square",
    )
    if not init["object_present"]:
        pytest.skip("tracker did not lock onto the synthetic square")

    # Shift the square a few pixels and track it.
    frame1 = np.full_like(frame0, 255)
    frame1[36:100, 36:100] = (220, 30, 30)
    upd = tool_registry.invoke(
        "sam3.tracker_update", tracker_id=init["tracker_id"], image=frame1
    )
    assert set(upd) == {"mask", "box", "confidence", "object_present"}

    closed = tool_registry.invoke("sam3.tracker_close", tracker_id=init["tracker_id"])
    assert closed["closed"] is True
    # Idempotent close.
    closed2 = tool_registry.invoke("sam3.tracker_close", tracker_id=init["tracker_id"])
    assert closed2["closed"] is False
