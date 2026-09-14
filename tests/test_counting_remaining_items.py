"""CPU contract for ``counting-remaining-items``: more / none / uncertain.

Every model-backed tool is a :class:`gap.testing.FakeContext` can. The
segmenter fake returns one scripted ``(score, points, box)`` list per call,
with masks sized to the image it was handed. That makes the weak-evidence
re-crop observable as a second, smaller image.
"""

from __future__ import annotations

import numpy as np
import pytest
from gap.testing import FakeContext

SIZE = 100


def _blob(center, n: int = 40, spread: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.uniform(-spread, spread, size=(n, 3))


def _rod(center, length: float = 0.12, width: float = 0.012, n: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    local = np.column_stack(
        (
            rng.uniform(-length / 2, length / 2, n),
            rng.uniform(-width / 2, width / 2, n),
            rng.uniform(-width / 2, width / 2, n),
        )
    )
    return np.asarray(center, dtype=np.float64) + local


def _camera(name: str) -> dict:
    return {
        "name": name,
        "rgb": np.zeros((SIZE, SIZE, 3), dtype=np.uint8),
        "depth": np.ones((SIZE, SIZE), dtype=np.float64),
        "intrinsics": np.eye(3),
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


def _observation(*names: str) -> dict:
    return {"cameras": [_camera(n) for n in names]}


def _ctx(calls: list[list[tuple]], hand=(0.0, 0.0, 2.0)) -> FakeContext:
    """``calls``: per segmenter call, a list of ``(score, points[, (x0, y0, x1, y1)])``."""
    queue = list(calls)
    clouds: list[np.ndarray] = []

    def segment(image, query, max_results):
        assert max_results == 0
        dets = queue.pop(0)
        h, w = np.asarray(image).shape[:2]
        masks = []
        for det in dets:
            mask = np.zeros((h, w), dtype=np.uint8)
            if len(det) > 2 and (h, w) == (SIZE, SIZE):
                x0, y0, x1, y1 = det[2]
                mask[y0:y1, x0:x1] = 1
            else:
                mask[:] = 1
            masks.append(mask)
            clouds.append(det[1])
        return {"masks": masks, "scores": [det[0] for det in dets]}

    def backproject(**_kw):
        return {"points": {"points": clouds.pop(0).astype(np.float32)}}

    return FakeContext(
        {
            "robot.get_ee_pose": {
                "pose": {
                    "position": dict(zip("xyz", hand, strict=True)),
                    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                }
            },
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": backproject,
        }
    )


@pytest.fixture(scope="module")
def count_remaining(skills_registry):
    return skills_registry.get("counting-remaining-items").canonical_scripts["count_remaining"].module


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_bundle_declares_its_contract(skills_registry):
    info = skills_registry.get("counting-remaining-items")
    assert info.kind == "skill"
    assert set(info.meta.exit_conditions) == {"more", "none", "uncertain"}
    assert set(info.meta.allowed_tools) == {
        "sam3.segment_text",
        "geometry.mask_to_world_points",
        "robot.get_ee_pose",
    }
    assert set(info.canonical_scripts) == {"count_remaining"}
    inputs = info.canonical_scripts["count_remaining"].schema.inputs
    assert {"observation", "object_description"} <= set(inputs)
    assert inputs["score_min"].default == pytest.approx(0.08)
    assert inputs["min_negative_views"].default == 2
    for limit in ("min_length_m", "max_length_m", "max_width_m", "min_aspect"):
        assert inputs[limit].default is None
    assert {"remaining", "status", "centers", "evidence"} <= set(
        info.canonical_scripts["count_remaining"].schema.outputs
    )


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


def test_a_strong_detection_in_the_first_camera_is_more(count_remaining):
    ctx = _ctx(
        [
            [
                (0.6, _blob((0.2, 0.3, 0.80))),
                (0.5, _blob((0.2, 0.31, 0.80), seed=1)),  # same item, 10 mm away
                (0.4, _blob((0.0, 0.3, 0.80), seed=2)),
            ]
        ]
    )
    out = count_remaining.run(ctx, _observation("overhead", "agentview"), "syringe")
    assert out["route"] == "more" and out["status"] == "more"
    assert out["remaining"] == 2
    assert len(out["centers"]) == 2
    assert out["evidence"][0]["camera"] == "overhead"
    # Positive evidence needs one view: agentview is never segmented.
    assert ctx.call_count("sam3.segment_text") == 1
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["query"] == "syringe"


def test_two_views_that_see_nothing_are_none(count_remaining):
    ctx = _ctx([[], []])
    out = count_remaining.run(ctx, _observation("agentview", "overhead"), "syringe")
    assert out == {
        "route": "none",
        "remaining": 0,
        "status": "none",
        "centers": [],
        "evidence": [{"negative_views": 2}],
    }


def test_one_negative_view_is_uncertain_not_none(count_remaining):
    ctx = _ctx([[]])
    out = count_remaining.run(ctx, _observation("overhead"), "syringe")
    assert out["route"] == "uncertain" and out["remaining"] == -1
    assert out["evidence"] == [{"negative_views": 1, "required_views": 2}]
    lowered = count_remaining.run(_ctx([[]]), _observation("overhead"), "syringe", min_negative_views=1)
    assert lowered["route"] == "none"


def test_camera_names_set_the_order_and_the_view_set(count_remaining):
    ctx = _ctx([[(0.5, _blob((0.2, 0.3, 0.8)))]])
    out = count_remaining.run(
        ctx, _observation("overhead", "agentview", "wrist"), "syringe", camera_names=["agentview", "overhead"]
    )
    assert out["route"] == "more"
    assert out["evidence"][0]["camera"] == "agentview"


# ---------------------------------------------------------------------------
# Weak evidence
# ---------------------------------------------------------------------------


def test_weak_evidence_is_resegmented_in_a_padded_crop(count_remaining):
    item = _blob((0.2, 0.3, 0.80))
    ctx = _ctx([[(0.05, item, (40, 40, 60, 60))], [(0.3, item)]])
    out = count_remaining.run(ctx, _observation("overhead", "agentview"), "syringe", recrop_padding_px=10)
    assert out["route"] == "more" and out["remaining"] == 1
    calls = ctx.calls_to("sam3.segment_text")
    assert len(calls) == 2
    assert np.asarray(calls[1].kwargs["image"]).shape[:2] == (40, 40)
    assert out["evidence"][0]["crop"] == (30, 30, 70, 70)


def test_evidence_that_stays_weak_is_uncertain(count_remaining):
    item = _blob((0.2, 0.3, 0.80))
    ctx = _ctx([[(0.05, item, (40, 40, 60, 60))], [(0.04, item)], [], []])
    out = count_remaining.run(ctx, _observation("overhead", "agentview"), "syringe")
    assert out["route"] == "uncertain" and out["remaining"] == -1
    assert out["evidence"] and out["evidence"][0]["score"] == pytest.approx(0.05)
    assert ctx.call_count("sam3.segment_text") == 3  # overhead, its re-crop, agentview


def test_recrop_can_be_disabled(count_remaining):
    ctx = _ctx([[(0.05, _blob((0.2, 0.3, 0.80)))], []])
    out = count_remaining.run(ctx, _observation("overhead", "agentview"), "syringe", recrop_weak=False)
    assert out["route"] == "uncertain"
    assert ctx.call_count("sam3.segment_text") == 2


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_exclusion_disc_height_limit_and_hand_remove_detections(count_remaining):
    obs = _observation("overhead", "agentview")
    near_aperture = _blob((0.55, 0.0, 0.80))
    out = count_remaining.run(
        _ctx([[(0.5, near_aperture)], []]),
        obs,
        "syringe",
        exclude_center={"x": 0.5, "y": 0.0, "z": 0.95},
    )
    assert out["route"] == "none"
    sticking_out = _blob((0.2, 0.3, 0.97))
    out = count_remaining.run(_ctx([[(0.5, sticking_out)], []]), obs, "syringe", max_z=0.95)
    assert out["route"] == "none"
    at_hand = _blob((0.2, 0.3, 0.80))
    ctx = _ctx([[(0.5, at_hand)], []], hand=(0.2, 0.3, 0.85))
    out = count_remaining.run(ctx, obs, "syringe")
    assert out["route"] == "none"
    assert ctx.call_count("robot.get_ee_pose") == 1


def test_hand_exclusion_zero_does_not_read_the_tcp(count_remaining):
    ctx = _ctx([[(0.5, _blob((0.2, 0.3, 0.80)))]], hand=(0.2, 0.3, 0.85))
    out = count_remaining.run(ctx, _observation("overhead", "agentview"), "syringe", hand_exclusion_m=0.0)
    assert out["route"] == "more"
    assert ctx.call_count("robot.get_ee_pose") == 0


def test_shape_limits_keep_rods_and_drop_specks(count_remaining):
    limits = {"min_length_m": 0.025, "max_length_m": 0.22, "max_width_m": 0.045, "min_aspect": 2.0}
    speck = _blob((0.2, 0.3, 0.80))
    out = count_remaining.run(_ctx([[(0.5, speck)], []]), _observation("overhead", "agentview"), "syringe", **limits)
    assert out["route"] == "none"
    rod = _rod((0.2, 0.3, 0.80))
    out = count_remaining.run(_ctx([[(0.5, rod)]]), _observation("overhead", "agentview"), "syringe", **limits)
    assert out["route"] == "more"
    assert out["evidence"][0]["extent"][0] > 0.09
