"""``verifying-placement``'s multi-view mode, and proof that its default is unchanged.

The sharps-disposal perception graph grew a two-view post-release check that
separates ``verified`` / ``clear`` / ``blocked`` / ``uncertain``. It was folded
into ``verify_placement`` behind ``views`` (default ``None``). Two claims are
checked here:

* **Opt-in.** With ``views`` left unset, the script at ``752dfbd`` (before the
  fold) and the script now make the same tool calls, with the same keyword
  values in the same order, and return the same value or raise the same error,
  over a grid of inputs. This follows ``test_promotion_is_behaviour_preserving``.
* **Multi-view behaviour.** Blocked, clear, uncertain (too few views, and weak
  above-rim evidence), verified, hand exclusion, the flat-lid skip and the
  aperture crop.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[1]
REL = "skills/verifying-placement/scripts/verify_placement.py"
BEFORE_REF = "752dfbd"

APERTURE = {"x": 0.5, "y": 0.0, "z": 0.95}
RIM_Z = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _at_ref(rel: str, ref: str) -> str:
    done = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        pytest.skip(f"{ref} is not in this checkout: {done.stderr.strip()}")
    return done.stdout


def _blob(center, n: int = 40, spread: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.uniform(-spread, spread, size=(n, 3))


def _camera(name: str = "overhead", size: int = 100) -> dict:
    """A pinhole 0.95 m below the aperture looking up +z: the aperture box projects inside."""
    return {
        "name": name,
        "rgb": np.zeros((size, size, 3), dtype=np.uint8),
        "depth": np.ones((size, size), dtype=np.float64),
        "intrinsics": np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        "pose": {
            "position": {"x": 0.5, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


def _pose(position) -> dict:
    x, y, z = position
    return {
        "pose": {
            "position": {"x": x, "y": y, "z": z},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
    }


def _multi_ctx(per_view: list[list[tuple[float, np.ndarray]]], hand=(0.0, 0.0, 2.0)) -> FakeContext:
    """One ``(score, points)`` list per segmenter call, consumed in camera order."""
    views = list(per_view)
    clouds: list[np.ndarray] = []

    def segment(image, query, max_results):
        dets = views.pop(0)
        clouds.extend(points for _, points in dets)
        h, w = np.asarray(image).shape[:2]
        return {
            "masks": [np.ones((h, w), dtype=np.uint8) for _ in dets],
            "scores": [score for score, _ in dets],
        }

    def backproject(**_kw):
        return {"points": {"points": clouds.pop(0).astype(np.float32)}}

    return FakeContext(
        {
            "robot.get_ee_pose": _pose(hand),
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": backproject,
        }
    )


def _observation(*names: str) -> dict:
    return {"cameras": [_camera(n) for n in names]}


@pytest.fixture(scope="module")
def verify_placement(skills_registry):
    return skills_registry.get("verifying-placement").canonical_scripts["verify_placement"].module


def _run(module, ctx, observation, **kw):
    return module.run(ctx, observation, "syringe", APERTURE, RIM_Z, views=["overhead", "agentview"], **kw)


# ---------------------------------------------------------------------------
# Default path == the script before the fold
# ---------------------------------------------------------------------------


def _default_scenario(score, center, n, camera_present):
    masks = [] if score is None else [np.ones((8, 8), dtype=np.uint8)]
    scores = [] if score is None else [score]
    points = _blob(center, n=n)
    observation = {
        "cameras": [
            {
                "name": "overhead" if camera_present else "wrist",
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "depth": np.ones((8, 8)),
                "intrinsics": np.eye(3),
                "pose": {"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}},
            }
        ]
    }
    responses = {
        "sam3.segment_text": {"masks": masks, "scores": scores},
        "geometry.mask_to_world_points": {"points": {"points": points.astype(np.float32)}},
        "robot.get_ee_pose": _pose((0.5, 0.0, 1.0)),
    }
    return observation, responses


DEFAULT_GRID = list(
    itertools.product(
        [None, 0.01, 0.5],
        [(0.5, 0.0, 0.90), (0.5, 0.0, 0.97), (0.75, 0.0, 0.90)],
        [40, 3],
        [True, False],
    )
)


def _drive(module, observation, responses):
    ctx = FakeContext(responses)
    try:
        result, error = module.run(ctx, observation, "syringe", APERTURE, RIM_Z), None
    except Exception as exc:  # noqa: BLE001 -- the error is part of the comparison
        result, error = None, f"{type(exc).__name__}: {exc}"
    calls = [(r.tool, json.dumps(r.kwargs, sort_keys=True, default=str)) for r in ctx.calls]
    return {"calls": calls, "result": result, "error": error}


@pytest.mark.parametrize("score,center,n,camera_present", DEFAULT_GRID)
def test_default_path_is_the_script_before_the_fold(score, center, n, camera_present):
    path = Path(tempfile.mkdtemp()) / "verify_placement_before.py"
    path.write_text(_at_ref(REL, BEFORE_REF))
    before = _load_path(path, "verify_placement_before")
    after = _load_path(ROOT / REL, "verify_placement_after")
    got_before = _drive(before, *_default_scenario(score, center, n, camera_present))
    got_after = _drive(after, *_default_scenario(score, center, n, camera_present))
    assert got_before == got_after
    assert all(tool != "robot.get_ee_pose" for tool, _ in got_after["calls"])


# ---------------------------------------------------------------------------
# Multi-view behaviour
# ---------------------------------------------------------------------------


def test_strong_detection_above_the_rim_is_blocked(verify_placement):
    above = _blob((0.51, 0.0, 0.99))
    ctx = _multi_ctx([[(0.5, above)], [(0.4, above)]])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "blocked" and out["status"] == "blocked"
    assert out["verified"] is False
    assert len(out["observations"]) == 2
    assert {o["camera"] for o in out["observations"]} == {"overhead", "agentview"}
    assert out["ignored_robot_masks"] == 0


def test_every_mask_is_judged_inside_an_aperture_crop(verify_placement):
    ctx = _multi_ctx([[], []])
    _run(verify_placement, ctx, _observation("overhead", "agentview"))
    calls = ctx.calls_to("sam3.segment_text")
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["max_results"] == 0
        h, w = np.asarray(call.kwargs["image"]).shape[:2]
        assert 12 <= h < 100 and 12 <= w < 100


def test_nothing_seen_in_two_views_is_clear_not_verified(verify_placement):
    ctx = _multi_ctx([[], []])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "clear" and out["status"] == "clear"
    assert out["verified"] is False
    assert "not established by absence" in out["evidence"]
    assert out["observations"] == []


def test_strong_detections_below_the_rim_are_verified(verify_placement):
    below = _blob((0.5, 0.0, 0.90))
    ctx = _multi_ctx([[(0.5, below)], []])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "verified" and out["verified"] is True
    assert len(out["observations"]) == 1


def test_top_within_the_tolerance_above_the_rim_is_not_blocked(verify_placement):
    # +4 mm above the rim: the s7 marginal jams that the 10 mm margin re-reads.
    marginal = _blob((0.5, 0.0, RIM_Z + 0.004), spread=0.003)
    ctx = _multi_ctx([[(0.5, marginal)], []])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "verified"
    ctx = _multi_ctx([[(0.5, marginal)], []])
    strict = _run(verify_placement, ctx, _observation("overhead", "agentview"), above_rim_tolerance_m=0.0)
    assert strict["route"] == "blocked"


def test_too_few_calibrated_views_is_uncertain(verify_placement):
    ctx = _multi_ctx([[(0.5, _blob((0.5, 0.0, 0.90)))]])
    out = _run(verify_placement, ctx, _observation("overhead"))
    assert out["route"] == "uncertain" and out["verified"] is False
    assert "2 calibrated inspection views are required, 1 available" in out["evidence"]


def test_a_camera_that_cannot_see_the_aperture_is_not_a_view(verify_placement):
    behind = _camera("agentview")
    behind["pose"] = {
        "position": {"x": 0.5, "y": 0.0, "z": 3.0},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }
    ctx = _multi_ctx([[]])
    out = _run(verify_placement, ctx, {"cameras": [_camera("overhead"), behind]})
    assert out["route"] == "uncertain"
    assert ctx.call_count("sam3.segment_text") == 1


def test_weak_above_rim_evidence_is_uncertain(verify_placement):
    above = _blob((0.5, 0.0, 0.99))
    ctx = _multi_ctx([[(0.02, above)], []])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "uncertain"
    assert "weak above-rim evidence" in out["evidence"]
    assert out["observations"][0]["score"] == pytest.approx(0.02)


def test_masks_at_the_live_hand_are_ignored(verify_placement):
    above = _blob((0.5, 0.0, 0.99))
    ctx = _multi_ctx([[(0.5, above)], [(0.5, above)]], hand=(0.5, 0.0, 1.05))
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "clear"
    assert out["ignored_robot_masks"] == 2
    assert ctx.call_count("robot.get_ee_pose") == 1
    # The same masks with the exclusion disabled are a jam, and the TCP is not read.
    ctx = _multi_ctx([[(0.5, above)], [(0.5, above)]], hand=(0.5, 0.0, 1.05))
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"), hand_exclusion_m=0.0)
    assert out["route"] == "blocked"
    assert ctx.call_count("robot.get_ee_pose") == 0


def test_flat_lid_masks_are_skipped(verify_placement):
    rng = np.random.default_rng(1)
    lid = np.column_stack(
        (
            0.5 + rng.uniform(-0.03, 0.03, 60),
            rng.uniform(-0.03, 0.03, 60),
            RIM_Z + rng.uniform(-0.0005, 0.0005, 60),
        )
    )
    # Lift a few lid points 15 mm: under the flat-lid fraction, still not a protrusion.
    lid[:4, 2] = RIM_Z + 0.015
    ctx = _multi_ctx([[(0.5, lid)], [(0.5, lid)]])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "clear"
    assert out["observations"] == []


def test_detections_off_the_aperture_are_not_judged(verify_placement):
    beside = _blob((0.5 + 0.105, 0.0, 0.99))
    ctx = _multi_ctx([[(0.5, beside)], []])
    out = _run(verify_placement, ctx, _observation("overhead", "agentview"))
    assert out["route"] == "clear"
