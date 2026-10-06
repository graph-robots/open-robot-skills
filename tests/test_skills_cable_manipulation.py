"""FakeContext tests for the three cable MANIPULATION bundles.

- seating-a-cable-crossing: the seat fraction is the densified share of the
  fitted curve inside the station's box on the required side, the plan's own
  ``station`` (not the pass index) picks the box, and the place/settle/look
  call order;

The planner behind ``cable.plan_support`` is not exercisable on a CPU: every
test cans its answer.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext

ROOT = Path(__file__).resolve().parents[1] / "skills"

BUNDLES = {
    "seating-a-cable-crossing": ("seat_station", ["robot.wait_steps"]),
}


def _load(bundle: str, script: str):
    path = ROOT / bundle / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(f"cable_manipulation_{script}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seat():
    return _load("seating-a-cable-crossing", "seat_station")






def _names(ctx: FakeContext) -> list[str]:
    return [c.tool for c in ctx.calls]


# ---------------------------------------------------------------------------
# Canned scene: a bench camera looking straight down from 1 m above the bench.
# ---------------------------------------------------------------------------

CAM_XY = (0.5, 0.2)
CAM_HEIGHT = 1.0
BENCH_Z = 0.75
FOCAL = 200.0
IMAGE = 128
CENTRE = IMAGE / 2


def _camera_frame() -> dict:
    """A frame whose pose maps pixel (u, v) at depth 1 m to the bench."""
    return {
        "name": "cable",
        "rgb": np.zeros((IMAGE, IMAGE, 3), dtype=np.uint8),
        "depth": np.full((IMAGE, IMAGE), CAM_HEIGHT, dtype=np.float64),
        "intrinsics": [[FOCAL, 0.0, CENTRE], [0.0, FOCAL, CENTRE], [0.0, 0.0, 1.0]],
        # A half turn about x: camera +z looks down world -z, camera +y is world -y.
        "pose": {
            "position": {"x": CAM_XY[0], "y": CAM_XY[1], "z": BENCH_Z + CAM_HEIGHT},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
        },
    }


def _observation() -> dict:
    return {"cameras": {"cable": _camera_frame()}}


def _rect_mask(u0: int, u1: int, v0: int, v1: int) -> np.ndarray:
    mask = np.zeros((IMAGE, IMAGE), dtype=np.uint8)
    mask[v0:v1, u0:u1] = 1
    return mask


def _rect_world(u0: int, u1: int, v0: int, v1: int) -> tuple[float, float, float]:
    """Where the camera above puts the median pixel of that rectangle."""
    u_med = (u0 + u1 - 1) / 2.0
    v_med = (v0 + v1 - 1) / 2.0
    return (
        CAM_XY[0] + (u_med - CENTRE) * CAM_HEIGHT / FOCAL,
        CAM_XY[1] - (v_med - CENTRE) * CAM_HEIGHT / FOCAL,
        BENCH_Z,
    )


# ---------------------------------------------------------------------------
# seating-a-cable-crossing
# ---------------------------------------------------------------------------

STATIONS = [[0.30, 0.00], [0.50, 0.20], [0.70, 0.40]]
SIDES = [-1.0, 1.0, -1.0]
SCENE = json.dumps({"stations": STATIONS, "sides": SIDES})
PLACE_LEG = {"stage": "place", "trajectory": {"waypoints": [{"positions": [0.1] * 6}]}}


def _straight_rod(x: float, y0: float, y1: float, knots: int = 21) -> list[list[float]]:
    """A 200 mm rod lying along y at a fixed x, as a polyline of knots."""
    return [[x, y0 + (y1 - y0) * i / (knots - 1), BENCH_Z] for i in range(knots)]


def _densified_share(points, station, side, inner=0.003, outer=0.045, half_y=0.035):
    """The seat fraction by its definition: resample the curve every 2 mm along
    its arclength, keep the samples abreast of the station, count those whose
    offset on the required side lies inside [inner, outer]."""
    pts = np.asarray(points, dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    samples = [np.interp(t, s, pts[:, 0:2].T[0]) for t in np.arange(0.0, s[-1], 0.002)]
    ys = [np.interp(t, s, pts[:, 1]) for t in np.arange(0.0, s[-1], 0.002)]
    inside = 0
    for x, y in zip(samples, ys, strict=True):
        if abs(y - station[1]) > half_y:
            continue
        off = (x - station[0]) * side
        if inner <= off <= outer:
            inside += 1
    return inside / len(samples)


def _seat_ctx(points, *, masks=None):
    mask = _rect_mask(60, 68, 0, IMAGE)
    return FakeContext(
        tool_responses={
            "robot.execute_trajectory": {"ok": True},
            "robot.wait_steps": {"waited": 30},
            "robot.get_observation": _observation(),
            "sam3.segment_text": ({"masks": [mask], "scores": [0.9]} if masks is None else masks),
            "curve.fit_centerline": {"points": points, "ordered": True},
        }
    )


def test_seat_fraction_is_the_densified_share_on_the_required_side(seat):
    station, side = STATIONS[1], SIDES[1]
    # 20 mm out on the required (+x) side: inside the [3, 45] mm box.
    rod = _straight_rod(station[0] + 0.020, station[1] - 0.10, station[1] + 0.10)
    ctx = _seat_ctx(rod)
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})

    out = seat.run(ctx, plan=plan, scene=SCENE, index=0, arm_id=0)

    expected = _densified_share(rod, station, side)
    assert 0.30 < expected < 0.40, "the canned rod should cross a 70 mm box of a 200 mm run"
    assert out["route"] == "done"
    assert out["fraction"] == pytest.approx(expected, abs=1e-3)
    assert out["fraction"] >= 0.047
    assert out["seated"] is True
    assert out["detail"].startswith("seated")

    # Place, settle, then look -- in that order, on the work arm.
    assert _names(ctx) == [
        "robot.execute_trajectory",
        "robot.wait_steps",
        "robot.get_observation",
        "sam3.segment_text",
        "curve.fit_centerline",
    ]
    place = ctx.calls_to("robot.execute_trajectory")[0].kwargs
    assert place["trajectory"] == PLACE_LEG["trajectory"]
    assert place["arm_id"] == 0
    assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 30}
    fit = ctx.calls_to("curve.fit_centerline")[0].kwargs
    assert fit["nodes"] == 42
    assert np.asarray(fit["camera_pose"]).shape == (4, 4)
    assert ctx.calls_to("sam3.segment_text")[0].kwargs["query"] == "thin white cable"


def test_seat_rod_on_the_wrong_side_counts_nothing(seat):
    station = STATIONS[1]
    rod = _straight_rod(station[0] - 0.020, station[1] - 0.10, station[1] + 0.10)
    ctx = _seat_ctx(rod)
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})

    out = seat.run(ctx, plan=plan, scene=SCENE, index=0, arm_id=0)

    assert out["fraction"] == 0.0
    assert out["seated"] is False
    assert out["route"] == "done"
    assert out["detail"] == "short at 0.000, no correction attempted"


def test_seat_uses_the_plans_station_not_the_pass_index(seat):
    station = STATIONS[1]
    rod = _straight_rod(station[0] + 0.020, station[1] - 0.10, station[1] + 0.10)

    # Pass 0 worked station 1: the plan says so, and the box measured is
    # station 1's, where the rod is.
    out = seat.run(
        _seat_ctx(rod),
        plan=json.dumps({"stages": [PLACE_LEG], "station": 1}),
        scene=SCENE,
        index=0,
        arm_id=0,
    )
    assert out["seated"] is True and out["fraction"] > 0.0

    # Without the plan's station the pass index falls through to station 0,
    # whose box the rod is nowhere near.
    out = seat.run(
        _seat_ctx(rod),
        plan=json.dumps({"stages": [PLACE_LEG]}),
        scene=SCENE,
        index=0,
        arm_id=0,
    )
    assert out["seated"] is False and out["fraction"] == 0.0


def test_seat_box_parameters_are_honoured(seat):
    station = STATIONS[1]
    # 20 mm out: inside the default box, outside a box that stops at 15 mm.
    rod = _straight_rod(station[0] + 0.020, station[1] - 0.10, station[1] + 0.10)
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})

    out = seat.run(_seat_ctx(rod), plan=plan, scene=SCENE, seat_outer=0.015)
    assert out["fraction"] == 0.0

    out = seat.run(_seat_ctx(rod), plan=plan, scene=SCENE, seat_fraction=0.5)
    assert out["seated"] is False and out["fraction"] > 0.0
    assert out["detail"].startswith("short at")


def test_seat_reports_when_it_cannot_measure(seat):
    plan = json.dumps({"stages": [PLACE_LEG], "station": 1})
    ctx = _seat_ctx([], masks={"masks": [], "scores": []})

    out = seat.run(ctx, plan=plan, scene=SCENE)

    assert out == {
        "route": "done",
        "seated": False,
        "fraction": 0.0,
        "detail": "could not measure the seat",
    }
    assert ctx.call_count("curve.fit_centerline") == 0


def test_seat_with_no_place_leg_touches_nothing(seat):
    ctx = _seat_ctx([])
    out = seat.run(ctx, plan=json.dumps({"stages": [{"stage": "move"}]}), scene=SCENE)
    assert out["detail"] == "no place leg"
    assert out["seated"] is False
    assert ctx.calls == []
