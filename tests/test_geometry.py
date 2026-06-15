"""CPU numerics for the geometry bundle against gap's synthetic observation.

``make_test_observation`` renders a real pinhole camera over ground-truth
boxes, so mask → points → OBB → grasp can be checked against true numbers,
not mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import PerceptionFailed
from gap.testing import make_test_observation

# A denser image than the (120, 160) default so back-projected point spacing
# (~depth/fx ≈ 3 mm) stays below the DBSCAN eps (5 mm).
_IMAGE_HW = (240, 320)
_CUBE_CENTER = (0.0, 0.0, 0.03)
_CUBE_SIZE = (0.06, 0.06, 0.06)


@pytest.fixture(scope="module")
def scene():
    obs, gt = make_test_observation(
        [("cube", _CUBE_CENTER, _CUBE_SIZE)], image_hw=_IMAGE_HW
    )
    return obs, gt


@pytest.fixture(scope="module")
def cube_world_points(scene, tool_registry):
    obs, gt = scene
    frame = obs["cameras"][0]
    mask = gt["cube"]["mask"].astype(np.uint8) * 255  # gap Mask convention
    out = tool_registry.invoke(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=frame["depth"],
        intrinsics=frame["intrinsics"],
        camera_pose=frame["pose"],
    )
    return out["points"]["points"]


# ---------------------------------------------------------------------------
# mask -> world points
# ---------------------------------------------------------------------------


def test_mask_to_world_points_reprojects_onto_gt_box(cube_world_points):
    pts = cube_world_points
    assert pts.shape[1] == 3
    assert pts.dtype == np.float32
    assert len(pts) > 200, "synthetic cube mask should yield a dense cloud"

    center = np.asarray(_CUBE_CENTER)
    half = np.asarray(_CUBE_SIZE) / 2.0
    # Back-projection uses integer pixel coords vs the renderer's half-pixel
    # centers — allow a half-pixel (~1.6 mm at 0.9 m) + float32 slack.
    tol = 5e-3
    lo, hi = center - half - tol, center + half + tol
    assert np.all(pts >= lo[None, :]), "points fall outside the gt box (low)"
    assert np.all(pts <= hi[None, :]), "points fall outside the gt box (high)"

    # The visible surface must include the top face (z ≈ 0.06).
    assert pts[:, 2].max() > 0.05


def test_mask_to_world_points_empty_mask(scene, tool_registry):
    obs, _ = scene
    frame = obs["cameras"][0]
    empty = np.zeros(_IMAGE_HW, dtype=np.uint8)
    out = tool_registry.invoke(
        "geometry.mask_to_world_points",
        mask=empty,
        depth=frame["depth"],
        intrinsics=frame["intrinsics"],
        camera_pose=frame["pose"],
    )
    assert out["points"]["points"].shape == (0, 3)


# ---------------------------------------------------------------------------
# filter + OBB fit
# ---------------------------------------------------------------------------


def test_filter_and_compute_obb_recovers_cube(cube_world_points, tool_registry):
    out = tool_registry.invoke(
        "geometry.filter_and_compute_obb",
        points={"points": cube_world_points},
    )
    obb = out["obb"]

    center = np.array([obb["center"]["x"], obb["center"]["y"], obb["center"]["z"]])
    extent = np.array([obb["extent"]["x"], obb["extent"]["y"], obb["extent"]["z"]])

    # Only camera-facing surfaces are observed (2.5D), so allow a couple cm.
    assert np.allclose(center, _CUBE_CENTER, atol=0.02), center
    # Half-extents (gap.types convention) of a 6 cm cube → 0.03 each.
    assert np.allclose(extent, np.asarray(_CUBE_SIZE) / 2.0, atol=0.015), extent

    q = obb["orientation"]
    norm = q["w"] ** 2 + q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2
    assert abs(norm - 1.0) < 1e-6


def test_compute_obb_needs_four_points(tool_registry):
    with pytest.raises(PerceptionFailed):
        tool_registry.invoke(
            "geometry.compute_obb",
            points={"points": np.zeros((2, 3), dtype=np.float32)},
        )


def test_filter_noise_all_noise_returns_original(tool_registry):
    # 20 isolated points: nothing clusters → defensive fallback to input.
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(20, 3)).astype(np.float32) * 10.0
    out = tool_registry.invoke(
        "geometry.filter_noise", points={"points": pts}, eps=0.005, min_samples=10
    )
    assert out["points"]["points"].shape == pts.shape


# ---------------------------------------------------------------------------
# grasp candidates
# ---------------------------------------------------------------------------


def _gt_obb():
    return {
        "center": {"x": _CUBE_CENTER[0], "y": _CUBE_CENTER[1], "z": _CUBE_CENTER[2]},
        "extent": {
            "x": _CUBE_SIZE[0] / 2.0,
            "y": _CUBE_SIZE[1] / 2.0,
            "z": _CUBE_SIZE[2] / 2.0,
        },
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def test_top_down_grasp_candidates_above_obb(tool_registry):
    out = tool_registry.invoke(
        "geometry.top_down_grasp_candidates", obb=_gt_obb(), z_offset=-0.01
    )
    poses = out["candidates"]["poses"]

    # 6 cm cube: neither tall (>0.08) nor flat (<0.04) → 2 legacy + 23 fan.
    assert len(poses) == 25

    top_z = _CUBE_CENTER[2] + _CUBE_SIZE[2] / 2.0  # 0.06
    primary = poses[0]
    assert primary["position"]["x"] == pytest.approx(_CUBE_CENTER[0])
    assert primary["position"]["y"] == pytest.approx(_CUBE_CENTER[1])
    assert primary["position"]["z"] == pytest.approx(top_z - 0.01)
    # Above the OBB centroid (i.e. on the object's upper half).
    assert primary["position"]["z"] > _CUBE_CENTER[2]
    # Legacy primary orientation: gripper world-Z down (w,x,y,z)=(0,1,0,0).
    assert primary["rotation"] == {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

    # Legacy alt shares the position but rotates about the gripper's local Z.
    alt = poses[1]
    assert alt["position"] == primary["position"]
    assert alt["rotation"] != primary["rotation"]

    # Every candidate stays above the table-clearance floor and over the OBB.
    for pose in poses:
        assert pose["position"]["z"] >= -0.05
        assert pose["position"]["x"] == pytest.approx(_CUBE_CENTER[0])
        assert pose["position"]["y"] == pytest.approx(_CUBE_CENTER[1])


def _obb_with_height(full_h: float) -> dict:
    return {
        "center": {"x": 0.5, "y": 0.2, "z": full_h / 2.0},
        "extent": {"x": 0.025, "y": 0.025, "z": full_h / 2.0},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _is_straight_down(rotation: dict, tol_deg: float = 1.0) -> bool:
    """True when the gripper approach axis (local +Z) points along world -Z."""
    from scipy.spatial.transform import Rotation

    r = rotation
    R = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]]).as_matrix()
    cos = float(np.dot(R[:, 2], [0.0, 0.0, -1.0]))
    return cos >= np.cos(np.radians(tol_deg))


def test_top_down_grasp_candidates_tall_obb_strictly_top_down(tool_registry):
    """Tall (13-15 cm) bottles/cartons must get NO pitched side-grasp
    candidates: a 30-degree tilted pinch on a tall object bears gravity
    asymmetrically and slips during transport (the measured G1 grocery
    failure mode). All candidates stay straight top-down — matching the
    winning acceptance pipeline."""
    out = tool_registry.invoke(
        "geometry.top_down_grasp_candidates", obb=_obb_with_height(0.14)
    )
    poses = out["candidates"]["poses"]
    assert len(poses) == 25  # 2 legacy + 23 yaw/depth fan, no pitch grasps
    for pose in poses:
        assert _is_straight_down(pose["rotation"]), (
            "tall OBB produced a non-vertical grasp candidate"
        )


def test_top_down_grasp_candidates_flat_obb_keeps_pitch_grasps(tool_registry):
    """Flat boxes (perception often sees only the top face) keep the four
    +-30 degree pitched side-grasps appended after the top-down fan."""
    out = tool_registry.invoke(
        "geometry.top_down_grasp_candidates", obb=_obb_with_height(0.03)
    )
    poses = out["candidates"]["poses"]
    assert len(poses) == 29  # 25 top-down + 4 pitched side-grasps
    pitched = [p for p in poses if not _is_straight_down(p["rotation"], tol_deg=5.0)]
    assert len(pitched) == 4


def test_top_down_grasp_from_obb_clamps_to_table(tool_registry):
    out = tool_registry.invoke(
        "geometry.top_down_grasp_from_obb", obb=_gt_obb(), z_offset=-1.0
    )
    assert out["pose"]["position"]["z"] == pytest.approx(-0.05)  # _TABLE_Z_MIN


# ---------------------------------------------------------------------------
# legacy canary tools
# ---------------------------------------------------------------------------


def test_iou_units(tool_registry):
    assert tool_registry.invoke(
        "geometry.iou", box_a=[0, 0, 2, 2], box_b=[1, 1, 3, 3]
    )["iou"] == pytest.approx(1.0 / 7.0)
    assert tool_registry.invoke(
        "geometry.iou", box_a=[0, 0, 1, 1], box_b=[2, 2, 3, 3]
    )["iou"] == 0.0
    assert tool_registry.invoke(
        "geometry.iou", box_a=[0, 0, 1, 1], box_b=[0, 0, 1, 1]
    )["iou"] == pytest.approx(1.0)


def test_pose_distance_units(tool_registry):
    assert tool_registry.invoke(
        "geometry.pose_distance", a=[0, 0, 0], b=[3, 4, 12]
    )["distance"] == pytest.approx(13.0)
    with pytest.raises(ValueError):
        tool_registry.invoke("geometry.pose_distance", a=[0, 0], b=[1, 1, 1])


def test_xy_distance_ignores_z(tool_registry):
    out = tool_registry.invoke(
        "geometry.compute_xy_distance",
        point_a={"x": 0.0, "y": 0.0, "z": 5.0},
        point_b={"x": 3.0, "y": 4.0, "z": -5.0},
    )
    assert out["distance"] == pytest.approx(5.0)
