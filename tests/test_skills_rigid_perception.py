"""CPU contracts for the rigid perception, reconstruction and verification bundles.

``perceiving-functional-features`` (the four end-to-end scripts beside
``perceive_feature``), ``reconstructing-collision-worlds`` (sliver filter,
corridor and approach-tube carves, packing, camera selection),
``verifying-grasps`` (the four gates), ``verifying-placement`` (three
verdicts) and ``perceiving-sorting-pairs`` (the graph-supplied identity
table). Every model-backed tool is a :class:`gap.testing.FakeContext` can;
the only real numerics are the scripts' own.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gap.testing import FakeContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
}


def _script(skills_registry, bundle, name):
    return skills_registry.get(bundle).canonical_scripts[name].module


def _frame(name: str = "overhead", h: int = 8, w: int = 8, pose=None, rgb=None) -> dict:
    return {
        "name": name,
        "rgb": rgb if rgb is not None else np.zeros((h, w, 3), dtype=np.uint8),
        "depth": np.ones((h, w), dtype=np.float64),
        "intrinsics": np.eye(3),
        "pose": pose or IDENTITY,
    }


def _cloud(points) -> dict:
    return {"points": np.asarray(points, dtype=np.float32).reshape(-1, 3)}


def _blob(center, n: int = 40, spread: float = 0.004, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.uniform(-spread, spread, size=(n, 3))


def _obb(center=(0.0, 0.0, 0.0), extent=(0.03, 0.01, 0.01)) -> dict:
    return {
        "center": dict(zip(("x", "y", "z"), map(float, center), strict=True)),
        "extent": dict(zip(("x", "y", "z"), map(float, extent), strict=True)),
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


# ---------------------------------------------------------------------------
# Bundle declarations
# ---------------------------------------------------------------------------


def test_functional_features_declares_six_scripts_and_two_exits(skills_registry):
    info = skills_registry.get("perceiving-functional-features")
    assert info.kind == "skill"
    assert set(info.meta.exit_conditions) == {"found", "not_found"}
    # Six since the tool-hanging fold (2026-09-11): `perceive_fixture_feature`
    # is the profile-declared fixture side of `perceive_object_feature`.
    assert set(info.canonical_scripts) == {
        "perceive_feature",
        "perceive_object_feature",
        "perceive_protruding_shaft",
        "perceive_directed_tip",
        "perceive_aperture",
        "perceive_fixture_feature",
    }
    assert set(info.meta.allowed_tools) == {
        "sam3.segment_text",
        "sam3.segment_box",
        "grounding-dino.detect",
        "geometry.mask_to_world_points",
        "geometry.filter_and_compute_obb",
        "geometry.fit_planar_feature",
        "geometry.fit_linear_feature",
    }
    assert info.meta.requires is None or not info.meta.requires.connector
    assert "Use when" in info.meta.description
    shaft = info.canonical_scripts["perceive_protruding_shaft"].schema.inputs
    assert shaft["required"].default is True
    assert shaft["label_keyword"].default == "hook"


def test_new_rigid_bundles_declare_their_contracts(skills_registry):
    world = skills_registry.get("reconstructing-collision-worlds")
    assert set(world.meta.exit_conditions) == {"built", "failed"}
    assert world.meta.requires.connector == [
        "motion.get_robot_collision_spheres",
        "motion.build_world_tsdf",
    ]
    assert set(world.meta.allowed_tools) == {
        "sam3.segment_text",
        "motion.get_robot_collision_spheres",
        "motion.build_world_tsdf",
        "geometry.build_world_config",
    }
    assert set(world.canonical_scripts) == {"build_collision_world"}

    grasps = skills_registry.get("verifying-grasps")
    assert set(grasps.meta.exit_conditions) == {"verified", "not_held"}
    assert grasps.meta.requires.connector == ["robot.describe_workspace"]
    assert "robot.go_to_pose_cartesian" in grasps.meta.allowed_tools
    inputs = grasps.canonical_scripts["verify_grasp"].schema.inputs
    assert inputs["lift_m"].default == pytest.approx(0.04)
    assert inputs["score_min"].default == pytest.approx(0.005)

    placement = skills_registry.get("verifying-placement")
    assert set(placement.meta.exit_conditions) == {"verified", "not_placed"}
    assert placement.meta.requires is None or not placement.meta.requires.connector
    assert set(placement.meta.allowed_tools) == {
        "sam3.segment_text",
        "geometry.mask_to_world_points",
    }

    sorting = skills_registry.get("perceiving-sorting-pairs")
    select_inputs = sorting.canonical_scripts["select_pair"].schema.inputs
    assert {"identity_hints", "canonical_targets", "allowed_labels"} <= set(select_inputs)
    for name in ("identity_hints", "canonical_targets", "allowed_labels"):
        assert select_inputs[name].default == ""


# ---------------------------------------------------------------------------
# perceive_object_feature
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perceive_object_feature(skills_registry):
    return _script(skills_registry, "perceiving-functional-features", "perceive_object_feature")


_CANDIDATES = json.dumps(
    [
        {
            "kind": "wrench",
            "object_description": "adjustable wrench",
            "feature_description": "circular ring end of adjustable wrench",
            "feature_type": "loop",
            "feature_score_min": 0.25,
        },
        {
            "kind": "scissors",
            "object_description": "scissors",
            "feature_description": "",
            "feature_type": "loop",
            "feature_score_min": 0.05,
        },
    ]
)


def _object_ctx(scores: dict[str, float], ring_points: np.ndarray, support_points=None):
    """Segmentation by query; back-projection by mask pixel count."""
    object_mask = np.zeros((8, 8), dtype=np.uint8)
    object_mask[2:6, 2:6] = 255  # 16 px
    ring_mask = np.zeros((8, 8), dtype=np.uint8)
    ring_mask[3:5, 3:5] = 255  # 4 px
    object_points = _blob((0.5, 0.0, 0.02), n=30, seed=1)

    def segment(image, query, max_results):
        score = scores.get(query)
        if score is None:
            return {"masks": [], "scores": []}
        mask = ring_mask if "ring" in query else object_mask
        return {"masks": [mask], "scores": [score]}

    def backproject(mask, depth, intrinsics, camera_pose):
        count = int(np.count_nonzero(mask))
        if count == 4:
            return {"points": _cloud(ring_points)}
        if count == 16:
            return {"points": _cloud(object_points)}
        return {
            "points": _cloud(support_points if support_points is not None else np.empty((0, 3)))
        }

    fit = {
        "pose": {
            "position": {"x": 0.5, "y": 0.0, "z": 0.02},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        "normal": {"x": 0.0, "y": 0.0, "z": 1.0},
        "radius": 0.02,
        "radius_inner": 0.015,
        "planarity": 1.0,
    }
    return FakeContext(
        tool_responses={
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": backproject,
            "geometry.filter_and_compute_obb": {"obb": _obb((0.5, 0.0, 0.02))},
            "geometry.fit_planar_feature": fit,
        }
    )


def test_object_feature_takes_the_strongest_candidate_and_fits_its_loop(perceive_object_feature):
    ring = _blob((0.52, 0.0, 0.02), n=12, seed=2)
    ctx = _object_ctx(
        {"adjustable wrench": 0.5, "scissors": 0.3, "circular ring end of adjustable wrench": 0.4},
        ring,
    )
    camera = _frame(
        pose={"position": {"x": 0.5, "y": 0.0, "z": 1.0}, "rotation": IDENTITY["rotation"]}
    )
    out = perceive_object_feature.run(
        ctx, {"cameras": [camera]}, _CANDIDATES, complete_to_support=False
    )
    assert out["target_kind"] == "wrench"
    feature = out["functional_feature"]
    assert feature["kind"] == "loop"
    assert feature["description"] == "circular ring end of adjustable wrench"
    assert feature["confidence"] == pytest.approx(0.5)
    assert feature["radius_inner"] == pytest.approx(0.015)
    assert feature["axis"]["z"] == pytest.approx(1.0)
    expected_center = np.median(ring, axis=0)
    assert out["feature_center"]["x"] == pytest.approx(expected_center[0])
    assert out["target_obb"]["center"]["x"] == pytest.approx(0.5)
    # Object rows are segmented one result each; the feature at its own floor.
    object_calls = [c for c in ctx.calls_to("sam3.segment_text") if c.kwargs["max_results"] == 1]
    assert [c.kwargs["query"] for c in object_calls] == ["adjustable wrench", "scissors"]
    fit_call = ctx.calls_to("geometry.fit_planar_feature")[0].kwargs
    assert fit_call["fit_circle_center"] is True
    assert fit_call["normal_hint"]["z"] > 0.0  # toward the camera above


def test_object_feature_instruction_restricts_the_rows_tried(perceive_object_feature):
    ctx = _object_ctx({"adjustable wrench": 0.5, "scissors": 0.3}, _blob((0.5, 0.0, 0.02)))
    out = perceive_object_feature.run(
        ctx,
        {"cameras": [_frame()]},
        _CANDIDATES,
        instruction="hang the scissors",
        complete_to_support=False,
    )
    assert out["target_kind"] == "scissors"
    assert [c.kwargs["query"] for c in ctx.calls_to("sam3.segment_text")] == ["scissors"]
    # An empty feature_description yields the OBB centre as a point feature.
    assert out["feature_center"] == out["target_obb"]["center"]
    assert out["functional_feature"]["radius_inner"] == 0.0
    assert out["functional_feature"]["description"] == ""
    assert ctx.call_count("geometry.fit_planar_feature") == 0


def test_object_feature_completes_a_thin_object_down_to_its_support(perceive_object_feature):
    support = _blob((0.5, 0.0, 0.0), n=40, spread=0.0005, seed=3)
    ctx = _object_ctx({"scissors": 0.4}, _blob((0.5, 0.0, 0.02)), support_points=support)
    out = perceive_object_feature.run(
        ctx, {"cameras": [_frame()]}, _CANDIDATES, instruction="scissors"
    )
    cloud = np.asarray(out["target_cloud"]["points"])
    assert len(cloud) == 60  # 30 observed + 30 phantom bottom points
    assert cloud[30:, 2] == pytest.approx(float(np.median(support[:, 2])) + 0.001, abs=1e-6)


def test_object_feature_raises_when_nothing_or_only_weak_detections(perceive_object_feature):
    with pytest.raises(ValueError, match="could not localize"):
        perceive_object_feature.run(
            _object_ctx({}, _blob((0.5, 0.0, 0.02))), {"cameras": [_frame()]}, _CANDIDATES
        )
    with pytest.raises(ValueError, match="too weak"):
        perceive_object_feature.run(
            _object_ctx({"scissors": 0.05}, _blob((0.5, 0.0, 0.02))),
            {"cameras": [_frame()]},
            _CANDIDATES,
        )
    with pytest.raises(ValueError, match="non-empty JSON list"):
        perceive_object_feature.run(FakeContext(), {"cameras": [_frame()]}, "[]")


# ---------------------------------------------------------------------------
# perceive_protruding_shaft
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perceive_protruding_shaft(skills_registry):
    return _script(skills_registry, "perceiving-functional-features", "perceive_protruding_shaft")


def _shaft_roi() -> np.ndarray:
    """A board at x=0.80 with a 55 mm shaft protruding toward -X at z=1."""
    rng = np.random.default_rng(1)
    board = np.column_stack(
        [np.full(70, 0.80), rng.uniform(-0.03, 0.03, 70), rng.uniform(0.97, 1.03, 70)]
    )
    shaft = np.column_stack(
        [np.linspace(0.74, 0.795, 30), rng.normal(0.0, 0.001, 30), 1.0 + rng.normal(0.0, 0.001, 30)]
    )
    return np.vstack([board, shaft])


def _shaft_ctx(detections: list[dict]) -> FakeContext:
    return FakeContext(
        tool_responses={
            "grounding-dino.detect": {"detections": detections},
            "sam3.segment_box": {"masks": [np.ones((8, 8), dtype=np.uint8)], "scores": [0.9]},
            "geometry.mask_to_world_points": {"points": _cloud(_shaft_roi())},
            "geometry.filter_and_compute_obb": {
                "obb": _obb((0.77, 0.0, 1.0), (0.03, 0.004, 0.003))
            },
        }
    )


def test_shaft_not_required_returns_not_found_without_raising(perceive_protruding_shaft):
    out = perceive_protruding_shaft.run(
        _shaft_ctx([]), {"cameras": [_frame()]}, "small hook on the board", required=False
    )
    assert out["route"] == "not_found"
    assert out["fixture_feature"] == {}
    assert out["fixture_kind"] == ""
    assert np.asarray(out["fixture_cloud"]["points"]).shape == (0, 3)


def test_shaft_required_raises_and_the_label_gate_skips_the_board(perceive_protruding_shaft):
    board_only = [{"label": "pegboard", "score": 0.9, "box": {"x1": 0, "y1": 0, "x2": 7, "y2": 1}}]
    with pytest.raises(ValueError, match="no thin SAM3-refined hook candidate"):
        perceive_protruding_shaft.run(_shaft_ctx(board_only), {"cameras": [_frame()]}, "hook")
    ctx = _shaft_ctx(board_only)
    out = perceive_protruding_shaft.run(ctx, {"cameras": [_frame()]}, "hook", required=False)
    assert out["route"] == "not_found"
    assert ctx.call_count("sam3.segment_box") == 0


def test_shaft_recovers_tip_axis_and_usable_length_from_the_roi_depth(perceive_protruding_shaft):
    hook = [
        {"label": "small yellow hook", "score": 0.6, "box": {"x1": 1, "y1": 3, "x2": 7, "y2": 4}}
    ]
    out = perceive_protruding_shaft.run(
        _shaft_ctx(hook), {"cameras": [_frame()]}, "small yellow hook on the board"
    )
    assert out["route"] == "found"
    assert out["fixture_kind"] == "shaft"
    assert out["fixture_axis"]["x"] == pytest.approx(-1.0)
    assert out["fixture_axis"]["y"] == pytest.approx(0.0, abs=1e-9)
    assert out["shaft_tip"]["x"] == pytest.approx(0.741, abs=0.003)
    assert out["shaft_tip"]["z"] == pytest.approx(1.0, abs=0.002)
    feature = out["fixture_feature"]
    assert feature["kind"] == "shaft"
    assert feature["usable_length"] == pytest.approx(0.059, abs=0.003)
    assert feature["seating_margin"] == pytest.approx(0.015)
    assert feature["radius_outer"] == pytest.approx(0.003)
    assert feature["pose"]["position"] == out["shaft_tip"]


# ---------------------------------------------------------------------------
# perceive_directed_tip
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perceive_directed_tip(skills_registry):
    return _script(skills_registry, "perceiving-functional-features", "perceive_directed_tip")


def _tapered_cloud() -> np.ndarray:
    """Wide at x=0, narrow at x=0.1, at z=1."""
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 0.10, 200)
    width = 0.012 * (1.0 - x / 0.10) + 0.0005
    y = rng.uniform(-1.0, 1.0, 200) * width
    z = 1.0 + rng.uniform(-1.0, 1.0, 200) * width * 0.3
    return np.column_stack([x, y, z])


def _tip_ctx(marker_points=None) -> FakeContext:
    body = _tapered_cloud()
    object_mask = np.ones((8, 8), dtype=np.uint8)
    marker_mask = np.zeros((8, 8), dtype=np.uint8)
    marker_mask[0:3, 0:3] = 1

    def segment(image, query, max_results):
        if "cap" in query:
            if marker_points is None:
                return {"masks": [], "scores": []}
            return {"masks": [marker_mask], "scores": [0.5]}
        return {"masks": [object_mask], "scores": [0.9]}

    def backproject(mask, depth, intrinsics, camera_pose):
        if int(np.count_nonzero(mask)) == 9:
            return {"points": _cloud(marker_points)}
        return {"points": _cloud(body)}

    return FakeContext(
        tool_responses={
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": backproject,
            "geometry.filter_and_compute_obb": {
                "obb": _obb((0.05, 0.0, 1.0), (0.05, 0.006, 0.002))
            },
        }
    )


def test_directed_tip_without_a_marker_points_at_the_narrow_end(perceive_directed_tip):
    ctx = _tip_ctx()
    out = perceive_directed_tip.run(ctx, {"cameras": [_frame()]}, "syringe")
    assert "marker_center" in out
    assert out["marker_center"]["x"] > 0.08
    assert out["insertion_pose"]["position"]["x"] > 0.095
    feature = out["functional_feature"]
    assert feature["kind"] == "tip"
    assert feature["axis"]["x"] == pytest.approx(1.0, abs=0.05)
    assert feature["radius_outer"] > 0.0
    assert feature["description"] == "tip of syringe"
    assert ctx.call_count("sam3.segment_text") == 1  # no marker query was asked


def test_directed_tip_marker_reverses_the_direction(perceive_directed_tip):
    marker = _blob((0.0, 0.0, 1.0), n=12, spread=0.002, seed=5)
    out = perceive_directed_tip.run(
        _tip_ctx(marker),
        {"cameras": [_frame()]},
        "syringe",
        direction_marker_description="red cap of syringe",
    )
    assert out["marker_center"]["x"] == pytest.approx(0.0, abs=0.003)
    assert out["functional_feature"]["axis"]["x"] == pytest.approx(-1.0, abs=0.05)
    assert out["insertion_pose"]["position"]["x"] < 0.005
    assert out["functional_feature"]["description"] == "red cap of syringe"


# ---------------------------------------------------------------------------
# perceive_aperture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perceive_aperture(skills_registry):
    return _script(skills_registry, "perceiving-functional-features", "perceive_aperture")


def test_aperture_comes_from_the_mask_ray_on_the_fitted_lid(perceive_aperture):
    # A 7x9 lid at z=1 seen by an identity camera at the origin: the hole's
    # mask centroid (col 4, row 3) casts a ray hitting the lid at (4, 3, 1).
    box = np.full((7, 9), 255, dtype=np.uint8)
    hole = np.zeros((7, 9), dtype=np.uint8)
    hole[2:5, 3:6] = 255
    x, y = np.meshgrid(np.linspace(-0.3, 0.3, 9), np.linspace(-0.2, 0.2, 7))
    lid = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    ctx = FakeContext(
        tool_responses={
            "sam3.segment_text": lambda image, query, max_results: {
                "masks": [hole if "hole" in query else box],
                "scores": [1.0],
            },
            "geometry.mask_to_world_points": {"points": _cloud(lid)},
        }
    )
    observation = {"cameras": [_frame(h=7, w=9)]}
    out = perceive_aperture.run(ctx, observation, "box", "hole in top of box")
    assert out["aperture_center"]["x"] == pytest.approx(4.0)
    assert out["aperture_center"]["y"] == pytest.approx(3.0)
    assert out["aperture_center"]["z"] == pytest.approx(1.0)
    assert out["fixture_axis"]["z"] == pytest.approx(1.0)
    assert out["aperture_radius"] > 0.0
    assert out["rim_z"] == pytest.approx(1.0)
    feature = out["fixture_feature"]
    assert feature["kind"] == "aperture"
    assert feature["radius_inner"] == pytest.approx(out["aperture_radius"])
    assert feature["description"] == "hole in top of box"


# ---------------------------------------------------------------------------
# reconstructing-collision-worlds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def build_collision_world(skills_registry):
    return _script(skills_registry, "reconstructing-collision-worlds", "build_collision_world")


_CUBE_FACES = np.array(
    [
        [0, 1, 3],
        [0, 3, 2],  # -x
        [4, 7, 5],
        [4, 6, 7],  # +x
        [0, 5, 1],
        [0, 4, 5],  # -y
        [2, 3, 7],
        [2, 7, 6],  # +y
        [0, 2, 6],
        [0, 6, 4],  # -z
        [1, 7, 3],
        [1, 5, 7],  # +z
    ],
    dtype=np.int32,
)


def _box_mesh(name: str, center, size) -> dict:
    cx, cy, cz = center
    sx, sy, sz = size
    vertices = np.array(
        [
            [cx + dx * sx / 2.0, cy + dy * sy / 2.0, cz + dz * sz / 2.0]
            for dx in (-1, 1)
            for dy in (-1, 1)
            for dz in (-1, 1)
        ],
        dtype=np.float32,
    )
    return {"name": name, "vertices": vertices, "faces": _CUBE_FACES.copy(), "pose": IDENTITY}


def _sliver() -> dict:
    vertices = np.array(
        [[0.3, 0.0, 0.9], [0.35, 0.0, 0.9], [0.3, 0.002, 0.9], [0.35, 0.002, 0.902]],
        dtype=np.float32,
    )
    return {
        "name": "sliver",
        "vertices": vertices,
        "faces": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        "pose": IDENTITY,
    }


def _scene(cube_center=(0.5, 0.0, 0.945)) -> list[dict]:
    return [
        _sliver(),
        _box_mesh("cube", cube_center, (0.02, 0.02, 0.03)),
        _box_mesh("slab", (0.5, 0.0, 0.89), (1.0, 1.0, 0.02)),
    ]


def _world_ctx(meshes: list[dict], segment=None) -> FakeContext:
    return FakeContext(
        tool_responses={
            "sam3.segment_text": segment or {"masks": [], "scores": []},
            "geometry.build_world_config": {
                "config": {"meshes": meshes},
                "mesh_names": [m["name"] for m in meshes],
            },
        }
    )


def _world_observation(extra_camera: bool = False) -> dict:
    cameras = [_frame("overhead")]
    if extra_camera:
        cameras.append(_frame("agentview", rgb=np.ones((8, 8, 3), dtype=np.uint8)))
    return {"cameras": cameras}


def test_world_drops_the_sliver_keeps_the_slab_and_the_names(build_collision_world):
    ctx = _world_ctx(_scene())
    out = build_collision_world.run(ctx, _world_observation(), np.zeros((8, 8), dtype=np.uint8))
    assert [m["name"] for m in out["world_config"]["meshes"]] == ["cube", "slab"]
    assert out["mesh_names"] == ["cube", "slab"]
    build = ctx.calls_to("geometry.build_world_config")[0].kwargs
    assert build["voxel_size"] == pytest.approx(0.008)
    assert build["robot_spheres"] == []  # the connector tool was absent and tolerated
    assert [m["name"] for m in build["object_masks"]] == ["grasp_target"]


def test_world_corridor_removes_the_component_bridging_the_aperture(build_collision_world):
    out = build_collision_world.run(
        _world_ctx(_scene()),
        _world_observation(),
        np.zeros((8, 8), dtype=np.uint8),
        corridor_center={"x": 0.5, "y": 0.0, "z": 0.95},
        corridor_radius=0.03,
        corridor_rim_z=0.95,
    )
    assert [m["name"] for m in out["world_config"]["meshes"]] == ["slab"]
    assert len(out["world_config"]["meshes"][0]["faces"]) == 12  # the slab is untouched
    # A zero radius means no corridor at all.
    out = build_collision_world.run(
        _world_ctx(_scene()),
        _world_observation(),
        np.zeros((8, 8), dtype=np.uint8),
        corridor_center={"x": 0.5, "y": 0.0, "z": 0.95},
        corridor_radius=0.0,
        corridor_rim_z=0.95,
    )
    assert [m["name"] for m in out["world_config"]["meshes"]] == ["cube", "slab"]


def test_world_pack_meshes_folds_every_component_into_one(build_collision_world):
    out = build_collision_world.run(
        _world_ctx(_scene()),
        _world_observation(),
        np.zeros((8, 8), dtype=np.uint8),
        pack_meshes=True,
    )
    meshes = out["world_config"]["meshes"]
    assert [m["name"] for m in meshes] == ["perceived_scene"]
    assert np.asarray(meshes[0]["vertices"]).shape == (16, 3)
    assert np.asarray(meshes[0]["faces"]).shape == (24, 3)
    assert int(np.asarray(meshes[0]["faces"]).max()) == 15
    assert out["mesh_names"] == ["perceived_scene"]


def test_world_approach_tube_follows_the_outward_sign(build_collision_world):
    tip = {"x": 0.5, "y": 0.0, "z": 0.95}
    axis = {"x": -1.0, "y": 0.0, "z": 0.0}
    kwargs = dict(fixture_tip=tip, fixture_axis=axis)
    outward = build_collision_world.run(
        _world_ctx(_scene((0.45, 0.0, 0.95))),
        _world_observation(),
        np.zeros((8, 8), dtype=np.uint8),
        outward_sign=1.0,
        **kwargs,
    )
    assert [m["name"] for m in outward["world_config"]["meshes"]] == ["slab"]
    inward = build_collision_world.run(
        _world_ctx(_scene((0.45, 0.0, 0.95))),
        _world_observation(),
        np.zeros((8, 8), dtype=np.uint8),
        outward_sign=-1.0,
        **kwargs,
    )
    assert [m["name"] for m in inward["world_config"]["meshes"]] == ["cube", "slab"]


def test_world_camera_selection_and_per_view_target_masks(build_collision_world):
    def segment(image, query, max_results):
        if query == "syringe":
            return {"masks": [np.ones((8, 8), dtype=np.uint8)], "scores": [0.5]}
        return {"masks": [], "scores": []}

    ctx = _world_ctx(_scene(), segment)
    build_collision_world.run(
        ctx,
        _world_observation(extra_camera=True),
        np.zeros((8, 8), dtype=np.uint8),
        target_description="syringe",
        camera_names="overhead",
    )
    build = ctx.calls_to("geometry.build_world_config")[0].kwargs
    assert [c["name"] for c in build["cameras"]] == ["overhead"]
    assert [m["name"] for m in build["object_masks"]] == ["grasp_target"]

    ctx = _world_ctx(_scene(), segment)
    build_collision_world.run(
        ctx,
        _world_observation(extra_camera=True),
        np.zeros((8, 8), dtype=np.uint8),
        target_description="syringe",
    )
    build = ctx.calls_to("geometry.build_world_config")[0].kwargs
    assert [c["name"] for c in build["cameras"]] == ["overhead", "agentview"]
    names = [m["name"] for m in build["object_masks"]]
    assert names == ["grasp_target", "grasp_target_view_1"]
    assert build["object_masks"][1]["camera_index"] == 1


def test_world_raises_when_nothing_survives(build_collision_world):
    with pytest.raises(ValueError, match="no scene mesh"):
        build_collision_world.run(
            _world_ctx([_sliver()]), _world_observation(), np.zeros((8, 8), dtype=np.uint8)
        )
    with pytest.raises(ValueError, match="requires RGB-D cameras"):
        build_collision_world.run(
            _world_ctx(_scene()),
            _world_observation(),
            np.zeros((8, 8), dtype=np.uint8),
            camera_names="wrist",
        )


# ---------------------------------------------------------------------------
# verifying-grasps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def verify_grasp(skills_registry):
    return _script(skills_registry, "verifying-grasps", "verify_grasp")


def _grasp_ctx(
    points, *, wrist_score=0.02, overhead_score=0.02, surface_z=0.0, ee_z=0.30
) -> FakeContext:
    wrist_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    overhead_rgb = np.ones((8, 8, 3), dtype=np.uint8)
    cameras = [_frame("eye_in_hand_0", rgb=wrist_rgb), _frame("overhead", rgb=overhead_rgb)]

    def segment(image, query, max_results):
        score = wrist_score if image is wrist_rgb else overhead_score
        return {"masks": [np.ones((8, 8), dtype=np.uint8)], "scores": [score]}

    return FakeContext(
        tool_responses={
            "robot.get_ee_pose": {
                "pose": {
                    "position": {"x": 0.5, "y": 0.0, "z": ee_z},
                    "rotation": IDENTITY["rotation"],
                }
            },
            "robot.go_to_pose_cartesian": {},
            "robot.get_observation": {"cameras": cameras},
            "sam3.segment_text": segment,
            "geometry.mask_to_world_points": {"points": _cloud(points)},
            "robot.describe_workspace": {"surface_z": surface_z},
        }
    )


def test_grasp_on_the_surface_is_not_held(verify_grasp):
    out = verify_grasp.run(_grasp_ctx(_blob((0.5, 0.0, 0.02))), "syringe")
    assert out["route"] == "not_held" and out["verified"] is False
    assert "above table" in out["reason"]
    assert out["camera"] == "eye_in_hand_0"


def test_grasp_far_from_the_hand_is_not_held(verify_grasp):
    # 9 cm above the table but 25 cm below the lifted hand at z=0.34.
    out = verify_grasp.run(_grasp_ctx(_blob((0.5, 0.0, 0.09))), "syringe")
    assert out["route"] == "not_held"
    assert "hand distance" in out["reason"]
    assert out["observed_center"]["z"] == pytest.approx(0.09, abs=0.005)


def test_grasp_falls_back_to_the_overhead_camera_below_the_wrist_floor(verify_grasp):
    ctx = _grasp_ctx(_blob((0.5, 0.0, 0.30)), wrist_score=0.001, overhead_score=0.02)
    out = verify_grasp.run(ctx, "syringe", marker_description="red cap")
    assert out["route"] == "verified"
    assert out["camera"] == "overhead"
    # Both queries were tried on the wrist before the overhead camera was asked.
    assert [c.kwargs["query"] for c in ctx.calls_to("sam3.segment_text")] == [
        "syringe",
        "red cap",
        "syringe",
    ]


def test_grasp_happy_path_reports_the_point_count_after_the_lift(verify_grasp):
    ctx = _grasp_ctx(_blob((0.5, 0.0, 0.30)))
    out = verify_grasp.run(ctx, "syringe")
    assert out["route"] == "verified" and out["verified"] is True
    assert out["point_count"] == 40
    assert out["camera"] == "eye_in_hand_0"
    assert out["reason"] == ""
    lifted = ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]
    assert lifted["position"]["z"] == pytest.approx(0.34)


def test_grasp_unseen_or_sparse_is_not_held_without_raising(verify_grasp):
    ctx = _grasp_ctx(_blob((0.5, 0.0, 0.30)), wrist_score=0.001, overhead_score=0.001)
    out = verify_grasp.run(ctx, "syringe")
    assert out["route"] == "not_held"
    assert "no camera sees" in out["reason"]
    assert ctx.call_count("robot.describe_workspace") == 0
    out = verify_grasp.run(_grasp_ctx(_blob((0.5, 0.0, 0.30), n=5)), "syringe")
    assert out["route"] == "not_held"
    assert "too few valid depth points" in out["reason"]
    assert out["point_count"] == 5


# ---------------------------------------------------------------------------
# verifying-placement
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def verify_placement(skills_registry):
    return _script(skills_registry, "verifying-placement", "verify_placement")


def _placement_ctx(score: float, points) -> FakeContext:
    return FakeContext(
        tool_responses={
            "sam3.segment_text": {"masks": [np.ones((8, 8), dtype=np.uint8)], "scores": [score]},
            "geometry.mask_to_world_points": {"points": _cloud(points)},
        }
    )


_APERTURE = {"x": 0.5, "y": 0.0, "z": 0.95}


def test_placement_vanished_object_counts_as_verified(verify_placement):
    ctx = _placement_ctx(0.01, _blob((0.5, 0.0, 0.97)))
    out = verify_placement.run(ctx, {"cameras": [_frame()]}, "syringe", _APERTURE, 0.95)
    assert out["route"] == "verified" and out["verified"] is True
    assert "disappeared" in out["evidence"]
    assert ctx.call_count("geometry.mask_to_world_points") == 0


def test_placement_visible_below_the_rim_is_verified(verify_placement):
    ctx = _placement_ctx(0.5, _blob((0.52, 0.01, 0.90)))
    out = verify_placement.run(ctx, {"cameras": [_frame()]}, "syringe", _APERTURE, 0.95)
    assert out["route"] == "verified"
    assert "below the aperture" in out["evidence"]


def test_placement_visible_above_or_beside_is_not_placed(verify_placement):
    above = verify_placement.run(
        _placement_ctx(0.5, _blob((0.5, 0.0, 0.97))),
        {"cameras": [_frame()]},
        "syringe",
        _APERTURE,
        0.95,
    )
    assert above["route"] == "not_placed" and above["verified"] is False
    beside = verify_placement.run(
        _placement_ctx(0.5, _blob((0.7, 0.0, 0.90))),
        {"cameras": [_frame()]},
        "syringe",
        _APERTURE,
        0.95,
    )
    assert beside["route"] == "not_placed"
    sparse = verify_placement.run(
        _placement_ctx(0.5, _blob((0.5, 0.0, 0.90), n=3)),
        {"cameras": [_frame()]},
        "syringe",
        _APERTURE,
        0.95,
    )
    assert sparse["route"] == "not_placed"
    assert "too few" in sparse["evidence"]


# ---------------------------------------------------------------------------
# perceiving-sorting-pairs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def select_pair(skills_registry):
    return _script(skills_registry, "perceiving-sorting-pairs", "select_pair")


_REGIONS = json.dumps(
    [
        {"label": "hammer", "obb": _obb((1.0, 0.0, 0.0))},
        {"label": "drill", "obb": _obb((2.0, 0.0, 0.0))},
    ]
)
_ANSWER = "TARGET: blue hammer handle; LABEL: hammer; BOX: 10,10,60,60; PIXEL: 30,30"


def _sort_ctx(answer: str, prompts: list[str]) -> tuple[FakeContext, dict]:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    camera = _frame("overhead", h=200, w=200, rgb=image)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:40, 20:40] = 1

    def vlm(**kwargs):
        prompts.append(kwargs["prompt"])
        return {"text": answer}

    ctx = FakeContext(
        tool_responses={
            "grounding-dino.detect": {
                "detections": [
                    {
                        "score": 0.9,
                        "label": "source bin",
                        "box": {"x1": 0, "y1": 0, "x2": 150, "y2": 150},
                    }
                ]
            },
            "vlm.query": vlm,
            "sam3.segment_box": {"masks": [mask], "scores": [0.8]},
            "geometry.mask_to_world_points": {"points": _cloud(_blob((0.4, 0.0, 0.02)))},
            "geometry.filter_and_compute_obb": {"obb": _obb((0.4, 0.0, 0.02))},
        }
    )
    return ctx, {"cameras": [camera]}


def test_select_pair_default_prompt_and_association(select_pair):
    prompts: list[str] = []
    ctx, observation = _sort_ctx(_ANSWER, prompts)
    out = select_pair.run(ctx, observation, "sort the tools", _REGIONS, "source bin")
    assert out["status"] == "found"
    assert out["target_label"] == "hammer"
    assert out["destination_obb"]["center"]["x"] == pytest.approx(1.0)
    assert "narrow graspable handle" in prompts[0]
    assert "hammer, drill" in prompts[0]
    segment = ctx.calls_to("sam3.segment_box")[0].kwargs
    assert segment["pixel_x"] == pytest.approx(30.0) and segment["use_point"] is True


def test_select_pair_identity_hints_replace_the_generic_cue(select_pair):
    prompts: list[str] = []
    ctx, observation = _sort_ctx(_ANSWER, prompts)
    select_pair.run(
        ctx,
        observation,
        "sort",
        _REGIONS,
        "source bin",
        identity_hints="HAMMER is the cyan T-shaped tool and TARGET must be 'blue hammer handle'.",
    )
    assert "HAMMER is the cyan T-shaped tool" in prompts[0]
    assert "narrow graspable handle" not in prompts[0]


def test_select_pair_allowed_labels_rejects_an_off_list_label(select_pair):
    prompts: list[str] = []
    ctx, observation = _sort_ctx(_ANSWER, prompts)
    with pytest.raises(ValueError, match="not one of \\['drill'\\]"):
        select_pair.run(ctx, observation, "sort", _REGIONS, "source bin", allowed_labels="drill")
    assert "labels not yet attempted are: drill." in prompts[0]
    with pytest.raises(ValueError, match="none of the layout labels"):
        select_pair.run(ctx, observation, "sort", _REGIONS, "source bin", allowed_labels="wrench")


def test_select_pair_canonical_targets_checks_only_listed_labels(select_pair):
    table = json.dumps({"Hammer": "Blue Hammer Handle"})
    ctx, observation = _sort_ctx(_ANSWER, [])
    out = select_pair.run(ctx, observation, "sort", _REGIONS, "source bin", canonical_targets=table)
    assert out["status"] == "found"
    wrong = "TARGET: hammer head; LABEL: hammer; BOX: 10,10,60,60; PIXEL: 30,30"
    ctx, observation = _sort_ctx(wrong, [])
    with pytest.raises(ValueError, match="expected 'blue hammer handle'"):
        select_pair.run(ctx, observation, "sort", _REGIONS, "source bin", canonical_targets=table)
    # Unlisted labels accept any TARGET, and without a table nothing is checked.
    drill = "TARGET: drill grip; LABEL: drill; BOX: 10,10,60,60; PIXEL: 30,30"
    ctx, observation = _sort_ctx(drill, [])
    out = select_pair.run(ctx, observation, "sort", _REGIONS, "source bin", canonical_targets=table)
    assert out["destination_obb"]["center"]["x"] == pytest.approx(2.0)
    ctx, observation = _sort_ctx(wrong, [])
    assert select_pair.run(ctx, observation, "sort", _REGIONS, "source bin")["status"] == "found"


def test_select_pair_parses_newlines_and_a_missing_pixel(select_pair):
    ctx, observation = _sort_ctx("TARGET: blue hammer handle\nLABEL: hammer\nBOX: 10,10,60,60", [])
    out = select_pair.run(ctx, observation, "sort", _REGIONS, "source bin")
    assert out["status"] == "found"
    segment = ctx.calls_to("sam3.segment_box")[0].kwargs
    assert segment["pixel_x"] == pytest.approx(35.0)  # the box centre stands in
    assert segment["pixel_y"] == pytest.approx(35.0)
    ctx, observation = _sort_ctx("DONE", [])
    assert select_pair.run(ctx, observation, "sort", _REGIONS, "source bin")["status"] == "finished"
