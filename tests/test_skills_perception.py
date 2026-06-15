"""Tests for the four perception skill bundles.

Canonical scripts run against :class:`gap.testing.FakeContext` with canned
model-tool responses shaped exactly like the landed tool bundles' returns
(grounding-dino.detect / sam3.segment_* / vlm.query / molmo.point_prompt),
while ``geometry.*`` calls delegate to the REAL geometry tools (pure CPU)
so mask → cloud → OBB numerics are checked against the synthetic
observation's ground truth, not mocks.

Loader-level checks assert the bundles discover via ``load_skills`` with
valid frontmatter and that every ``gap.allowed_tools`` name resolves to a
connector tool or a landed tool-bundle function.
"""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import PerceptionFailed, ToolError
from gap.testing import FakeContext, make_test_observation

# ---------------------------------------------------------------------------
# Bundle / contract declarations
# ---------------------------------------------------------------------------

PERCEPTION_BUNDLES: dict[str, set[str]] = {
    "perceiving-objects": {"perceive_dino_vlm"},
    "perceiving-objects-oneshot": {"perceive_simple"},
    "perceiving-objects-multiview": {
        "perceive_dino", "perceive_point", "perceive_dino_vlm",
        "merge", "select_best",
    },
    "perceiving-object-parts": {"perceive_subpart"},
}

#: Output-field contracts downstream graphs bind ``$ref``s against.
EXPECTED_SCRIPT_OUTPUTS: dict[tuple[str, str], set[str]] = {
    ("perceiving-objects", "perceive_dino_vlm"): {"found", "cloud", "mask", "score"},
    ("perceiving-objects-oneshot", "perceive_simple"): {"found", "cloud", "mask", "score"},
    ("perceiving-objects-multiview", "perceive_dino"): {"found", "cloud", "mask", "score"},
    ("perceiving-objects-multiview", "perceive_point"): {"found", "cloud", "mask", "score"},
    ("perceiving-objects-multiview", "perceive_dino_vlm"): {"found", "cloud", "mask", "score"},
    ("perceiving-objects-multiview", "merge"): {"cloud", "mask", "obb"},
    ("perceiving-objects-multiview", "select_best"): {"selected_index", "vlm_response"},
    ("perceiving-object-parts", "perceive_subpart"): {
        "found", "obb", "mask", "cloud", "subpart_mask", "score",
        "parent_obb", "parent_cloud",
    },
}

# Connector-owned tool names come from the engine-side validator
# (gap.skills.validate.connector_tool_names) — derived from the real
# connector code, so this suite can't drift from the runtime surface.

# A denser image than the (120, 160) default so back-projected point spacing
# (~depth/fx ≈ 4 mm) stays below the DBSCAN eps (5 mm) of the real geometry
# tools the scripts call.
_IMAGE_HW = (240, 320)
_CUBE_CENTER = (0.0, 0.0, 0.03)
_CUBE_SIZE = (0.06, 0.06, 0.06)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _mask_u8(mask_bool: np.ndarray) -> np.ndarray:
    return mask_bool.astype(np.uint8) * 255


def _bbox_of(mask_bool: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask_bool)
    return {
        "x1": float(xs.min()), "y1": float(ys.min()),
        "x2": float(xs.max() + 1), "y2": float(ys.max() + 1),
    }


def _centroid_of(mask_bool: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask_bool)
    return float(xs.mean()), float(ys.mean())


#: A decoy detection box in the sky corner of the synthetic frame, far from
#: the rendered objects (never contained in / containing the target box).
_DECOY_BOX = {"x1": 5.0, "y1": 5.0, "x2": 65.0, "y2": 65.0}


def _geometry_delegates(tool_registry) -> dict:
    """Route geometry.* FakeContext calls to the real (CPU) geometry tools."""
    names = (
        "geometry.mask_to_world_points",
        "geometry.filter_and_compute_obb",
        "geometry.filter_noise",
        "geometry.compute_obb",
    )
    return {
        name: (lambda _n: (lambda **kw: tool_registry.invoke(_n, **kw)))(name)
        for name in names
    }


def _script(skills_registry, bundle: str, name: str):
    return skills_registry.get(bundle).canonical_scripts[name].module


@pytest.fixture(scope="module")
def scene():
    obs, gt = make_test_observation(
        [("cube", _CUBE_CENTER, _CUBE_SIZE)], image_hw=_IMAGE_HW
    )
    return obs, gt


@pytest.fixture()
def po_module(skills_registry, monkeypatch):
    """perceiving-objects' canonical script with the result cache disabled."""
    mod = _script(skills_registry, "perceiving-objects", "perceive_dino_vlm")
    monkeypatch.setattr(mod, "_CACHE_ENABLED", False)
    return mod


def _assert_cloud_on_cube(points: np.ndarray) -> None:
    """The back-projected cloud must land on the ground-truth cube surface."""
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) > 200, "synthetic cube mask should yield a dense cloud"
    center = np.asarray(_CUBE_CENTER)
    half = np.asarray(_CUBE_SIZE) / 2.0 + 5e-3  # half-pixel + float32 slack
    assert np.all(points >= (center - half)[None, :])
    assert np.all(points <= (center + half)[None, :])


# ---------------------------------------------------------------------------
# Loader-level: discovery, frontmatter, allowed_tools, declared resources
# ---------------------------------------------------------------------------


def test_perception_bundles_discover_with_valid_frontmatter(skills_registry):
    for bundle, scripts in PERCEPTION_BUNDLES.items():
        info = skills_registry.get(bundle)
        assert info.kind == "skill"
        assert info.namespace == "skills"
        assert info.meta.name == bundle
        desc = info.meta.description
        assert 0 < len(desc) <= 1024
        assert "use when" in desc.lower()
        assert set(info.canonical_scripts) == scripts
        assert {"found", "not_found"} <= set(info.meta.exit_conditions)
        assert info.meta.streaming is False
        # produces_outputs use gap.types schema names, never proto-era names.
        assert info.meta.produces_outputs
        for type_name in info.meta.produces_outputs.values():
            assert type_name in {"OrientedBoundingBox", "Mask", "PointCloud"}


def test_perception_allowed_tools_all_exist(skills_registry):
    from gap.skills.validate import known_tool_names

    landed_tool_names = known_tool_names(
        [info.meta for info in skills_registry.list_skills()]
    )
    for bundle in PERCEPTION_BUNDLES:
        info = skills_registry.get(bundle)
        assert info.meta.allowed_tools, f"{bundle}: no allowed_tools declared"
        missing = set(info.meta.allowed_tools) - landed_tool_names
        assert not missing, (
            f"{bundle}: allowed_tools reference unknown tools {sorted(missing)}"
        )


def test_perception_declared_resources_exist_on_disk(skills_registry):
    """Engine-side format validation (the `gap skills check` rules) finds
    no errors — covers resource paths, gap: block shape, and type names."""
    from gap.skills.validate import validate_bundle_meta

    for bundle in PERCEPTION_BUNDLES:
        info = skills_registry.get(bundle)
        issues = validate_bundle_meta(info.meta, kind="skill", bundle_dir=info.bundle_dir)
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, f"{bundle}: {[str(i) for i in errors]}"


def test_perception_script_output_contracts(skills_registry):
    """Schema introspection must expose the exact output fields downstream
    graphs bind ``$ref``s against (same names as the legacy sources)."""
    for (bundle, script), expected in EXPECTED_SCRIPT_OUTPUTS.items():
        sinfo = skills_registry.get(bundle).canonical_scripts[script]
        assert set(sinfo.schema.outputs) == expected, (
            f"{bundle}::{script} output contract drifted"
        )


# ---------------------------------------------------------------------------
# perceiving-objects (DINO → pairwise VLM tournament → SAM3 → 3D fusion)
# ---------------------------------------------------------------------------


def test_perceiving_objects_happy_path(po_module, tool_registry, scene):
    obs, gt = scene
    cube = gt["cube"]["mask"]
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [{"box": _bbox_of(cube), "label": "object", "score": 0.9}],
        },
        "sam3.segment_box": {"masks": [_mask_u8(cube)], "scores": [0.95]},
        **_geometry_delegates(tool_registry),
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert set(out) == {"found", "cloud", "mask", "score"}
    assert out["found"] is True
    assert out["score"] == pytest.approx(0.95)
    assert out["mask"].dtype == np.uint8 and out["mask"].shape == _IMAGE_HW
    _assert_cloud_on_cube(out["cloud"]["points"])
    # Single detection -> the tournament short-circuits without a VLM call,
    # and segmentation went through the box prompt (not the text fallback).
    assert ctx.call_count("vlm.query") == 0
    assert ctx.call_count("sam3.segment_text") == 0


def test_perceiving_objects_pairwise_tournament_picks_vlm_winner(
    po_module, tool_registry, scene,
):
    obs, gt = scene
    cube = gt["cube"]["mask"]
    cube_box = _bbox_of(cube)
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [
                {"box": _DECOY_BOX, "label": "object", "score": 0.8},
                {"box": cube_box, "label": "object", "score": 0.7},
            ],
        },
        "vlm.query": {"text": "B"},  # the pairwise prompt: A=decoy, B=cube
        "sam3.segment_box": {"masks": [_mask_u8(cube)], "scores": [0.92]},
        **_geometry_delegates(tool_registry),
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    # Exactly one pairwise comparison for 2 crops, and the VLM's pick (the
    # cube box) is what got segmented.
    assert ctx.call_count("vlm.query") == 1
    seg_calls = ctx.calls_to("sam3.segment_box")
    assert len(seg_calls) == 1
    assert seg_calls[0].kwargs["box"] == cube_box
    _assert_cloud_on_cube(out["cloud"]["points"])


def test_perceiving_objects_not_found_returns_found_false(po_module, scene):
    obs, _ = scene
    ctx = FakeContext({
        "grounding-dino.detect": {"detections": []},
        "sam3.segment_text": {"masks": [], "scores": [], "boxes": []},
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="unicorn")

    assert out["found"] is False
    assert out["score"] == 0.0
    assert len(out["cloud"]["points"]) == 0
    assert out["mask"].size == 0
    # The text-prompt fallback net was tried before giving up.
    assert ctx.call_count("sam3.segment_text") == 1


def test_perceiving_objects_cache_roundtrip(
    skills_registry, tool_registry, scene, monkeypatch, tmp_path,
):
    """Second identical call must short-circuit on the pickle cache."""
    mod = _script(skills_registry, "perceiving-objects", "perceive_dino_vlm")
    monkeypatch.setattr(mod, "_CACHE_ENABLED", True)
    monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path)
    obs, gt = scene
    cube = gt["cube"]["mask"]
    responses = {
        "grounding-dino.detect": {
            "detections": [{"box": _bbox_of(cube), "label": "object", "score": 0.9}],
        },
        "sam3.segment_box": {"masks": [_mask_u8(cube)], "scores": [0.95]},
        **_geometry_delegates(tool_registry),
    }

    ctx1 = FakeContext(responses)
    out1 = mod.run(ctx1, cameras=obs["cameras"], object_name="red cube")
    assert out1["found"] is True
    assert ctx1.call_count("grounding-dino.detect") == 1

    ctx2 = FakeContext({})  # any tool call would raise ToolError
    out2 = mod.run(ctx2, cameras=obs["cameras"], object_name="red cube")
    assert out2["found"] is True
    assert out2["score"] == pytest.approx(out1["score"])
    np.testing.assert_array_equal(out2["mask"], out1["mask"])
    np.testing.assert_allclose(out2["cloud"]["points"], out1["cloud"]["points"])
    assert ctx2.calls == []


# ---------------------------------------------------------------------------
# perceiving-objects: `safe` wrist-fallback gate + object_description plumbing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dual_cam_scene():
    """Exterior + wrist camera pair (the wrist is a renamed copy of the
    exterior frame) so run() takes the `safe` gate path instead of the
    single-camera legacy path."""
    obs, gt = make_test_observation(
        [("cube", _CUBE_CENTER, _CUBE_SIZE)], image_hw=_IMAGE_HW
    )
    ext = obs["cameras"][0]
    wrist = dict(ext)
    wrist["name"] = "robot0_eye_in_hand"
    return {"cameras": [ext, wrist], "arms": obs["arms"]}, gt


_DESC = "shiny red plastic cube, ~6 cm wide"


def _dual_cam_responses(tool_registry, cube_mask) -> dict:
    return {
        "grounding-dino.detect": {
            "detections": [
                {"box": _DECOY_BOX, "label": "object", "score": 0.8},
                {"box": _bbox_of(cube_mask), "label": "object", "score": 0.7},
            ],
        },
        "vlm.query": {"text": "B"},
        "sam3.segment_box": {"masks": [_mask_u8(cube_mask)], "scores": [0.92]},
        **_geometry_delegates(tool_registry),
    }


def test_safe_gate_verified_exterior_pick_skips_wrist(
    po_module, tool_registry, dual_cam_scene,
):
    """Verify YES on the exterior pick -> exterior result, wrist never
    identified; object_description must reach BOTH the pairwise tournament
    prompt and the close-up verify prompt."""
    obs, gt = dual_cam_scene
    cube = gt["cube"]["mask"]
    ctx = FakeContext({
        **_dual_cam_responses(tool_registry, cube),
        "vlm.query_yes_no": {"answer": True, "text": "Yes, it matches."},
    })

    out = po_module.run(
        ctx, cameras=obs["cameras"], object_name="red cube",
        object_description=_DESC,
    )

    assert out["found"] is True
    _assert_cloud_on_cube(out["cloud"]["points"])
    # Exterior verified -> the wrist camera is never DINO-identified.
    assert ctx.call_count("grounding-dino.detect") == 1
    assert ctx.call_count("vlm.query_yes_no") == 1
    # The description is injected into the tournament prompt...
    (pair_call,) = ctx.calls_to("vlm.query")
    assert _DESC in pair_call.kwargs["prompt"]
    # ...and into the verify question, anchored on the described
    # shape/colors so the VLM does not reject on its category prior.
    (verify_call,) = ctx.calls_to("vlm.query_yes_no")
    assert verify_call.kwargs["prompt"] == (
        'Is the main object in this close-up a "red cube"? '
        f"It should look like: {_DESC}. "
        "Judge by the described shape and colors; printed text may "
        "be illegible at this resolution."
    )


def test_safe_gate_rejected_exterior_falls_to_verified_wrist(
    po_module, tool_registry, dual_cam_scene,
):
    """Verify NO on the exterior pick -> wrist identified; wrist verify YES
    -> the wrist result is used (single-view cloud)."""
    obs, gt = dual_cam_scene
    cube = gt["cube"]["mask"]
    ctx = FakeContext({
        **_dual_cam_responses(tool_registry, cube),
        "vlm.query_yes_no": [
            {"answer": False, "text": "No, it looks like a small book."},
            {"answer": True, "text": "Yes, that is the cube."},
        ],
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    _assert_cloud_on_cube(out["cloud"]["points"])
    # Both cameras DINO-identified (exterior, then the wrist fallback).
    assert ctx.call_count("grounding-dino.detect") == 2
    assert ctx.call_count("vlm.query_yes_no") == 2


def test_safe_gate_nothing_verified_keeps_conservative_exterior(
    po_module, tool_registry, dual_cam_scene,
):
    """Verify NO on both picks -> never fuse the unverified wrist; the
    exterior result is kept (the dev study's zero-regression policy)."""
    obs, gt = dual_cam_scene
    cube = gt["cube"]["mask"]
    ctx = FakeContext({
        **_dual_cam_responses(tool_registry, cube),
        "vlm.query_yes_no": {"answer": False, "text": "No."},
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    _assert_cloud_on_cube(out["cloud"]["points"])
    assert ctx.call_count("grounding-dino.detect") == 2
    assert ctx.call_count("vlm.query_yes_no") == 2
    # The conservative fallback reuses the exterior result — no extra
    # segment_text net is run since the exterior identify produced a result.
    assert ctx.call_count("sam3.segment_text") == 0


def test_verify_prompt_without_description_is_bare_question(
    po_module, tool_registry, dual_cam_scene,
):
    obs, gt = dual_cam_scene
    cube = gt["cube"]["mask"]
    ctx = FakeContext({
        **_dual_cam_responses(tool_registry, cube),
        "vlm.query_yes_no": {"answer": True, "text": "Yes."},
    })

    po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    (verify_call,) = ctx.calls_to("vlm.query_yes_no")
    assert verify_call.kwargs["prompt"] == (
        'Is the main object in this close-up a "red cube"?'
    )


# ---------------------------------------------------------------------------
# perceiving-objects: wrist-support cloud fusion on the verified-exterior path
# ---------------------------------------------------------------------------
# A single exterior view of a tall object yields a front-face sliver cloud
# whose OBB centre is biased toward the camera by half the object depth
# (measured 12-13 mm on LIBERO tall bottles/cartons -> off-centre pinch ->
# slip during transport). On the verified path, wrist views must contribute
# cloud GEOMETRY (dev-era multiview fusion, intersection-guarded) while the
# exterior pick stays authoritative for identity (mask/score).


@pytest.fixture(scope="module")
def dual_cam_two_obj_scene():
    """Exterior + wrist camera pair over a scene with the target cube and a
    spatially distant decoy box (for non-intersection cases)."""
    obs, gt = make_test_observation(
        [
            ("cube", _CUBE_CENTER, _CUBE_SIZE),
            ("decoy", (0.25, 0.0, 0.03), (0.06, 0.06, 0.06)),
        ],
        image_hw=_IMAGE_HW,
    )
    ext = obs["cameras"][0]
    wrist = dict(ext)
    wrist["name"] = "robot0_eye_in_hand"
    return {"cameras": [ext, wrist], "arms": obs["arms"]}, gt


def _ext_cloud_size(tool_registry, scene_obs, mask_bool) -> int:
    cloud = tool_registry.invoke(
        "geometry.mask_to_world_points",
        mask=_mask_u8(mask_bool),
        depth=scene_obs["cameras"][0]["depth"],
        intrinsics=scene_obs["cameras"][0]["intrinsics"],
        camera_pose=scene_obs["cameras"][0]["pose"],
    )["points"]
    return len(cloud["points"])


def test_wrist_support_fuses_intersecting_segment_text_cloud(
    po_module, tool_registry, dual_cam_two_obj_scene,
):
    """Wrist segment_text finds the same object -> its cloud is FUSED into
    the output; the exterior pick's mask/score stay authoritative even when
    the wrist score is higher (wrist masks live in a moving camera frame)."""
    obs, gt = dual_cam_two_obj_scene
    cube = gt["cube"]["mask"]
    ext_mask = _mask_u8(cube)
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [{"box": _bbox_of(cube), "label": "object", "score": 0.9}],
        },
        "sam3.segment_box": {"masks": [ext_mask], "scores": [0.92]},
        "sam3.segment_text": {"masks": [_mask_u8(cube)], "scores": [0.99]},
        "vlm.query_yes_no": {"answer": True, "text": "Yes."},
        **_geometry_delegates(tool_registry),
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    n_ext = _ext_cloud_size(tool_registry, obs, cube)
    # Both views contributed the full cube cloud (identical frames).
    assert len(out["cloud"]["points"]) == 2 * n_ext
    _assert_cloud_on_cube(out["cloud"]["points"])
    # Anchor (exterior) mask/score win despite the higher wrist score.
    np.testing.assert_array_equal(out["mask"], ext_mask)
    assert out["score"] == pytest.approx(0.92)
    # The wrist went through segment_text; no projection fallback needed.
    assert ctx.call_count("sam3.segment_text") == 1
    assert ctx.call_count("sam3.segment_box") == 1


def test_wrist_support_projection_fallback_seeds_segment_box(
    po_module, tool_registry, dual_cam_two_obj_scene,
):
    """Wrist segment_text finds nothing -> the exterior cloud is projected
    into the wrist frame and seeds sam3.segment_box there; the resulting
    cloud fuses when it intersects the exterior cloud."""
    obs, gt = dual_cam_two_obj_scene
    cube = gt["cube"]["mask"]
    ext_mask = _mask_u8(cube)
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [{"box": _bbox_of(cube), "label": "object", "score": 0.9}],
        },
        # 1st: exterior identify; 2nd: wrist projection-seeded fallback.
        "sam3.segment_box": [
            {"masks": [ext_mask], "scores": [0.92]},
            {"masks": [_mask_u8(cube)], "scores": [0.88]},
        ],
        "sam3.segment_text": {"masks": [], "scores": [], "boxes": []},
        "vlm.query_yes_no": {"answer": True, "text": "Yes."},
        **_geometry_delegates(tool_registry),
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    n_ext = _ext_cloud_size(tool_registry, obs, cube)
    assert len(out["cloud"]["points"]) == 2 * n_ext
    _assert_cloud_on_cube(out["cloud"]["points"])
    np.testing.assert_array_equal(out["mask"], ext_mask)
    # The projection seed box must cover the cube's wrist-frame projection
    # (identical camera copy -> same pixels as the exterior cube bbox).
    seg_calls = ctx.calls_to("sam3.segment_box")
    assert len(seg_calls) == 2
    seed = seg_calls[1].kwargs["box"]
    cube_box = _bbox_of(cube)
    assert seed["x1"] <= cube_box["x1"] + 5 and seed["x2"] >= cube_box["x2"] - 5
    assert seed["y1"] <= cube_box["y1"] + 5 and seed["y2"] >= cube_box["y2"] - 5


def test_wrist_support_rejects_non_intersecting_cloud(
    po_module, tool_registry, dual_cam_two_obj_scene,
):
    """A wrist segment of a DIFFERENT object (cloud does not intersect the
    verified exterior cloud) must never fuse — the dev-era multiview guard.
    The output stays exterior-only."""
    obs, gt = dual_cam_two_obj_scene
    cube = gt["cube"]["mask"]
    decoy = gt["decoy"]["mask"]
    ext_mask = _mask_u8(cube)
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [{"box": _bbox_of(cube), "label": "object", "score": 0.9}],
        },
        # 1st: exterior identify (cube); 2nd: wrist projection fallback
        # whose SAM grab lands on the decoy.
        "sam3.segment_box": [
            {"masks": [ext_mask], "scores": [0.92]},
            {"masks": [_mask_u8(decoy)], "scores": [0.95]},
        ],
        # Wrist segment_text also lands on the decoy: intersection guard
        # must reject it and try the projection fallback next.
        "sam3.segment_text": {"masks": [_mask_u8(decoy)], "scores": [0.9]},
        "vlm.query_yes_no": {"answer": True, "text": "Yes."},
        **_geometry_delegates(tool_registry),
    })

    out = po_module.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    n_ext = _ext_cloud_size(tool_registry, obs, cube)
    assert len(out["cloud"]["points"]) == n_ext  # exterior-only, no fusion
    _assert_cloud_on_cube(out["cloud"]["points"])
    np.testing.assert_array_equal(out["mask"], ext_mask)
    assert out["score"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# perceiving-objects-oneshot (DINO → one-shot VLM set-of-marks → SAM3)
# ---------------------------------------------------------------------------


def test_oneshot_happy_path(skills_registry, tool_registry, scene):
    mod = _script(skills_registry, "perceiving-objects-oneshot", "perceive_simple")
    obs, gt = scene
    cube = gt["cube"]["mask"]
    cube_box = _bbox_of(cube)
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [
                {"box": _DECOY_BOX, "label": "object", "score": 0.8},
                {"box": cube_box, "label": "object", "score": 0.7},
            ],
        },
        "vlm.query": {"text": "B"},
        "sam3.segment_box": {"masks": [_mask_u8(cube)], "scores": [0.91]},
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert set(out) == {"found", "cloud", "mask", "score"}
    assert out["found"] is True
    assert out["score"] == pytest.approx(0.91)
    # One set-of-marks pick; the picked letter (B) maps to the cube box.
    assert ctx.call_count("vlm.query") == 1
    assert ctx.calls_to("sam3.segment_box")[0].kwargs["box"] == cube_box
    _assert_cloud_on_cube(out["cloud"]["points"])


def test_oneshot_vlm_none_is_clean_not_found(skills_registry, scene):
    """The load-bearing loop-exit path: VLM 'none' -> found False, NO raise,
    and no segmentation/geometry work happens."""
    mod = _script(skills_registry, "perceiving-objects-oneshot", "perceive_simple")
    obs, gt = scene
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [
                {"box": _bbox_of(gt["cube"]["mask"]), "label": "object", "score": 0.9},
            ],
        },
        "vlm.query": {"text": "none"},
    })

    out = mod.run(ctx, cameras=obs["cameras"],
                  object_name="any grocery item on the floor")

    assert out["found"] is False
    assert out["score"] == 0.0
    assert len(out["cloud"]["points"]) == 0
    assert out["mask"].size == 0
    assert ctx.call_count("sam3.segment_box") == 0
    assert ctx.call_count("geometry.mask_to_world_points") == 0


def test_oneshot_no_detections_is_clean_not_found(skills_registry, scene):
    mod = _script(skills_registry, "perceiving-objects-oneshot", "perceive_simple")
    obs, _ = scene
    ctx = FakeContext({"grounding-dino.detect": {"detections": []}})

    out = mod.run(ctx, cameras=obs["cameras"], object_name="anything")

    assert out["found"] is False
    assert ctx.call_count("vlm.query") == 0


# ---------------------------------------------------------------------------
# perceiving-objects-multiview (3 detectors + VLM-disambiguated merge)
# ---------------------------------------------------------------------------


def test_multiview_perceive_point_happy_path(skills_registry, tool_registry, scene):
    mod = _script(skills_registry, "perceiving-objects-multiview", "perceive_point")
    obs, gt = scene
    cube = gt["cube"]["mask"]
    cx, cy = _centroid_of(cube)
    ctx = FakeContext({
        "molmo.point_prompt": {"pixel_x": cx, "pixel_y": cy, "found": True},
        "sam3.segment_point": {"masks": [_mask_u8(cube)], "scores": [0.9]},
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    assert out["score"] == pytest.approx(0.9)
    point_call = ctx.calls_to("sam3.segment_point")[0]
    assert point_call.kwargs["pixel_x"] == pytest.approx(cx)
    assert point_call.kwargs["pixel_y"] == pytest.approx(cy)
    _assert_cloud_on_cube(out["cloud"]["points"])


def test_multiview_perceive_point_molmo_offline_falls_back_to_text(
    skills_registry, tool_registry, scene,
):
    """No Molmo endpoint -> the point path degrades to sam3.segment_text
    instead of crashing the subgraph (the documented Molmo-optional path)."""
    mod = _script(skills_registry, "perceiving-objects-multiview", "perceive_point")
    obs, gt = scene
    cube = gt["cube"]["mask"]

    def _molmo_offline(**kw):
        raise ToolError("molmo.point_prompt", "GAP_MOLMO_BASE_URL is not set")

    ctx = FakeContext({
        "molmo.point_prompt": _molmo_offline,
        "sam3.segment_text": {
            "masks": [_mask_u8(cube)], "scores": [0.85], "boxes": [_bbox_of(cube)],
        },
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    assert out["score"] == pytest.approx(0.85)
    assert ctx.call_count("sam3.segment_text") == 1


def test_multiview_perceive_dino_happy_path(skills_registry, tool_registry, scene):
    mod = _script(skills_registry, "perceiving-objects-multiview", "perceive_dino")
    obs, gt = scene
    cube = gt["cube"]["mask"]
    cube_box = _bbox_of(cube)
    cx, cy = _centroid_of(cube)
    ctx = FakeContext({
        "molmo.point_prompt": {"pixel_x": cx, "pixel_y": cy, "found": True},
        "grounding-dino.detect": {
            "detections": [
                {"box": _DECOY_BOX, "label": "red cube", "score": 0.9},
                {"box": cube_box, "label": "red cube", "score": 0.6},
            ],
        },
        "sam3.segment_box": {"masks": [_mask_u8(cube)], "scores": [0.93]},
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    # The Molmo point selects the containing cube box (not the higher-scored
    # decoy) and refines SAM3 with point+box.
    seg_call = ctx.calls_to("sam3.segment_box")[0]
    assert seg_call.kwargs["box"] == cube_box
    assert seg_call.kwargs["use_point"] is True
    assert seg_call.kwargs["pixel_x"] == pytest.approx(cx)
    _assert_cloud_on_cube(out["cloud"]["points"])


def test_multiview_perceive_dino_vlm_happy_path(skills_registry, tool_registry, scene):
    mod = _script(skills_registry, "perceiving-objects-multiview", "perceive_dino_vlm")
    obs, gt = scene
    cube = gt["cube"]["mask"]
    cube_box = _bbox_of(cube)
    ctx = FakeContext({
        "grounding-dino.detect": {
            "detections": [
                {"box": _DECOY_BOX, "label": "object", "score": 0.8},
                {"box": cube_box, "label": "object", "score": 0.7},
            ],
        },
        "vlm.query": {"text": "B"},
        "sam3.segment_box": {"masks": [_mask_u8(cube)], "scores": [0.94]},
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(ctx, cameras=obs["cameras"], object_name="red cube")

    assert out["found"] is True
    assert ctx.calls_to("sam3.segment_box")[0].kwargs["box"] == cube_box
    _assert_cloud_on_cube(out["cloud"]["points"])


def test_multiview_merge_vlm_disambiguates_and_computes_obb(
    skills_registry, tool_registry, scene,
):
    mod = _script(skills_registry, "perceiving-objects-multiview", "merge")
    obs, gt = scene
    frame = obs["cameras"][0]
    cube = gt["cube"]["mask"]
    cloud = tool_registry.invoke(
        "geometry.mask_to_world_points",
        mask=_mask_u8(cube), depth=frame["depth"],
        intrinsics=frame["intrinsics"], camera_pose=frame["pose"],
    )["points"]

    dino_mask = _mask_u8(cube)
    point_mask = _mask_u8(cube)  # distinct array so selection is observable
    dino_result = {"found": True, "cloud": cloud, "mask": dino_mask, "score": 0.7}
    point_result = {"found": True, "cloud": cloud, "mask": point_mask, "score": 0.8}

    ctx = FakeContext({
        "vlm.query": {"text": "B"},  # pick the second candidate (point)
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(
        ctx, cameras=obs["cameras"], object_name="red cube",
        dino_result=dino_result, point_result=point_result,
        vlm_result={"found": False, "cloud": None, "mask": None, "score": 0.0},
    )

    assert set(out) == {"cloud", "mask", "obb"}
    # select_best ran exactly once and its pick (index 1) won.
    assert ctx.call_count("vlm.query") == 1
    assert out["mask"] is point_mask
    # The OBB comes from the real geometry.filter_and_compute_obb on the
    # winning cloud and must land on the ground-truth cube.
    obb = out["obb"]
    assert abs(obb["center"]["x"] - _CUBE_CENTER[0]) < 0.02
    assert abs(obb["center"]["y"] - _CUBE_CENTER[1]) < 0.02
    assert 0.0 < obb["extent"]["x"] < 0.08
    assert 0.0 < obb["extent"]["z"] < 0.08
    obb_call = ctx.calls_to("geometry.filter_and_compute_obb")[0]
    assert obb_call.kwargs["points"] is cloud


def test_multiview_merge_single_candidate_skips_vlm(
    skills_registry, tool_registry, scene,
):
    mod = _script(skills_registry, "perceiving-objects-multiview", "merge")
    obs, gt = scene
    frame = obs["cameras"][0]
    cube = gt["cube"]["mask"]
    cloud = tool_registry.invoke(
        "geometry.mask_to_world_points",
        mask=_mask_u8(cube), depth=frame["depth"],
        intrinsics=frame["intrinsics"], camera_pose=frame["pose"],
    )["points"]
    mask = _mask_u8(cube)

    ctx = FakeContext(_geometry_delegates(tool_registry))
    out = mod.run(
        ctx, cameras=obs["cameras"], object_name="red cube",
        dino_result={"found": True, "cloud": cloud, "mask": mask, "score": 0.9},
    )

    assert out["mask"] is mask
    assert ctx.call_count("vlm.query") == 0


def test_multiview_merge_raises_perception_failed_when_nothing_found(
    skills_registry, scene,
):
    mod = _script(skills_registry, "perceiving-objects-multiview", "merge")
    obs, _ = scene
    ctx = FakeContext({})

    with pytest.raises(PerceptionFailed, match="No perception path found"):
        mod.run(
            ctx, cameras=obs["cameras"], object_name="red cube",
            dino_result={"found": False, "cloud": None, "mask": None, "score": 0.0},
            point_result=None,
            vlm_result=None,
        )
    assert ctx.calls == []


def test_multiview_select_best_needs_two_candidates(skills_registry, scene):
    mod = _script(skills_registry, "perceiving-objects-multiview", "select_best")
    obs, gt = scene
    ctx = FakeContext({})

    out = mod.run(
        ctx, image=obs["cameras"][0]["rgb"],
        masks=[_mask_u8(gt["cube"]["mask"])], object_name="red cube",
    )

    assert out["selected_index"] == -1
    assert ctx.calls == []


# ---------------------------------------------------------------------------
# perceiving-object-parts (hierarchical parent → crop → subpart)
# ---------------------------------------------------------------------------

_PAN_CENTER = (0.0, 0.0, 0.03)
_PAN_SIZE = (0.12, 0.12, 0.06)
_HANDLE_CENTER = (0.11, 0.0, 0.045)
_HANDLE_SIZE = (0.10, 0.03, 0.03)


@pytest.fixture(scope="module")
def pan_scene():
    obs, gt = make_test_observation(
        [("pan", _PAN_CENTER, _PAN_SIZE), ("handle", _HANDLE_CENTER, _HANDLE_SIZE)],
        image_hw=_IMAGE_HW,
    )
    return obs, gt


def _clip_box(box: dict, img_h: int, img_w: int, pad: int) -> tuple[int, int, int, int]:
    """Mirror of the script's crop clipping, for building crop-frame fakes."""
    x1 = max(0, int(box["x1"]) - pad)
    y1 = max(0, int(box["y1"]) - pad)
    x2 = min(img_w, int(box["x2"]) + pad)
    y2 = min(img_h, int(box["y2"]) + pad)
    return x1, y1, x2, y2


def test_parts_happy_path(skills_registry, tool_registry, pan_scene):
    mod = _script(skills_registry, "perceiving-object-parts", "perceive_subpart")
    obs, gt = pan_scene
    h, w = _IMAGE_HW
    pan = gt["pan"]["mask"]
    handle = gt["handle"]["mask"]
    parent_mask_bool = pan | handle           # the whole physical pan
    parent_box = _bbox_of(parent_mask_bool)
    handle_box = _bbox_of(handle)

    # Crop frames the script will derive (padding_px=30 parent, 8 subpart).
    px1, py1, _, _ = _clip_box(parent_box, h, w, 30)
    hx1, hy1, hx2, hy2 = _clip_box(handle_box, h, w, 8)

    def fake_dino(image, query, **kw):
        if "handle" not in query:
            # Parent sweep ("frying pan.") on the full frame.
            assert image.shape[:2] == _IMAGE_HW
            return {"detections": [
                {"box": parent_box, "label": "frying pan", "score": 0.9},
            ]}
        # Subpart sweep on the parent crop: a whole-parent box that scores
        # HIGHER than the true handle box — the area filter must reject it.
        whole_in_crop = {
            "x1": parent_box["x1"] - px1, "y1": parent_box["y1"] - py1,
            "x2": parent_box["x2"] - px1, "y2": parent_box["y2"] - py1,
        }
        handle_in_crop = {
            "x1": handle_box["x1"] - px1, "y1": handle_box["y1"] - py1,
            "x2": handle_box["x2"] - px1, "y2": handle_box["y2"] - py1,
        }
        return {"detections": [
            {"box": whole_in_crop, "label": "handle", "score": 0.95},
            {"box": handle_in_crop, "label": "handle", "score": 0.8},
        ]}

    def fake_segment_text(image, query, **kw):
        # Called on the tight subpart crop; return the handle mask in
        # crop coordinates.
        assert image.shape[:2] == (hy2 - hy1, hx2 - hx1)
        return {
            "masks": [_mask_u8(handle[hy1:hy2, hx1:hx2])],
            "scores": [0.9],
            "boxes": [],
        }

    ctx = FakeContext({
        "grounding-dino.detect": fake_dino,
        "sam3.segment_box": {"masks": [_mask_u8(parent_mask_bool)], "scores": [0.9]},
        "sam3.segment_text": fake_segment_text,
        **_geometry_delegates(tool_registry),
    })

    out = mod.run(
        ctx, cameras=obs["cameras"],
        parent_prompt="frying pan",
        subpart_prompt="long horizontal handle of the frying pan",
    )

    assert set(out) == {
        "found", "obb", "mask", "cloud", "subpart_mask", "score",
        "parent_obb", "parent_cloud",
    }
    assert out["found"] is True
    assert out["score"] == pytest.approx(0.8)  # the true handle box's score

    # Single parent detection -> no VLM disambiguation needed.
    assert ctx.call_count("vlm.query") == 0

    # The subpart OBB sits on the ground-truth handle, NOT the pan.
    obb = out["obb"]
    assert abs(obb["center"]["x"] - _HANDLE_CENTER[0]) < 0.02
    assert abs(obb["center"]["y"] - _HANDLE_CENTER[1]) < 0.015
    assert _HANDLE_CENTER[2] - 0.03 < obb["center"]["z"] < _HANDLE_CENTER[2] + 0.03

    # The subpart cloud lands within the handle's gt extents (+ slack).
    pts = out["cloud"]["points"]
    assert len(pts) >= 10
    center = np.asarray(_HANDLE_CENTER)
    half = np.asarray(_HANDLE_SIZE) / 2.0 + 7e-3
    assert np.all(np.abs(pts - center[None, :]) <= half[None, :])

    # `mask` is the PARENT mask (collision isolation); the subpart's own
    # silhouette is `subpart_mask`.
    np.testing.assert_array_equal(out["mask"], _mask_u8(parent_mask_bool))
    sub_on = out["subpart_mask"] > 0
    assert sub_on.any()
    assert not sub_on[pan & ~handle].any(), "subpart mask leaked onto the pan body"

    # Parent geometry rides along for downstream placement reasoning.
    assert len(out["parent_cloud"]["points"]) > len(pts)
    assert out["parent_obb"]["extent"]["x"] > 0.0


def test_parts_not_found_returns_found_false(skills_registry, pan_scene):
    mod = _script(skills_registry, "perceiving-object-parts", "perceive_subpart")
    obs, _ = pan_scene
    ctx = FakeContext({"grounding-dino.detect": {"detections": []}})

    out = mod.run(
        ctx, cameras=obs["cameras"],
        parent_prompt="frying pan", subpart_prompt="handle",
    )

    assert out["found"] is False
    assert out["score"] == 0.0
    assert len(out["cloud"]["points"]) == 0
    assert out["mask"].size == 0 and out["subpart_mask"].size == 0
    assert out["obb"]["extent"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert out["parent_obb"]["extent"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert len(out["parent_cloud"]["points"]) == 0
