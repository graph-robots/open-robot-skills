"""The sharps-disposal fold into two perception bundles: opt-in, and what each option does.

``perceive_directed_tip`` (perceiving-functional-features) and
``register_held`` (registering-held-objects) gained the candidate search,
gates, ranking, direction cascade and hold verification that the
``sharps_disposal/gap_perception_v2`` graph grew as local scripts. The deal is
the one ``test_promotion_is_behaviour_preserving`` states: every new parameter
defaults to what the script did before.

The first half proves it the same way: each script is driven as it was at
``BASE_REF`` and as it is now, through a recording :class:`FakeContext`, over a
grid of inputs with only pre-existing arguments, and the full tool-call
sequence (names and keyword values), the return value and any raised error
must match. ``perceive_directed_tip`` may add exactly the two new output keys.

The second half is the behaviour of each option on synthetic RGB-D scenes: an
orthographic overhead camera at 1 mm per pixel whose depth image *is* the
world z, so the depth rings the script reads see the same surfaces its masks
do.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from gap.testing import FakeContext
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]

#: The main commit the fold branched from: the scripts' pre-fold bodies.
BASE_REF = "752dfbd"
TIP = "skills/perceiving-functional-features/scripts/perceive_directed_tip.py"
HELD = "skills/registering-held-objects/scripts/register_held.py"

IDENTITY_ROTATION = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
RED = {"channel": 0, "min_value": 110, "min_margin": 40, "min_pixels": 8,
       "max_lateral_m": 0.010, "min_along_m": 0.025}


def _pose(x=0.0, y=0.0, z=0.0, rotation=None):
    rot = IDENTITY_ROTATION if rotation is None else rotation
    return {"position": {"x": x, "y": y, "z": z}, "rotation": dict(rot)}


def _rot(rotation: Rotation) -> dict[str, float]:
    q = rotation.as_quat()
    return {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}


def _load(source: str, name: str):
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _at_ref(rel: str, ref: str = BASE_REF) -> str:
    done = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        pytest.skip(f"{ref} is not in this checkout: {done.stderr.strip()}")
    return done.stdout


@pytest.fixture(scope="module")
def tip_before():
    return _load(_at_ref(TIP), "directed_tip_before")


@pytest.fixture(scope="module")
def tip_after():
    return _load((ROOT / TIP).read_text(), "directed_tip_after")


@pytest.fixture(scope="module")
def held_before():
    return _load(_at_ref(HELD), "register_held_before")


@pytest.fixture(scope="module")
def held_after():
    return _load((ROOT / HELD).read_text(), "register_held_after")


def _encode(value):
    if isinstance(value, np.ndarray):
        digest = hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest()
        return {"ndarray": list(value.shape), "dtype": str(value.dtype), "sha1": digest}
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _canon(value):
    return json.loads(json.dumps(value, sort_keys=True, default=_encode))


def _drive(module, ctx, *args, **kwargs):
    try:
        result, error = module.run(ctx, *args, **kwargs), None
    except Exception as exc:  # noqa: BLE001 -- the error is part of the comparison
        result, error = None, f"{type(exc).__name__}: {exc}"
    calls = [(record.tool, _canon(record.kwargs)) for record in ctx.calls]
    return calls, result, error


# ---------------------------------------------------------------------------
# A synthetic overhead RGB-D scene
# ---------------------------------------------------------------------------


class Scene:
    """Orthographic overhead view: pixel (row, col) is world (col mm, row mm, depth)."""

    def __init__(self, shape=(120, 200)):
        self.height = np.zeros(shape, dtype=np.float64)
        self.rgb = np.full((*shape, 3), 90, dtype=np.uint8)
        self.objects: list[tuple[np.ndarray, float]] = []
        self.markers: list[tuple[np.ndarray, float]] = []

    def rect(self, rows, cols) -> np.ndarray:
        mask = np.zeros(self.height.shape, dtype=np.uint8)
        mask[rows[0]:rows[1], cols[0]:cols[1]] = 1
        return mask

    def body(self, rows, cols, z=0.011) -> np.ndarray:
        self.height[rows[0]:rows[1], cols[0]:cols[1]] = z
        return self.rect(rows, cols)

    def paint(self, rows, cols, colour=(200, 20, 20)):
        self.rgb[rows[0]:rows[1], cols[0]:cols[1]] = colour

    def camera(self, name="overhead") -> dict:
        return {"name": name, "rgb": self.rgb, "depth": self.height,
                "intrinsics": np.eye(3), "pose": _pose()}

    def observation(self, name="overhead") -> dict:
        return {"cameras": [self.camera(name)]}

    def ctx(self) -> FakeContext:
        def segment(image, query, max_results):
            pool = self.markers if "cap" in query else self.objects
            pool = sorted(pool, key=lambda entry: -entry[1])
            if max_results > 0:
                pool = pool[:max_results]
            return {"masks": [m for m, _ in pool], "scores": [s for _, s in pool]}

        def backproject(mask, depth, intrinsics, camera_pose):
            rows, cols = np.nonzero(np.asarray(mask) > 0)
            points = np.column_stack([cols * 0.001, rows * 0.001, np.asarray(depth)[rows, cols]])
            return {"points": {"points": points.astype(np.float32)}}

        def obb(points):
            cloud = np.asarray(points["points"], dtype=np.float64)
            center = np.median(cloud, axis=0)
            extent = np.ptp(cloud, axis=0)
            return {"obb": {"center": dict(zip("xyz", map(float, center), strict=True)),
                            "extent": dict(zip("xyz", map(float, extent), strict=True)),
                            "orientation": dict(IDENTITY_ROTATION)}}

        return FakeContext({"sam3.segment_text": segment,
                            "geometry.mask_to_world_points": backproject,
                            "geometry.filter_and_compute_obb": obb})


def _syringe_with_flange(scene: Scene, score=0.9) -> np.ndarray:
    """Narrow rod on the left, a 24 mm flange, a 12 mm barrel to the right."""
    mask = scene.body((48, 52), (10, 70)) | scene.body((38, 62), (70, 74)) | scene.body((44, 56), (74, 174))
    scene.objects.append((mask, score))
    return mask


# ---------------------------------------------------------------------------
# Parity: perceive_directed_tip
# ---------------------------------------------------------------------------


def _tip_scene(body, object_score, second, marker):
    scene = Scene(shape=(40, 60))
    if body == "narrow_right":
        mask = scene.body((15, 25), (5, 44)) | scene.body((18, 22), (44, 56))
    elif body == "narrow_left":
        mask = scene.body((18, 22), (4, 16)) | scene.body((15, 25), (16, 55))
    elif body == "flat":
        mask = scene.body((15, 25), (5, 55))
    else:  # "few": fewer than 30 depth points
        mask = scene.body((15, 17), (5, 10))
    if object_score is not None:
        scene.objects.append((mask, object_score))
    if second:
        scene.objects.append((scene.body((30, 36), (10, 50)), 0.7))
    if marker == "cap":
        scene.markers.append((scene.rect((17, 22), (5, 9)), 0.5))
    elif marker == "few":
        scene.markers.append((scene.rect((17, 19), (5, 7)), 0.5))
    elif marker == "weak":
        scene.markers.append((scene.rect((17, 22), (5, 9)), 0.03))
    return scene


TIP_GRID = list(itertools.product(
    [0.9, 0.05, None],             # object score (None: SAM returns nothing)
    [False, True],                 # a second, lower-scored mask
    [None, "cap", "few", "weak"],  # marker detection
    ["", "red cap"],               # direction_marker_description
    ["overhead", "side"],          # the observation's camera name
    [{}, {"object_score_min": 0.5, "marker_score_min": 0.6}],
))


@pytest.mark.parametrize("body", ["narrow_right", "narrow_left", "flat", "few"])
def test_directed_tip_defaults_are_the_script_before_the_fold(tip_before, tip_after, body):
    mismatches = []
    for object_score, second, marker, description, camera, extra in TIP_GRID:
        runs = []
        for module in (tip_before, tip_after):
            scene = _tip_scene(body, object_score, second, marker)
            runs.append(_drive(module, scene.ctx(), scene.observation(camera), "syringe",
                               direction_marker_description=description, **extra))
        (calls_b, result_b, error_b), (calls_a, result_a, error_a) = runs
        if result_a is not None:
            added = set(result_a) - set(result_b or {})
            assert added == {"support_z", "jaw_clearance_m"}, added
            result_a = {k: v for k, v in result_a.items() if k not in added}
        case = (object_score, second, marker, description, camera, extra)
        if calls_b != calls_a or error_b != error_a or _canon(result_b) != _canon(result_a):
            mismatches.append(case)
    assert not mismatches, mismatches


def test_directed_tip_parity_grid_exercises_every_exit(tip_after):
    """The grid above is only proof if it reaches the found and every raising path."""
    seen = set()
    for body in ("narrow_right", "few"):
        for object_score, second, marker, description, camera, extra in TIP_GRID:
            scene = _tip_scene(body, object_score, second, marker)
            _, result, error = _drive(tip_after, scene.ctx(), scene.observation(camera), "syringe",
                                      direction_marker_description=description, **extra)
            seen.add("found" if result is not None else error.split(":")[1].strip()[:20])
    assert {"found", "observation has no c", "could not perceive '", "'syringe' mask conta"} <= seen


# ---------------------------------------------------------------------------
# Parity: register_held
# ---------------------------------------------------------------------------


def _rod_points(center, length=0.10, n=60, axis=0):
    t = np.linspace(-0.5, 0.5, n)
    points = np.zeros((n, 3))
    points[:, axis] = length * t
    others = [i for i in range(3) if i != axis]
    points[:, others[0]] = 0.004 * np.cos(7.0 * np.pi * t)
    points[:, others[1]] = 0.003 * np.sin(5.0 * np.pi * t)
    return points + np.asarray(center, dtype=np.float64)


def _held_scenario(scores, cloud_kind, marker_seen, hand_z):
    body = {
        "same": _rod_points((0.005, 0.0, 0.06)),
        "outlier": _rod_points((0.0, 0.0, 0.06), length=1.0),
        "far": _rod_points((0.0, 0.1, 0.06)),
        "few": _rod_points((0.0, 0.0, 0.06))[:5],
    }[cloud_kind]
    marker = np.column_stack([np.full(10, 0.05), np.zeros(10), np.linspace(0.055, 0.065, 10)])

    def segment(image, query, max_results):
        if query == "red cap":
            return {"masks": [np.full((8, 8), 2, np.uint8)], "scores": [0.7]} if marker_seen else {}
        return {"masks": [np.full((8, 8), 1, np.uint8) for _ in scores], "scores": list(scores)}

    def backproject(mask, depth, intrinsics, camera_pose):
        chosen = marker if int(np.max(mask)) == 2 else body
        return {"points": {"points": chosen.astype(np.float32)}}

    def fit(points, normal_hint, fit_circle_center):
        center = np.median(np.asarray(points["points"], dtype=np.float64), axis=0)
        tilt = Rotation.from_euler("x", 10, degrees=True) * Rotation.from_euler("z", 40, degrees=True)
        return {"pose": _pose(*map(float, center), rotation=_rot(tilt))}

    def attach(**kwargs):
        return {"attached_object": {"frame": "tcp", "spheres": [], "n": len(kwargs)}}

    return FakeContext({
        "robot.get_ee_pose": {"pose": _pose(z=hand_z)},
        "sam3.segment_text": segment,
        "geometry.mask_to_world_points": backproject,
        "geometry.fit_planar_feature": fit,
        "curobo.cloud_to_attachment": attach,
        "geometry.cloud_to_attachment": attach,
    })


def _camera(name):
    return {"name": name, "rgb": np.zeros((8, 8, 3), np.uint8), "depth": np.ones((8, 8)),
            "intrinsics": np.eye(3), "pose": _pose(z=0.4)}


HELD_GRID = list(itertools.product(
    [[0.9], [0.1], [0.03], []],            # wrist mask scores
    ["same", "outlier", "far", "few"],     # what the mask back-projects to
    [False, True],                         # marker described and seen
    [False, True],                         # grasp_pose given (hand lifted 20 cm)
))
HELD_CAMERAS = [[], ["eye_in_hand"], ["eye_in_hand", "overhead"], ["overhead"]]


@pytest.mark.parametrize("kind", ["tip", "loop", "shaft"])
@pytest.mark.parametrize("with_prior", [False, True])
def test_register_held_defaults_are_the_script_before_the_fold(held_before, held_after, kind, with_prior):
    mismatches = []
    for index, (scores, cloud_kind, marker, grasp) in enumerate(HELD_GRID):
        cameras = [_camera(name) for name in HELD_CAMERAS[index % len(HELD_CAMERAS)]]
        kwargs = {
            "attachment_source": ("reference", "observed")[index % 2],
            "attachment_fit_type": ("morphit", "morphit", "surface")[index % 3],
            "direction_marker_description": "red cap" if marker else "",
        }
        if grasp:
            kwargs["grasp_pose"] = _pose(z=0.15)
        if with_prior:
            kwargs["prior_feature_in_tcp"] = _pose(z=-0.25)
            kwargs["prior_object_in_tcp"] = _pose(z=-0.24)
            if index % 3 == 0:
                kwargs["prior_attached_object"] = {"frame": "tcp", "spheres": [], "arm_id": 0}
        if index % 5 == 0:
            kwargs["arm_id"] = 1
        if index % 7 == 0:
            kwargs["camera_name_filter"] = "overhead"
        feature = {"pose": _pose(z=0.05), "kind": kind, "description": "ring", "radius_outer": 0.003}
        reference = {"points": _rod_points((0.0, 0.0, 0.06)).astype(np.float32)}
        runs = []
        for module in (held_before, held_after):
            ctx = _held_scenario(scores, cloud_kind, marker, 0.35 if grasp else 0.3)
            runs.append(_drive(module, ctx, cameras, reference, feature, "syringe", **kwargs))
        if runs[0][0] != runs[1][0] or runs[0][2] != runs[1][2] or _canon(runs[0][1]) != _canon(runs[1][1]):
            mismatches.append((scores, cloud_kind, marker, grasp, sorted(kwargs)))
    assert not mismatches, mismatches


# ---------------------------------------------------------------------------
# perceive_directed_tip: each option
# ---------------------------------------------------------------------------


def _backprojected_sizes(ctx: FakeContext) -> list[int]:
    return [int(np.count_nonzero(r.kwargs["mask"])) for r in ctx.calls_to("geometry.mask_to_world_points")]


def test_defaults_report_cloud_support_and_no_clearance(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    ctx = scene.ctx()
    out = tip_after.run(ctx, scene.observation(), "syringe")
    assert out["support_z"] == pytest.approx(0.011, abs=1e-6)
    assert math.isnan(out["jaw_clearance_m"])
    assert ctx.call_count("geometry.mask_to_world_points") == 1


def test_support_ring_reads_the_surface_beside_the_mask(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    ctx = scene.ctx()
    out = tip_after.run(ctx, scene.observation(), "syringe", support_ring_px=14)
    assert out["support_z"] == pytest.approx(0.0, abs=1e-6)
    assert ctx.call_count("geometry.mask_to_world_points") == 2


def test_rescue_floor_accepts_a_low_score_mask_only_on_marker_colour(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.04))
    scene.paint((42, 48), (140, 150))
    with pytest.raises(ValueError, match="could not perceive 'syringe'"):
        tip_after.run(scene.ctx(), scene.observation(), "syringe")
    with pytest.raises(ValueError, match="could not perceive 'syringe'"):
        tip_after.run(scene.ctx(), scene.observation(), "syringe", candidate_score_floor=0.02)
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe",
                        candidate_score_floor=0.02, marker_rgb_rule=RED)
    assert out["marker_center"]["x"] == pytest.approx(0.1445, abs=1e-3)
    assert out["insertion_pose"]["position"]["x"] > 0.145


def test_overlapping_masks_keep_the_best_scored_and_size_gates_reject_fragments(tip_after):
    scene = Scene()
    whole = scene.body((40, 50), (30, 150))
    scene.objects += [(whole, 0.9), (scene.rect((40, 50), (130, 150)), 0.95)]
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", candidate_score_floor=0.02)
    assert len(out["target_cloud"]["points"]) == 200  # the fragment won and absorbed the whole
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe",
                        candidate_score_floor=0.02, min_span_m=0.08)
    assert len(out["target_cloud"]["points"]) == 1200
    ctx = scene.ctx()
    out = tip_after.run(ctx, scene.observation(), "syringe",
                        candidate_score_floor=0.02, min_mask_pixels=500)
    assert len(out["target_cloud"]["points"]) == 1200
    assert 200 not in _backprojected_sizes(ctx)  # gated before any depth was read


def test_width_gate_rejects_a_mask_merging_neighbours(tip_after):
    scene = Scene()
    single = scene.body((40, 50), (30, 150))
    scene.body((90, 100), (30, 150))
    merged = scene.rect((40, 100), (30, 150))
    scene.objects += [(single, 0.9), (merged, 0.97)]
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", candidate_score_floor=0.02)
    assert len(out["target_cloud"]["points"]) == 7200
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe",
                        candidate_score_floor=0.02, max_width_m=0.045, max_span_m=0.20)
    assert len(out["target_cloud"]["points"]) == 1200


def test_exclusion_zone_skips_a_delivered_object_and_its_marker(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    scene.objects.append((scene.body((100, 110), (30, 150), z=0.2), 0.95))
    scene.markers.append((scene.rect((102, 108), (140, 150)), 0.5))
    kwargs = {"direction_marker_description": "red cap", "candidate_score_floor": 0.02}
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", **kwargs)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.1045, abs=1e-3)
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", max_z=0.05, **kwargs)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0445, abs=1e-3)
    assert out["marker_center"]["y"] == pytest.approx(0.1045, abs=1e-3)  # a marker on the other one
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe",
                        # Covers the delivered one's centre (30.5 mm) and its marker (24.5 mm),
                        # not the target's (67 mm).
                        exclude_center={"x": 0.12, "y": 0.1045, "z": 0.0}, exclude_radius_m=0.04,
                        **kwargs)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0445, abs=1e-3)
    assert out["marker_center"]["y"] < 0.06


def test_raised_gate_rejects_a_flat_look_alike(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((100, 110), (30, 150), z=0.002), 0.95))
    with pytest.raises(ValueError, match="stands above its support"):
        tip_after.run(scene.ctx(), scene.observation(), "syringe", min_height_above_floor_m=0.005)
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe",
                        candidate_score_floor=0.02, min_height_above_floor_m=0.005)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0445, abs=1e-3)


def test_jaw_clearance_ranks_the_candidate_with_room_across_the_jaw(tip_after):
    scene = Scene(shape=(200, 200))
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    scene.objects.append((scene.body((60, 70), (30, 150)), 0.8))
    scene.body((75, 80), (0, 200))  # an undetected wall, 10.5 mm across B's axis
    kwargs = {"candidate_score_floor": 0.02, "jaw_clearance_enough_m": 0.045}
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", jaw_clearance_free_m=0.021, **kwargs)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0445, abs=1e-3)  # 15.5 mm beats 10.5 mm
    assert out["jaw_clearance_m"] == pytest.approx(0.0155, abs=1e-4)
    scene.objects.append((scene.body((150, 160), (30, 150)), 0.5))
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", candidate_score_floor=0.02)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0445, abs=1e-3)
    assert math.isnan(out["jaw_clearance_m"])
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", jaw_clearance_free_m=0.021, **kwargs)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.1545, abs=1e-3)
    assert out["jaw_clearance_m"] == pytest.approx(0.08)


def test_own_body_half_width_ignores_the_edge_the_mask_stopped_short_of(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    scene.body((50, 58), (30, 150))  # the object's own rounded edge, outside its mask
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", jaw_clearance_free_m=0.021)
    assert out["jaw_clearance_m"] == pytest.approx(0.0115, abs=1e-4)
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", jaw_clearance_free_m=0.021,
                        own_body_half_width_m=0.013)
    assert out["jaw_clearance_m"] == pytest.approx(0.08)


def test_marker_colour_off_the_axis_sends_a_candidate_last(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((80, 110), (30, 150)), 0.95))
    scene.paint((106, 110), (140, 150))
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", candidate_score_floor=0.02)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0945, abs=1e-3)
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe",
                        candidate_score_floor=0.02, marker_rgb_rule=RED)
    assert out["target_obb"]["center"]["y"] == pytest.approx(0.0445, abs=1e-3)


def test_direction_cascade_narrow_end_then_width_landmark(tip_after):
    scene = Scene()
    _syringe_with_flange(scene)
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe")
    assert out["insertion_pose"]["position"]["x"] < 0.015
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", width_landmark_to_marker_m=0.088)
    assert out["marker_center"]["x"] == pytest.approx(0.16, abs=0.004)
    assert out["insertion_pose"]["position"]["x"] > 0.165


def test_direction_cascade_marker_colour_outranks_the_detected_marker(tip_after):
    scene = Scene()
    _syringe_with_flange(scene)
    scene.markers.append((scene.rect((46, 54), (160, 174)), 0.5))
    scene.paint((48, 52), (10, 20))
    kwargs = {"direction_marker_description": "red cap"}
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", **kwargs)
    assert out["insertion_pose"]["position"]["x"] > 0.165
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", marker_rgb_rule=RED, **kwargs)
    assert out["marker_center"]["x"] == pytest.approx(0.0145, abs=1e-3)
    assert out["insertion_pose"]["position"]["x"] < 0.015


def test_detected_marker_off_the_axis_is_ignored_when_gated(tip_after):
    scene = Scene()
    _syringe_with_flange(scene)
    scene.markers.append((scene.rect((90, 100), (160, 174)), 0.5))
    kwargs = {"direction_marker_description": "red cap"}
    out = tip_after.run(scene.ctx(), scene.observation(), "syringe", **kwargs)
    assert out["marker_center"]["y"] == pytest.approx(0.0945, abs=1e-3)
    for gate in ({"marker_max_lateral_m": 0.012}, {"marker_max_gap_m": 0.025}):
        out = tip_after.run(scene.ctx(), scene.observation(), "syringe", **kwargs, **gate)
        assert out["insertion_pose"]["position"]["x"] < 0.015, gate


def test_axis_from_svd_does_not_tilt_toward_an_off_centre_marker(tip_after):
    scene = Scene()
    _syringe_with_flange(scene)
    scene.markers.append((scene.rect((60, 66), (160, 174)), 0.5))
    kwargs = {"direction_marker_description": "red cap"}
    tilted = tip_after.run(scene.ctx(), scene.observation(), "syringe", **kwargs)["functional_feature"]["axis"]
    assert abs(tilted["y"]) > 0.1
    axis = tip_after.run(scene.ctx(), scene.observation(), "syringe", axis_from_svd=True,
                         **kwargs)["functional_feature"]["axis"]
    assert abs(axis["y"]) < 0.02 and abs(axis["z"]) < 1e-6 and axis["x"] > 0.99


def test_marker_rgb_rule_must_be_complete(tip_after):
    scene = Scene()
    scene.objects.append((scene.body((40, 50), (30, 150)), 0.9))
    with pytest.raises(ValueError, match="marker_rgb_rule is missing"):
        tip_after.run(scene.ctx(), scene.observation(), "syringe", marker_rgb_rule={"channel": 0})


# ---------------------------------------------------------------------------
# register_held: each option
# ---------------------------------------------------------------------------

HAND = (0.5, 0.0, 0.30)


def _held_ctx(masks_by_camera: dict[str, list[tuple[int, float]]], clouds: dict[int, np.ndarray]):
    attachments: list[dict] = []

    def segment(image, query, max_results):
        entries = masks_by_camera.get(str(image[0, 0, 0]), [])
        return {"masks": [np.full((8, 8), key, np.uint8) for key, _ in entries],
                "scores": [score for _, score in entries]}

    def backproject(mask, **_):
        return {"points": {"points": clouds[int(np.max(mask))].astype(np.float32)}}

    def attach(**kwargs):
        attachments.append(kwargs)
        return {"attached_object": {"frame": "tcp", "spheres": []}}

    ctx = FakeContext({"robot.get_ee_pose": {"pose": _pose(*HAND)},
                       "sam3.segment_text": segment,
                       "geometry.mask_to_world_points": backproject,
                       "curobo.cloud_to_attachment": attach})
    return ctx, attachments


def _named_camera(name, tag):
    camera = _camera(name)
    camera["rgb"] = np.full((8, 8, 3), tag, np.uint8)  # lets the fake tell cameras apart
    return camera


def _tray_scene(with_blob=False):
    """A held rod 5 mm off its grasp-time pose, a higher-scored spare 80 mm from the hand."""
    held = _rod_points((0.505, 0.0, 0.25), length=0.12, axis=1)
    spare = _rod_points((0.5, 0.08, 0.30), length=0.12, axis=1)
    blob = np.random.default_rng(3).uniform(-0.005, 0.005, (40, 3)) + [0.5, 0.0, 0.29]
    entries = [(2, 0.9), (1, 0.3)] + ([(3, 0.5)] if with_blob else [])
    ctx, attachments = _held_ctx({"1": entries}, {1: held, 2: spare, 3: blob})
    cameras = [_named_camera("eye_in_hand", 1)]
    reference = {"points": _rod_points((0.5, 0.0, 0.25), length=0.12, axis=1)}
    feature = {"pose": _pose(0.5, 0.0, 0.25), "kind": "tip"}
    return ctx, attachments, cameras, reference, feature


def test_nearest_to_hand_search_takes_the_held_object_not_the_top_score(held_after):
    ctx, _, cameras, reference, feature = _tray_scene()
    out = held_after.run(ctx, cameras, reference, feature, "syringe")
    assert out["registration_confidence"] == 0.25  # the spare's cloud moved the feature 9 cm: rejected
    ctx, attachments, cameras, reference, feature = _tray_scene()
    out = held_after.run(ctx, cameras, reference, feature, "syringe", attachment_source="observed",
                         held_candidate_cameras=["eye_in_hand"], held_score_min=0.005)
    assert out["registration_confidence"] == pytest.approx(0.3)
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(0.005, abs=1e-6)
    # The attachment is fitted to the held cloud (y 0), not the spare's (y 0.08).
    assert np.median(np.asarray(attachments[0]["points"]["points"]), axis=0)[1] == pytest.approx(0.0, abs=1e-3)
    assert {r.kwargs["max_results"] for r in ctx.calls_to("sam3.segment_text")} == {0}
    ctx, _, cameras, reference, feature = _tray_scene()
    out = held_after.run(ctx, cameras, reference, feature, "syringe",
                         held_candidate_cameras=["overhead"], held_score_min=0.005)
    assert out["registration_confidence"] == 0.25 and ctx.call_count("sam3.segment_text") == 0


def test_rod_shape_gate_skips_a_nearer_blob(held_after):
    ctx, _, cameras, reference, feature = _tray_scene(with_blob=True)
    out = held_after.run(ctx, cameras, reference, feature, "syringe",
                         held_candidate_cameras=["eye_in_hand"], held_score_min=0.005)
    assert out["registration_confidence"] == 0.25  # the blob was nearest, then failed the extent check
    ctx, _, cameras, reference, feature = _tray_scene(with_blob=True)
    out = held_after.run(ctx, cameras, reference, feature, "syringe",
                         held_candidate_cameras=["eye_in_hand"], held_score_min=0.005,
                         held_extent_m=0.10, rod_length_range_m=[0.075, 0.18], rod_aspect_min=3.0)
    assert out["registration_confidence"] == pytest.approx(0.3)


def test_presence_only_never_shifts_the_carried_feature(held_after):
    ctx, _, cameras, reference, feature = _tray_scene()
    out = held_after.run(ctx, cameras, reference, feature, "syringe", presence_only=True,
                         held_candidate_cameras=["eye_in_hand"], held_score_min=0.005)
    assert out["registration_confidence"] == pytest.approx(0.3)
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(0.0, abs=1e-9)
    assert out["feature_in_tcp"]["position"]["z"] == pytest.approx(-0.05, abs=1e-9)
    assert out["registration_method"] == "rigid_grasp_prior"
    with pytest.raises(ValueError, match="presence_only"):
        held_after.run(ctx, cameras, reference, dict(feature, registration_profile={"query": "x"}),
                       "syringe", presence_only=True)


def test_require_observed_raises_instead_of_the_low_confidence_fallback(held_after):
    ctx, _, _, reference, feature = _tray_scene()
    out = held_after.run(ctx, [], reference, feature, "syringe")
    assert out["registration_confidence"] == 0.25
    with pytest.raises(RuntimeError, match="not observed"):
        held_after.run(ctx, [], reference, feature, "syringe", require_observed=True)
    ctx, _, cameras, reference, feature = _tray_scene()
    out = held_after.run(ctx, cameras, reference, feature, "syringe", require_observed=True,
                         held_candidate_cameras=["eye_in_hand"], held_score_min=0.005)
    assert out["registration_confidence"] == pytest.approx(0.3)


@pytest.mark.parametrize("tilt_deg,snap_deg,snapped", [(5, 15, True), (5, 0, False), (20, 15, False)])
def test_snap_feature_axis_to_the_nearest_hand_axis(held_after, tilt_deg, snap_deg, snapped):
    ctx, _ = _held_ctx({}, {})
    rotation = Rotation.from_euler("x", -90 + tilt_deg, degrees=True)
    feature = {"pose": _pose(*HAND, rotation=_rot(rotation)), "kind": "tip"}
    reference = {"points": _rod_points(HAND, axis=1)}
    out = held_after.run(ctx, [], reference, feature, "syringe", snap_feature_axis_deg=snap_deg)
    q = out["feature_in_tcp"]["rotation"]
    z_axis = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).apply([0.0, 0.0, 1.0])
    assert np.allclose(z_axis, [0.0, 1.0, 0.0], atol=1e-9) is snapped
    assert np.degrees(np.arccos(np.clip(z_axis[1], -1, 1))) == pytest.approx(0.0 if snapped else tilt_deg, abs=1e-6)
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_new_parameters_are_declared_with_their_opt_out_defaults(skills_registry):
    tip = skills_registry.get("perceiving-functional-features").canonical_scripts["perceive_directed_tip"]
    inputs = tip.schema.inputs
    for name, default in {"max_results": 3, "candidate_score_floor": None, "min_mask_pixels": 0,
                          "max_span_m": None, "exclude_center": None, "jaw_clearance_free_m": None,
                          "marker_rgb_rule": None, "width_landmark_to_marker_m": None,
                          "axis_from_svd": False, "support_ring_px": 0}.items():
        assert inputs[name].default == default, name
    held = skills_registry.get("registering-held-objects")
    inputs = held.canonical_scripts["register_held"].schema.inputs
    for name, default in {"held_candidate_cameras": None, "presence_only": False,
                          "snap_feature_axis_deg": 0.0, "require_observed": False}.items():
        assert inputs[name].default == default, name
    assert set(held.meta.exit_conditions) == {"registered", "lost"}
