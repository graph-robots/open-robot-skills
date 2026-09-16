"""register_held's first pass bounds a loop observation by the carried pre-grasp loop.

From the wrist camera a wrench's jaw opening reads as a "circular ring end"
within a few hundredths of score of the ring itself, and sub-millimetre
differences in how the object sits in the jaws decide which ranks first
(tool_hanging/wrenches_level1 episodes 8/12/20: 7/7 wrong-end draws on one
runner, 0/2 on the other, the same observation up to a 2 px shift). A rigidly
grasped loop cannot be farther from where the pre-grasp perception left it than
the profile's ``maximum_translation_jump_m``, so the masks are tried best-first
and the first that fits within the bound is the observation; none fitting keeps
the carried prior at fallback confidence. Without a profile nothing changes.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1] / "skills"


def load(skill, script):
    path = ROOT / skill / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"test_first_pass_{skill}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self, handlers):
        self.handlers, self.calls = handlers, []

    def tool(self, name, **kwargs):
        self.calls.append((name, kwargs))
        handler = self.handlers[name]
        return handler(**kwargs) if callable(handler) else handler

    def calls_to(self, name):
        return [kwargs for called, kwargs in self.calls if called == name]


def pose(x=0.0, y=0.0, z=0.0):
    return {"position": {"x": x, "y": y, "z": z}, "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}


def wrist_camera(name="eye_in_hand_0"):
    return {
        "name": name,
        "rgb": np.zeros((8, 8, 3), np.uint8),
        "depth": np.ones((8, 8), np.float32),
        "intrinsics": [[1.0, 0.0, 4.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
        "pose": pose(),
    }


def ring_cloud(center, n=40, radius=0.015):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)]) + np.asarray(center)
    return {"points": pts.astype(np.float32)}


PRE_GRASP_RING = (0.780, -0.009, 0.747)   # what the pre-grasp perception said
RING_SEEN = (0.784, -0.008, 0.747)        # the ring, 4 mm from that
JAW_END = (0.562, -0.008, 0.743)          # the wrench's other end, 218 mm away

JAW_MASK = np.zeros((8, 8), np.uint8)
JAW_MASK[6:, 3:5] = 255
RING_MASK = np.full((8, 8), 255, np.uint8)


def _context(masks, scores, attachment_calls):
    def lift(mask, **_):
        # The ring mask is the full frame; the jaw mask a few pixels at the bottom.
        return {"points": ring_cloud(RING_SEEN) if int(np.count_nonzero(mask)) > 10 else ring_cloud(JAW_END)}

    def fit(points, **_):
        center = np.asarray(points["points"], dtype=np.float64).mean(axis=0)
        return {"pose": pose(*center), "radius": 0.015}

    def attach(**kwargs):
        attachment_calls.append(kwargs)
        return {"attached_object": {"frame": "tcp", "spheres": []}}

    return Context(
        {
            "robot.get_ee_pose": {"pose": pose()},
            "sam3.segment_text": {"masks": list(masks), "scores": list(scores)},
            "geometry.mask_to_world_points": lift,
            "geometry.fit_planar_feature": fit,
            "curobo.cloud_to_attachment": attach,
        }
    )


def _feature(profile):
    feature = {"kind": "loop", "description": "circular ring end of adjustable wrench", "pose": pose(*PRE_GRASP_RING)}
    if profile is not None:
        feature["registration_profile"] = profile
    return feature


PROFILE = {"strategy": "semantic_feature", "maximum_translation_jump_m": 0.025}


def test_the_ring_ranked_second_is_the_observation_when_the_first_mask_is_the_other_end():
    module = load("registering-held-objects", "register_held.py")
    calls = []
    ctx = _context([JAW_MASK, RING_MASK], [0.31, 0.29], calls)
    out = module.run(ctx, [wrist_camera()], ring_cloud(PRE_GRASP_RING), _feature(PROFILE), "wrench")
    assert out["registration_method"] == "semantic_feature" and not out["fallback_used"]
    assert out["registration_confidence"] == pytest.approx(0.29)  # the accepted mask's score, not the top one
    position = out["feature_in_tcp"]["position"]
    assert (position["x"], position["y"]) == pytest.approx(RING_SEEN[:2], abs=1e-6)
    # The jaw-end mask was lifted and fitted, rejected on the 218 mm jump, then the ring mask.
    assert len(ctx.calls_to("geometry.mask_to_world_points")) == 2
    fits = ctx.calls_to("geometry.fit_planar_feature")
    assert len(fits) == 2
    assert np.asarray(fits[0]["points"]["points"]).mean(axis=0)[0] == pytest.approx(JAW_END[0], abs=1e-6)
    assert np.asarray(fits[1]["points"]["points"]).mean(axis=0)[0] == pytest.approx(RING_SEEN[0], abs=1e-6)


def test_no_mask_within_the_bound_keeps_the_carried_pre_grasp_loop_at_fallback_confidence():
    module = load("registering-held-objects", "register_held.py")
    calls = []
    reference = ring_cloud(PRE_GRASP_RING)
    ctx = _context([JAW_MASK, JAW_MASK], [0.31, 0.29], calls)
    out = module.run(ctx, [wrist_camera()], reference, _feature(PROFILE), "wrench", attachment_source="observed")
    assert out["registration_method"] == "rigid_grasp_prior" and out["fallback_used"]
    assert out["registration_confidence"] == 0.25
    assert out["translation_uncertainty_m"] == pytest.approx(0.010)
    position = out["feature_in_tcp"]["position"]
    assert (position["x"], position["y"], position["z"]) == pytest.approx(PRE_GRASP_RING, abs=1e-9)
    # The rejected cloud never becomes the collision model: the attachment is fitted from the reference.
    assert calls[0]["points"] is reference
    assert len(ctx.calls_to("geometry.fit_planar_feature")) == 2


def test_a_mask_below_reliable_score_is_not_tried():
    module = load("registering-held-objects", "register_held.py")
    ctx = _context([JAW_MASK, RING_MASK], [0.31, 0.19], [])
    out = module.run(ctx, [wrist_camera()], ring_cloud(PRE_GRASP_RING), _feature(PROFILE), "wrench")
    assert out["registration_method"] == "rigid_grasp_prior"
    assert len(ctx.calls_to("geometry.fit_planar_feature")) == 1  # the 0.19 mask is never lifted


def test_the_top_mask_within_the_bound_is_accepted_with_the_same_calls_as_before():
    module = load("registering-held-objects", "register_held.py")
    ctx = _context([RING_MASK, JAW_MASK], [0.35, 0.19], [])
    out = module.run(ctx, [wrist_camera()], ring_cloud(PRE_GRASP_RING), _feature(PROFILE), "wrench")
    assert out["registration_method"] == "semantic_feature"
    assert out["registration_confidence"] == pytest.approx(0.35)
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(RING_SEEN[0], abs=1e-6)
    assert len(ctx.calls_to("geometry.mask_to_world_points")) == 1
    assert len(ctx.calls_to("geometry.fit_planar_feature")) == 1


def test_without_a_profile_the_top_mask_is_taken_as_before():
    module = load("registering-held-objects", "register_held.py")
    ctx = _context([JAW_MASK, RING_MASK], [0.31, 0.29], [])
    out = module.run(ctx, [wrist_camera()], ring_cloud(PRE_GRASP_RING), _feature(None), "wrench")
    assert out["registration_method"] == "semantic_feature"
    assert out["feature_in_tcp"]["position"]["x"] == pytest.approx(JAW_END[0], abs=1e-6)
    assert len(ctx.calls_to("geometry.fit_planar_feature")) == 1
