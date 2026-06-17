"""FakeContext-driven tests for the manipulation skill + policy bundles.

Each bundle's canonical scripts (or class-based callable) run against
:class:`gap.testing.FakeContext` with canned tool responses — no robot, no
GPU, no LLM. The assertions pin the behaviors the source bundles tuned:

- grasping-with-planner: full open → … → close call sequence, the planner
  kwargs (cad=0.005, ignore_obstacle_names, use_grasp_approach=False), and
  the PlanningFailed failure path;
- grasping-direct-ik: the align-then-descend pose progression (OBB top +
  0.15 m clearance, rotation preserved);
- grasping-short-axis: short-axis orientation math on an elongated OBB,
  the thin-bar Z clamp, and the base-offset slide;
- transporting-objects: drop-pose Z math (panda_hand_to_tcp=0.1029,
  margin=max(0.03, clearance)), yaw-only drop rotation, the
  lift-translate waypoint plan at z=0.45, and descend/release/retract
  sequencing;
- tracking-objects: tracker_init exactly once per session (statefulness
  across run() visits), per-tick ctx.publish snapshots, tracker_close in
  the finally;
- pi05-libero / molmoact-libero: PolicyLoopSkill wiring through a stub
  PolicyExecutor (no websocket), each using its own preset; gripper-cycle
  and VLM termination knobs. These are kind="policy" bundles (under
  policies/), not kind="skill".

Loader-level checks: skill bundles discover as kind="skill", policy
bundles as kind="policy"; declared allowed_tools stay within the known
tool catalog (connector names ∪ bundle registrations); the callable
bundles (tracking + both policies) register their @tool entries.
"""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import PlanningFailed
from gap.testing import FakeContext
from scipy.spatial.transform import Rotation

MANIPULATION_BUNDLES = (
    "grasping-with-planner",
    "grasping-direct-ik",
    "grasping-short-axis",
    "transporting-objects",
    "tracking-objects",
)

POLICY_BUNDLES = (
    "pi05-libero",
    "molmoact-libero",
)

# Connector tool names come from gap.skills.validate.connector_tool_names
# (derived from the real connector code) — no hand-maintained list here.


# ---------------------------------------------------------------------------
# Canned-data builders
# ---------------------------------------------------------------------------


def _vec3(x, y, z):
    return {"x": float(x), "y": float(y), "z": float(z)}


def _quat(w, x, y, z):
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def _pose(x, y, z, quat=(0.0, 1.0, 0.0, 0.0)):
    return {"position": _vec3(x, y, z), "rotation": _quat(*quat)}


def _obb(center, extent, quat=(1.0, 0.0, 0.0, 0.0)):
    return {
        "center": _vec3(*center),
        "extent": _vec3(*extent),
        "orientation": _quat(*quat),
    }


def _trajectory(n=3, dof=7):
    return {
        "waypoints": [
            {"positions": np.full(dof, float(i), dtype=np.float64)} for i in range(n)
        ]
    }


def _observation(ee_z=0.30):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "cameras": [{
            "name": "agentview",
            "rgb": rgb,
            "depth": np.ones((8, 8), dtype=np.float32),
            "intrinsics": np.eye(3),
            "pose": _pose(0.0, -0.7, 0.9),
        }],
        "arms": [{
            "joint_state": {"positions": np.zeros(7, dtype=np.float64)},
            "gripper_fraction": 1.0,
            "ee_pose": _pose(0.3, 0.0, ee_z),
        }],
    }


def _quat_to_matrix(q):
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()


def _script(skills_registry, bundle, name):
    return skills_registry.get(bundle).canonical_scripts[name].module


# ---------------------------------------------------------------------------
# Loader-level
# ---------------------------------------------------------------------------


def test_all_manipulation_bundles_discover(skills_registry):
    skill_names = {info.name for info in skills_registry.list_skills(kind="skill")}
    assert set(MANIPULATION_BUNDLES) <= skill_names
    for bundle in MANIPULATION_BUNDLES:
        info = skills_registry.get(bundle)
        assert info.kind == "skill"
        assert info.namespace == "skills"
        assert info.meta.description

    policy_names = {info.name for info in skills_registry.list_skills(kind="policy")}
    assert set(POLICY_BUNDLES) <= policy_names
    for bundle in POLICY_BUNDLES:
        info = skills_registry.get(bundle)
        assert info.kind == "policy"
        assert info.namespace == "policies"
        assert info.meta.description
        assert info.meta.serving is not None, (
            f"{bundle}: policy bundle has no gap.serving: block"
        )


def test_allowed_tools_are_known_names(skills_registry):
    from gap.skills.validate import known_tool_names

    known = known_tool_names([info.meta for info in skills_registry.list_skills()])
    for bundle in (*MANIPULATION_BUNDLES, *POLICY_BUNDLES):
        info = skills_registry.get(bundle)
        unknown = set(info.meta.allowed_tools) - known
        assert not unknown, f"{bundle}: unknown allowed_tools {sorted(unknown)}"


def test_canonical_scripts_have_schemas(skills_registry):
    # Script-shipping bundles: the four grasping/transporting ones.
    for bundle in MANIPULATION_BUNDLES[:4]:
        info = skills_registry.get(bundle)
        assert info.canonical_scripts, f"{bundle}: no canonical scripts"
        for name, script in info.canonical_scripts.items():
            assert script.schema.outputs, f"{bundle}::{name}: no output schema"


def test_callable_bundles_register_tools(skills_registry, tool_registry):
    assert "tracking-objects.track" in tool_registry
    assert "pi05-libero.run" in tool_registry
    assert "molmoact-libero.run" in tool_registry
    for bundle in ("tracking-objects", "pi05-libero", "molmoact-libero"):
        info = skills_registry.get(bundle)
        assert info.skill_class is not None, f"{bundle}: no Skill class wired"
        assert info.tools_module is not None
        assert set(info.meta.tools), f"{bundle}: SKILL.md declares no gap.tools"
    assert skills_registry.get("tracking-objects").meta.streaming is True
    assert skills_registry.get("pi05-libero").meta.streaming is False


# ---------------------------------------------------------------------------
# grasping-with-planner
# ---------------------------------------------------------------------------


class TestGraspingWithPlanner:
    TARGET_OBB = _obb((0.4, 0.1, 0.05), (0.04, 0.03, 0.05))

    def _happy_ctx(self, traj):
        return FakeContext({
            "robot.open_gripper": {"position": 1.0},
            "geometry.top_down_grasp_candidates": {
                "candidates": {"poses": [
                    _pose(0.4, 0.1, 0.06), _pose(0.4, 0.1, 0.06, (0.0, 0.0, 1.0, 0.0)),
                ]}
            },
            "robot.get_ee_pose": {"pose": _pose(0.3, 0.0, 0.3)},
            "robot.go_to_pose": None,
            "robot.get_observation": _observation(),
            "geometry.build_world_config": {
                "config": {"meshes": []}, "mesh_names": ["scene", "target"],
            },
            "curobo.plan_to_grasp_poses": {
                "success": True, "trajectory": traj, "goalset_index": 0,
            },
            "robot.execute_trajectory": None,
            "robot.close_gripper": {"position": 0.21},
        })

    def test_happy_path_call_sequence(self, skills_registry):
        traj = _trajectory()
        ctx = self._happy_ctx(traj)
        approach = _script(skills_registry, "grasping-with-planner", "approach_above")
        build_world = _script(skills_registry, "grasping-with-planner", "build_world")
        plan_grasp = _script(skills_registry, "grasping-with-planner", "plan_grasp")

        # The 8-state canonical flow, driven by hand.
        ctx.tool("robot.open_gripper", settle_steps=40)
        cands = ctx.tool("geometry.top_down_grasp_candidates", obb=self.TARGET_OBB)
        poses = cands["candidates"]["poses"]
        out = approach.run(
            ctx,
            target_position=poses[0]["position"],
            rotation=poses[0]["rotation"],
            target_obb=self.TARGET_OBB,
        )
        assert out == {"done": True}
        obs = ctx.tool("robot.get_observation")
        world = build_world.run(
            ctx, observation=obs,
            target_mask=np.full((8, 8), 255, dtype=np.uint8),
            target_obb=self.TARGET_OBB, target_name="target",
        )
        plan = plan_grasp.run(
            ctx, world_config=world["config"], observation=obs,
            grasp_poses=poses, target_name="target",
        )
        assert plan["trajectory"] is traj
        ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
        ctx.tool("robot.close_gripper", settle_steps=60)

        # Tuned approach behavior: three go_to_pose moves, all at the safe
        # height max(obb_top + 0.15, 0.35), the last pre-rotated to the
        # grasp rotation.
        gtp = ctx.calls_to("robot.go_to_pose")
        assert len(gtp) == 3
        expected_z = max(0.05 + 0.05 + 0.15, 0.35)
        for call in gtp:
            assert call.kwargs["pose"]["position"]["z"] == pytest.approx(expected_z)
        assert gtp[2].kwargs["pose"]["rotation"] == poses[0]["rotation"]

        # build_world preferred the pixel-accurate mask path.
        bw = ctx.calls_to("geometry.build_world_config")[0].kwargs
        assert bw["object_masks"][0]["name"] == "target"
        assert "target_obb" not in bw

        # Planner kwargs carry the tuned constants verbatim.
        pk = ctx.calls_to("curobo.plan_to_grasp_poses")[0].kwargs
        assert pk["collision_activation_distance"] == pytest.approx(0.005)
        assert pk["robot_collision_sphere_buffer"] == pytest.approx(-0.01)
        assert pk["ignore_obstacle_names"] == ["target"]
        assert pk["use_grasp_approach"] is False
        assert pk["grasp_pose_is_fingertip"] is True

        # Order: open before plan; execute after plan; close last with the
        # tuned settle constants.
        order = [c.tool for c in ctx.calls]
        assert order[0] == "robot.open_gripper"
        assert ctx.calls_to("robot.open_gripper")[0].kwargs["settle_steps"] == 40
        assert order.index("curobo.plan_to_grasp_poses") > order.index(
            "geometry.build_world_config"
        )
        assert order.index("robot.execute_trajectory") > order.index(
            "curobo.plan_to_grasp_poses"
        )
        assert order[-1] == "robot.close_gripper"
        assert ctx.calls_to("robot.close_gripper")[0].kwargs["settle_steps"] == 60
        assert ctx.calls_to("robot.execute_trajectory")[0].kwargs["trajectory"] is traj

    def test_planning_failure_raises(self, skills_registry):
        plan_grasp = _script(skills_registry, "grasping-with-planner", "plan_grasp")
        ctx = FakeContext({
            "curobo.plan_to_grasp_poses": {
                "success": False, "trajectory": None, "goalset_index": 0,
            },
        })
        with pytest.raises(PlanningFailed):
            plan_grasp.run(
                ctx, world_config={"meshes": []}, observation=_observation(),
                grasp_poses=[_pose(0.4, 0.1, 0.06)], target_name="target",
            )
        # Failure happens before any execution.
        assert ctx.call_count("robot.execute_trajectory") == 0

    def test_plan_grasp_autowraps_bare_pose(self, skills_registry):
        plan_grasp = _script(skills_registry, "grasping-with-planner", "plan_grasp")
        traj = _trajectory()
        ctx = FakeContext({
            "curobo.plan_to_grasp_poses": {
                "success": True, "trajectory": traj, "goalset_index": 0,
            },
        })
        plan_grasp.run(
            ctx, world_config={"meshes": []}, observation=_observation(),
            grasp_poses=_pose(0.4, 0.1, 0.06), target_name="target",
        )
        sent = ctx.calls_to("curobo.plan_to_grasp_poses")[0].kwargs["grasp_poses"]
        assert isinstance(sent, list) and len(sent) == 1


# ---------------------------------------------------------------------------
# grasping-direct-ik
# ---------------------------------------------------------------------------


class TestGraspingDirectIk:
    def test_descend_sequence_pose_progression(self, skills_registry):
        compute_align = _script(
            skills_registry, "grasping-direct-ik", "compute_align_pose"
        )
        target_obb = _obb((0.4, 0.1, 0.05), (0.04, 0.03, 0.06))
        grasp_pose = _pose(0.41, 0.12, 0.07, (0.0, 0.96, 0.0, 0.28))

        ctx = FakeContext({"robot.go_to_pose": None})
        out = compute_align.run(ctx, grasp_pose=grasp_pose, target_obb=target_obb)
        align = out["align_pose"]

        # Align pose: same XY + rotation as the grasp, Z = OBB top + 0.15.
        assert align["position"]["x"] == pytest.approx(0.41)
        assert align["position"]["y"] == pytest.approx(0.12)
        assert align["position"]["z"] == pytest.approx(0.05 + 0.06 + 0.15)
        assert align["rotation"] == grasp_pose["rotation"]
        # The script is pure math: no tool calls.
        assert ctx.calls == []

        # rotate_align then descend — straight-down progression with the
        # rotation already locked at the align pose.
        ctx.tool("robot.go_to_pose", pose=align)
        ctx.tool("robot.go_to_pose", pose=grasp_pose)
        gtp = ctx.calls_to("robot.go_to_pose")
        assert len(gtp) == 2
        first, second = gtp[0].kwargs["pose"], gtp[1].kwargs["pose"]
        assert first["rotation"] == second["rotation"]
        assert first["position"]["x"] == pytest.approx(second["position"]["x"])
        assert first["position"]["y"] == pytest.approx(second["position"]["y"])
        assert first["position"]["z"] > second["position"]["z"]
        assert second["position"]["z"] == pytest.approx(0.07)

    def test_plan_to_pose_failure_raises(self, skills_registry):
        plan_to_pose = _script(skills_registry, "grasping-direct-ik", "plan_to_pose")
        ctx = FakeContext({
            "curobo.plan_to_pose": {"success": False, "trajectory": None},
        })
        with pytest.raises(PlanningFailed):
            plan_to_pose.run(
                ctx, world_config={"meshes": []}, observation=_observation(),
                target_pose=_pose(0.4, 0.1, 0.2),
            )


# ---------------------------------------------------------------------------
# grasping-short-axis
# ---------------------------------------------------------------------------


class TestGraspingShortAxis:
    def test_short_axis_orientation_on_elongated_obb(self, skills_registry):
        compute = _script(skills_registry, "grasping-short-axis", "compute_grasp")
        # Handle along X: half-extents (0.10, 0.012, 0.012) — short
        # horizontal axis is Y, so the jaws must close across Y.
        handle = _obb((0.4, 0.1, 0.05), (0.10, 0.012, 0.012))
        ctx = FakeContext({})
        out = compute.run(ctx, target_obb=handle)
        pose = out["grasp_pose"]

        R = _quat_to_matrix(pose["rotation"])
        # Gripper Z points straight down world -Z.
        assert R[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
        # Finger-opening axis (tool Y) aligned with world Y (the short axis).
        assert abs(np.dot(R[:, 1], [0.0, 1.0, 0.0])) == pytest.approx(1.0)
        # Full rotation matrix: pi about Y (diag(-1, 1, -1)).
        assert R == pytest.approx(np.diag([-1.0, 1.0, -1.0]), abs=1e-9)

        # XY at the OBB centre.
        assert pose["position"]["x"] == pytest.approx(0.4)
        assert pose["position"]["y"] == pytest.approx(0.1)
        # Thin-bar clamp: top + z_offset = 0.062 - 0.04 = 0.022 plunges
        # below the OBB centre (0.05) — clamped to the centre, never into
        # the support surface.
        assert pose["position"]["z"] == pytest.approx(0.05)

    def test_handle_along_y_closes_across_x(self, skills_registry):
        compute = _script(skills_registry, "grasping-short-axis", "compute_grasp")
        handle = _obb((0.4, 0.1, 0.05), (0.012, 0.10, 0.012))
        out = compute.run(FakeContext({}), target_obb=handle)
        R = _quat_to_matrix(out["grasp_pose"]["rotation"])
        assert abs(np.dot(R[:, 1], [1.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_tall_object_keeps_z_offset_descent(self, skills_registry):
        compute = _script(skills_registry, "grasping-short-axis", "compute_grasp")
        # Tall bottle: top = 0.20, raw = 0.16 stays above the centre (0.10)
        # — no clamp, the tuned -0.04 fingertip descent is preserved.
        tall = _obb((0.4, 0.1, 0.10), (0.02, 0.012, 0.10))
        out = compute.run(FakeContext({}), target_obb=tall)
        assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.16)

    def test_offset_from_base_slides_outward(self, skills_registry):
        offset = _script(skills_registry, "grasping-short-axis", "offset_from_base")
        handle = _obb((0.4, 0.1, 0.05), (0.10, 0.012, 0.012))
        base = _obb((0.2, 0.1, 0.05), (0.10, 0.10, 0.04))
        grasp = _pose(0.4, 0.1, 0.05)
        out = offset.run(
            FakeContext({}), handle_obb=handle, grasp_pose=grasp, base_obb=base,
        )
        adj = out["adjusted_grasp"]
        # Slide along +X (base→handle) by 0.3 * long half-extent (0.10).
        assert adj["position"]["x"] == pytest.approx(0.4 + 0.03)
        assert adj["position"]["y"] == pytest.approx(0.1)
        assert adj["position"]["z"] == pytest.approx(0.05)
        assert adj["rotation"] == grasp["rotation"]

    def test_offset_from_base_noop_without_base(self, skills_registry):
        offset = _script(skills_registry, "grasping-short-axis", "offset_from_base")
        handle = _obb((0.4, 0.1, 0.05), (0.10, 0.012, 0.012))
        grasp = _pose(0.4, 0.1, 0.05)
        out = offset.run(FakeContext({}), handle_obb=handle, grasp_pose=grasp)
        assert out["adjusted_grasp"] is grasp

    def test_finalize_trajectory_converges_last_waypoint(self, skills_registry):
        finalize = _script(
            skills_registry, "grasping-short-axis", "finalize_trajectory"
        )
        traj = _trajectory(n=4)
        ctx = FakeContext({"robot.move_to_joints": None})
        out = finalize.run(ctx, trajectory=traj)
        assert out == {"done": True}
        mtj = ctx.calls_to("robot.move_to_joints")[0].kwargs
        assert mtj["joint_config"]["positions"] == list(
            traj["waypoints"][-1]["positions"]
        )
        assert mtj["max_steps"] == 120  # full convergence before close

    def test_per_pose_plan_retry_then_failure(self, skills_registry):
        plan_grasp = _script(skills_registry, "grasping-short-axis", "plan_grasp")
        ctx = FakeContext({
            "curobo.plan_to_grasp_poses": {
                "success": False, "trajectory": None, "goalset_index": 0,
            },
        })
        poses = [_pose(0.4, 0.1, 0.05), _pose(0.4, 0.1, 0.05, (0, 0, 1, 0))]
        with pytest.raises(PlanningFailed):
            plan_grasp.run(
                ctx, world_config={"meshes": []}, observation=_observation(),
                grasp_poses=poses, target_name="target", retries=2,
            )
        # Per-pose loop: 2 candidates x 2 retries, each a singleton goalset.
        calls = ctx.calls_to("curobo.plan_to_grasp_poses")
        assert len(calls) == 4
        assert all(len(c.kwargs["grasp_poses"]) == 1 for c in calls)


# ---------------------------------------------------------------------------
# transporting-objects
# ---------------------------------------------------------------------------


class TestTransportingObjects:
    CONTAINER = _obb((0.5, -0.2, 0.05), (0.10, 0.10, 0.05))
    HELD = _obb((0.4, 0.1, 0.03), (0.03, 0.03, 0.03))

    def test_compute_drop_pose_z_math(self, skills_registry):
        compute = _script(skills_registry, "transporting-objects", "compute_drop_pose")
        ee_at_grasp = _pose(0.4, 0.1, 0.15)  # top-down, yaw 0
        out = compute.run(
            FakeContext({"robot.get_ee_pose": {"pose": ee_at_grasp}}),
            container_obb=self.CONTAINER,
            held_obb=self.HELD,
            ee_pose_at_grasp=ee_at_grasp,
        )
        # container top = 0.10; margin = max(0.03, 0.05) = 0.05;
        # desired_obj_z = 0.10 + 0.05 + 0.03 = 0.18 (< ceiling 0.199);
        # ee_to_obj = live ee z 0.15 - 0.03 = 0.12 -> ee_z_at_drop = 0.30;
        # tcp = 0.30 - 0.1029 (panda hand->tcp). The at-grasp ee height is
        # measured LIVE (robot.get_ee_pose) — the wired ee_pose_at_grasp
        # is the yaw source and the fallback only.
        assert out["drop_position"]["x"] == pytest.approx(0.5)
        assert out["drop_position"]["y"] == pytest.approx(-0.2)
        assert out["drop_position"]["z"] == pytest.approx(0.30 - 0.1029)
        assert out["approach_pose"]["position"]["z"] == pytest.approx(
            out["drop_position"]["z"] + 0.20
        )
        # Yaw-only drop rotation: grasp was yaw-0 top-down, so the drop
        # rotation is the canonical top-down quat (up to sign).
        R = _quat_to_matrix(out["drop_pose"]["rotation"])
        assert R == pytest.approx(np.diag([1.0, -1.0, -1.0]), abs=1e-9)

    def test_compute_drop_pose_yaw_preserved(self, skills_registry):
        compute = _script(skills_registry, "transporting-objects", "compute_drop_pose")
        # Grasp rotation: top-down with a 45 deg wrist yaw.
        yaw = np.pi / 4
        R_grasp = (
            Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)
        ).as_quat()  # xyzw
        ee_at_grasp = {
            "position": _vec3(0.4, 0.1, 0.15),
            "rotation": _quat(R_grasp[3], R_grasp[0], R_grasp[1], R_grasp[2]),
        }
        out = compute.run(
            FakeContext({"robot.get_ee_pose": {"pose": ee_at_grasp}}),
            container_obb=self.CONTAINER,
            held_obb=self.HELD,
            ee_pose_at_grasp=ee_at_grasp,
        )
        R_drop = _quat_to_matrix(out["drop_pose"]["rotation"])
        # Still top-down (gripper Z down)...
        assert R_drop[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
        # ...with the grasp-time yaw preserved on the X axis.
        assert np.arctan2(R_drop[1, 0], R_drop[0, 0]) == pytest.approx(yaw)

    def test_compute_drop_pose_no_held_geometry_fallback(self, skills_registry):
        compute = _script(skills_registry, "transporting-objects", "compute_drop_pose")
        out = compute.run(FakeContext({}), container_obb=self.CONTAINER)
        # Legacy contract: TCP just above the rim by drop_clearance.
        assert out["drop_position"]["z"] == pytest.approx(0.10 + 0.05)

    def test_drop_offset_identity_when_parent_equals_held(self, skills_registry):
        drop_offset = _script(skills_registry, "transporting-objects", "drop_offset_pose")
        drop_pose = _pose(0.5, -0.2, 0.2)
        out = drop_offset.run(
            FakeContext({}),
            drop_pose=drop_pose,
            ee_pose_at_grasp=_pose(0.4, 0.1, 0.15),
            held_obb=self.HELD,
            parent_obb=self.HELD,
        )
        assert out["drop_position"]["x"] == pytest.approx(0.5)
        assert out["drop_position"]["y"] == pytest.approx(-0.2)
        assert out["drop_position"]["z"] == pytest.approx(0.2)

    def test_drop_offset_shifts_parent_centroid_to_target(self, skills_registry):
        drop_offset = _script(skills_registry, "transporting-objects", "drop_offset_pose")
        # Pan body 0.2 m further along -X than the grasped handle, same
        # top-down yaw at grasp and drop -> drop shifts +0.2 in X so the
        # body centroid lands on the original drop XY.
        parent = _obb((0.2, 0.1, 0.05), (0.10, 0.10, 0.04))
        out = drop_offset.run(
            FakeContext({}),
            drop_pose=_pose(0.5, -0.2, 0.2),
            ee_pose_at_grasp=_pose(0.4, 0.1, 0.15),
            held_obb=self.HELD,
            parent_obb=parent,
        )
        assert out["drop_position"]["x"] == pytest.approx(0.5 + 0.2)
        assert out["drop_position"]["y"] == pytest.approx(-0.2)
        assert out["drop_position"]["z"] == pytest.approx(0.2)

    def test_waypoint_move_two_cartesian_legs(self, skills_registry):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")
        ctx = FakeContext({
            "robot.get_observation": _observation(),
            "robot.go_to_pose_cartesian": None,
        })
        out = waypoint.run(ctx, drop_x=0.5, drop_y=-0.2)
        assert out == {"done": True}
        legs = ctx.calls_to("robot.go_to_pose_cartesian")
        assert len(legs) == 2
        # Leg 1: vertical lift at the CURRENT XY to the safe height.
        p1 = legs[0].kwargs["pose"]["position"]
        assert p1["z"] == pytest.approx(0.45)
        # Leg 2: lateral translate to the drop XY at constant height.
        p2 = legs[1].kwargs["pose"]["position"]
        assert (p2["x"], p2["y"], p2["z"]) == (0.5, -0.2, pytest.approx(0.45))
        assert p1["x"] != p2["x"] or p1["y"] != p2["y"]
        order = [c.tool for c in ctx.calls]
        assert order == [
            "robot.get_observation", "robot.go_to_pose_cartesian",
            "robot.go_to_pose_cartesian",
        ]

    def test_waypoint_move_failure_raises(self, skills_registry):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")
        def _second_leg_fails(**kwargs):
            if kwargs["pose"]["position"]["x"] == pytest.approx(0.5):
                raise RuntimeError("linear plan failed")
            return None

        ctx = FakeContext({
            "robot.get_observation": _observation(),
            "robot.go_to_pose_cartesian": _second_leg_fails,
        })
        with pytest.raises(PlanningFailed):
            waypoint.run(ctx, drop_x=0.5, drop_y=-0.2)

    def test_descend_release_sequencing(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "descend_release")
        drop_position = _vec3(0.5, -0.2, 0.2)
        drop_rotation = _quat(0.0, 1.0, 0.0, 0.0)
        ctx = FakeContext({
            "robot.go_to_pose": None,
            "robot.open_gripper": {"position": 1.0},
            "robot.go_home": None,
        })
        out = release.run(ctx, drop_position=drop_position, drop_rotation=drop_rotation)
        assert out["drop_position"] is drop_position
        # Strict descend -> release -> retract order, with the tuned
        # open-settle so the object lands before the retract.
        order = [c.tool for c in ctx.calls]
        assert order == ["robot.go_to_pose", "robot.open_gripper", "robot.go_home"]
        assert ctx.calls_to("robot.go_to_pose")[0].kwargs["pose"]["position"] is drop_position
        assert ctx.calls_to("robot.go_to_pose")[0].kwargs["pose"]["rotation"] is drop_rotation
        assert ctx.calls_to("robot.open_gripper")[0].kwargs["settle_steps"] == 60

    def test_descend_release_linear_routes_through_connector_cartesian(self, skills_registry):
        release = _script(
            skills_registry, "transporting-objects", "descend_release_linear"
        )
        ctx = FakeContext({
            "robot.go_to_pose_cartesian": None,
            "robot.open_gripper": {"position": 1.0},
            "robot.go_home": None,
        })
        release.run(ctx, drop_position=_vec3(0.5, -0.2, 0.2))
        order = [c.tool for c in ctx.calls]
        # Linear descent now goes through the connector's TCP-aware cartesian
        # tool; the cuRobo→plan_to_pose fallback lives inside the backend, so
        # the script no longer needs an explicit go_to_pose fallback path.
        assert order == [
            "robot.go_to_pose_cartesian",
            "robot.open_gripper",
            "robot.go_home",
        ]
        cart_pose = ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]
        assert cart_pose["position"] == _vec3(0.5, -0.2, 0.2)
        assert ctx.call_count("robot.execute_trajectory") == 0


# ---------------------------------------------------------------------------
# tracking-objects
# ---------------------------------------------------------------------------


class _StubStream:
    """Minimal observation_stream stand-in (.latest() only)."""

    def __init__(self, obs):
        self._obs = obs
        self.reads = 0

    def latest(self):
        self.reads += 1
        return self._obs


def _tracker_ctx(update_responses, init_present=True):
    mask = np.full((8, 8), 255, dtype=np.uint8)
    box = {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0}
    return FakeContext({
        "sam3.tracker_init": {
            "tracker_id": "trk-1" if init_present else "",
            "initial_mask": mask if init_present else None,
            "initial_box": box if init_present else None,
            "score": 0.9 if init_present else 0.0,
            "object_present": init_present,
        },
        "sam3.tracker_update": list(update_responses),
        "sam3.tracker_close": {"closed": True},
    })


def _upd(present=True, conf=0.8):
    return {
        "mask": np.full((8, 8), 255, dtype=np.uint8) if present else None,
        "box": {"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0} if present else None,
        "confidence": conf if present else 0.0,
        "object_present": present,
    }


class TestTrackingObjects:
    def test_init_once_then_updates_and_close(self, skills_registry):
        info = skills_registry.get("tracking-objects")
        skill = info.skill_class()
        stream = _StubStream(_observation())
        ctx = _tracker_ctx([_upd(conf=0.8), _upd(conf=0.7), _upd(conf=0.95)])

        out = skill.run(
            ctx, observation_stream=stream, target_prompt="red cup",
            max_updates=3, update_hz=1e6,
        )

        assert ctx.call_count("sam3.tracker_init") == 1
        assert ctx.call_count("sam3.tracker_update") == 3
        assert ctx.call_count("sam3.tracker_close") == 1
        init_kwargs = ctx.calls_to("sam3.tracker_init")[0].kwargs
        assert init_kwargs["text"] == "red cup"
        upd_kwargs = ctx.calls_to("sam3.tracker_update")[0].kwargs
        assert upd_kwargs["tracker_id"] == "trk-1"
        assert out["object_present"] is True
        assert out["n_updates"] == 3
        assert out["final_confidence"] == pytest.approx(0.95)

        # Streaming contract: one snapshot published per update tick.
        assert len(ctx.published) == 3
        assert ctx.published[-1]["n_updates"] == 3
        assert ctx.published[-1]["object_present"] is True

    def test_statefulness_resumes_session_across_runs(self, skills_registry):
        info = skills_registry.get("tracking-objects")
        skill = info.skill_class()
        stream = _StubStream(_observation())
        ctx = _tracker_ctx([_upd(), _upd()])

        skill.run(
            ctx, observation_stream=stream, target_prompt="red cup",
            max_updates=1, update_hz=1e6, close_on_exit=False,
        )
        skill.run(
            ctx, observation_stream=stream, target_prompt="red cup",
            max_updates=1, update_hz=1e6, close_on_exit=False,
        )
        # tracker_init ran exactly once across both visits; the second
        # visit resumed the open session.
        assert ctx.call_count("sam3.tracker_init") == 1
        assert ctx.call_count("sam3.tracker_update") == 2
        assert ctx.call_count("sam3.tracker_close") == 0

        skill.close(ctx)
        assert ctx.call_count("sam3.tracker_close") == 1
        skill.close(ctx)  # idempotent
        assert ctx.call_count("sam3.tracker_close") == 1

    def test_lost_streak_breaks_loop(self, skills_registry):
        info = skills_registry.get("tracking-objects")
        skill = info.skill_class()
        stream = _StubStream(_observation())
        ctx = _tracker_ctx([_upd(present=False)] * 5)
        out = skill.run(
            ctx, observation_stream=stream, target_prompt="red cup",
            max_updates=5, update_hz=1e6, allow_lost_frames=2,
        )
        assert ctx.call_count("sam3.tracker_update") == 2
        assert out["object_present"] is False
        # The last good (init) mask is held out as the final state.
        assert out["final_mask"] is not None
        assert ctx.call_count("sam3.tracker_close") == 1

    def test_seed_failure_returns_immediately(self, skills_registry):
        info = skills_registry.get("tracking-objects")
        skill = info.skill_class()
        stream = _StubStream(_observation())
        ctx = _tracker_ctx([], init_present=False)
        out = skill.run(
            ctx, observation_stream=stream, target_prompt="unicorn",
            max_updates=5, update_hz=1e6,
        )
        assert out["object_present"] is False
        assert out["n_updates"] == 0
        assert ctx.call_count("sam3.tracker_update") == 0
        # The tool already closed its failed session: nothing to close.
        assert ctx.call_count("sam3.tracker_close") == 0

    def test_tool_form_dispatches_through_registry(self, skills_registry, tool_registry):
        stream = _StubStream(_observation())
        ctx = _tracker_ctx([_upd()])
        out = tool_registry.invoke(
            "tracking-objects.track", ctx=ctx,
            observation_stream=stream, target_prompt="red cup",
            max_updates=1, update_hz=1e6,
        )
        assert out["n_updates"] == 1
        assert ctx.call_count("sam3.tracker_init") == 1
        assert ctx.call_count("sam3.tracker_close") == 1  # tool form self-closes


# ---------------------------------------------------------------------------
# pi05-libero / molmoact-libero (policy skills)
# ---------------------------------------------------------------------------


class _StubClient:
    """openpi WebsocketClientPolicy stand-in: scripted action chunks."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.infer_calls = []

    def infer(self, obs):
        self.infer_calls.append(obs)
        if not self._chunks:
            raise AssertionError("StubClient: scripted chunks exhausted")
        return {"actions": self._chunks.pop(0)}


class _StubPolicyExecutor:
    def __init__(self, client):
        self._client = client
        self.requested = []

    def client_for(self, policy_id):
        self.requested.append(policy_id)
        return self._client


def _policy_ctx(extra=None):
    responses = {"sim.apply_policy_action": None}
    responses.update(extra or {})
    return FakeContext(responses)


class TestPolicySkills:
    def _run(self, skills_registry, ctx, client, *, bundle="pi05-libero", **kwargs):
        info = skills_registry.get(bundle)
        skill = info.skill_class()
        executor = _StubPolicyExecutor(client)
        ctx.policy_executor = executor
        stream = _StubStream(_observation())
        defaults = dict(
            observation_stream=stream,
            prompt="pick up the cream cheese",
            settle_steps=0,
        )
        defaults.update(kwargs)
        out = skill.run(ctx, **defaults)
        return out, executor, stream

    def test_loop_wiring_max_windows(self, skills_registry):
        chunk = np.zeros((4, 7), dtype=np.float64)
        client = _StubClient([chunk, chunk])
        ctx = _policy_ctx()
        out, executor, stream = self._run(
            skills_registry, ctx, client, max_windows=2, replan_every=2,
        )
        assert out == {"status": "max_windows", "num_windows": 2, "num_steps": 4}
        # One client resolution through the executor's cache, keyed by the
        # bundle's own preset (the skill owns its model — no policy_id).
        assert executor.requested == ["pi05-libero"]
        # One stream read + one inference per window.
        assert stream.reads == 2
        assert len(client.infer_calls) == 2
        # The obs dict carries the prompt verbatim (openpi contract).
        assert client.infer_calls[0]["prompt"] == "pick up the cream cheese"
        # replan_every rows forwarded per window via the passthrough tool.
        assert ctx.call_count("sim.apply_policy_action") == 4
        first_action = ctx.calls_to("sim.apply_policy_action")[0].kwargs
        assert first_action["action"] == [0.0] * 7
        assert first_action["arm_id"] == 0

    def test_settle_steps_send_dummy_actions(self, skills_registry):
        chunk = np.zeros((1, 7), dtype=np.float64)
        client = _StubClient([chunk])
        ctx = _policy_ctx()
        self._run(
            skills_registry, ctx, client,
            max_windows=1, replan_every=1, settle_steps=3,
        )
        calls = ctx.calls_to("sim.apply_policy_action")
        assert len(calls) == 3 + 1
        # LIBERO dummy action: zero EE deltas + gripper open (-1).
        assert calls[0].kwargs["action"] == [0.0] * 6 + [-1.0]

    def test_gripper_cycle_termination(self, skills_registry):
        def chunk(grip):
            row = np.zeros(7, dtype=np.float64)
            row[-1] = grip
            return row[None, :]

        # close, hold, hold, open -> cycle completes on window 4.
        client = _StubClient([chunk(1.0), chunk(1.0), chunk(1.0), chunk(-1.0)])
        ctx = _policy_ctx()
        out, _, _ = self._run(
            skills_registry, ctx, client,
            max_windows=10, replan_every=1, gripper_cycle_termination=True,
        )
        assert out["status"] == "gripper_cycle"
        assert out["num_windows"] == 4
        assert out["num_steps"] == 4

    def test_momentary_close_does_not_fire_cycle(self, skills_registry):
        def chunk(grip):
            row = np.zeros(7, dtype=np.float64)
            row[-1] = grip
            return row[None, :]

        # A 1-window close (failed grasp) resets the detector.
        client = _StubClient([chunk(1.0), chunk(-1.0), chunk(0.0)])
        ctx = _policy_ctx()
        out, _, _ = self._run(
            skills_registry, ctx, client,
            max_windows=3, replan_every=1, gripper_cycle_termination=True,
        )
        assert out["status"] == "max_windows"

    def test_vlm_termination(self, skills_registry):
        chunk = np.zeros((1, 7), dtype=np.float64)
        client = _StubClient([chunk] * 3)
        ctx = _policy_ctx({"vlm.query_yes_no": {"answer": True, "text": "yes"}})
        out, _, _ = self._run(
            skills_registry, ctx, client,
            max_windows=10, replan_every=1,
            termination_prompt="is the object in the basket?", term_period=1,
        )
        assert out["status"] == "completed_by_vlm"
        assert out["num_windows"] == 1
        vlm = ctx.calls_to("vlm.query_yes_no")[0].kwargs
        assert vlm["prompt"] == "is the object in the basket?"

    def test_missing_policy_executor_raises(self, skills_registry):
        info = skills_registry.get("pi05-libero")
        skill = info.skill_class()
        ctx = _policy_ctx()
        ctx.policy_executor = None
        with pytest.raises(RuntimeError, match="PolicyExecutor"):
            skill.run(
                ctx,
                observation_stream=_StubStream(_observation()),
                prompt="pick",
            )

    def test_tool_form_dispatches_through_registry(self, skills_registry, tool_registry):
        chunk = np.zeros((1, 7), dtype=np.float64)
        client = _StubClient([chunk])
        ctx = _policy_ctx()
        ctx.policy_executor = _StubPolicyExecutor(client)
        out = tool_registry.invoke(
            "pi05-libero.run", ctx=ctx,
            observation_stream=_StubStream(_observation()),
            prompt="pick", settle_steps=0,
            max_windows=1, replan_every=1,
        )
        assert out["status"] == "max_windows"
        assert out["num_steps"] == 1

    def test_molmoact_uses_its_own_preset(self, skills_registry):
        chunk = np.zeros((1, 7), dtype=np.float64)
        client = _StubClient([chunk])
        ctx = _policy_ctx()
        _, executor, _ = self._run(
            skills_registry, ctx, client, bundle="molmoact-libero",
            max_windows=1, replan_every=1,
        )
        # Each policy skill resolves the client cache by its own preset
        # (bundle name), proving the model identity is per-skill.
        assert executor.requested == ["molmoact-libero"]
