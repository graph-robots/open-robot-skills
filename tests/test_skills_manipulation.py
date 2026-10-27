"""FakeContext-driven tests for the manipulation skill + policy bundles.

Each bundle's canonical scripts (or class-based callable) run against
:class:`gap.testing.FakeContext` with canned tool responses — no robot, no
GPU, no LLM. The assertions pin the behaviors the source bundles tuned:

- grasping-with-planner: full open → … → close call sequence, the planner
  kwargs (cad=0.005, ignore_obstacle_names, use_grasp_approach=False), and
  the PlanningFailed failure path;
- grasping-direct-ik: the align-then-descend pose progression (clearance
  pinned or asked of robot.describe_workspace, measured from the higher of
  OBB top and grasp), and the close-axis grasp refinement through
  robot.describe_gripper / robot.grasp_frame with its fingertip floor;
- grasping-short-axis: short-axis orientation math on an elongated OBB,
  the thin-bar Z clamp, and the base-offset slide;
- transporting-objects: drop-pose Z math (wrist-to-TCP from
  robot.describe_arm's tcp_offset on the hand's approach axis,
  margin=max(0.03, clearance)), yaw-only drop rotation composed by
  robot.grasp_frame, the lift-translate waypoint plan at the workspace's
  transport_z with its reach-limit descent ladder, container-derived
  transport/hover heights with the reached-pose gate and the planned-leg
  fallback, and descend/release/retract sequencing;
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
from gap.testing import FakeContext
from gap_core.errors import PlanningFailed
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
        "waypoints": [{"positions": np.full(dof, float(i), dtype=np.float64)} for i in range(n)]
    }


def _observation(ee_z=0.30):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "cameras": [
            {
                "name": "agentview",
                "rgb": rgb,
                "depth": np.ones((8, 8), dtype=np.float32),
                "intrinsics": np.eye(3),
                "pose": _pose(0.0, -0.7, 0.9),
            }
        ],
        "arms": [
            {
                "joint_state": {"positions": np.zeros(7, dtype=np.float64)},
                "gripper_fraction": 1.0,
                "ee_pose": _pose(0.3, 0.0, ee_z),
            }
        ],
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
        assert info.meta.serving is not None, f"{bundle}: policy bundle has no gap.serving: block"


def test_allowed_tools_are_known_names(skills_registry):
    from gap.skills.validate import known_tool_names

    known = known_tool_names([info.meta for info in skills_registry.list_skills()])
    for bundle in (*MANIPULATION_BUNDLES, *POLICY_BUNDLES):
        info = skills_registry.get(bundle)
        # A richer connector's tools count as resolvable once the bundle
        # declares them under gap.requires.connector (the checker's rule).
        declared = set(info.meta.requires.connector) if info.meta.requires else set()
        unknown = set(info.meta.allowed_tools) - known - declared
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
        return FakeContext(
            {
                "robot.open_gripper": {"position": 1.0},
                "geometry.top_down_grasp_candidates": {
                    "candidates": {
                        "poses": [
                            _pose(0.4, 0.1, 0.06),
                            _pose(0.4, 0.1, 0.06, (0.0, 0.0, 1.0, 0.0)),
                        ]
                    }
                },
                "robot.get_ee_pose": {"pose": _pose(0.3, 0.0, 0.3)},
                "robot.go_to_pose": None,
                "robot.get_observation": _observation(),
                "geometry.build_world_config": {
                    "config": {"meshes": []},
                    "mesh_names": ["scene", "target"],
                },
                "curobo.plan_to_grasp_poses": {
                    "success": True,
                    "trajectory": traj,
                    "goalset_index": 0,
                },
                "robot.execute_trajectory": None,
                "robot.close_gripper": {"position": 0.21},
            }
        )

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
            ctx,
            observation=obs,
            target_mask=np.full((8, 8), 255, dtype=np.uint8),
            target_obb=self.TARGET_OBB,
            target_name="target",
        )
        plan = plan_grasp.run(
            ctx,
            world_config=world["config"],
            observation=obs,
            grasp_poses=poses,
            target_name="target",
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
        assert order.index("robot.execute_trajectory") > order.index("curobo.plan_to_grasp_poses")
        assert order[-1] == "robot.close_gripper"
        assert ctx.calls_to("robot.close_gripper")[0].kwargs["settle_steps"] == 60
        assert ctx.calls_to("robot.execute_trajectory")[0].kwargs["trajectory"] is traj

    def test_planning_failure_raises(self, skills_registry):
        plan_grasp = _script(skills_registry, "grasping-with-planner", "plan_grasp")
        ctx = FakeContext(
            {
                "curobo.plan_to_grasp_poses": {
                    "success": False,
                    "trajectory": None,
                    "goalset_index": 0,
                },
            }
        )
        with pytest.raises(PlanningFailed):
            plan_grasp.run(
                ctx,
                world_config={"meshes": []},
                observation=_observation(),
                grasp_poses=[_pose(0.4, 0.1, 0.06)],
                target_name="target",
            )
        # Failure happens before any execution.
        assert ctx.call_count("robot.execute_trajectory") == 0

    def test_plan_grasp_autowraps_bare_pose(self, skills_registry):
        plan_grasp = _script(skills_registry, "grasping-with-planner", "plan_grasp")
        traj = _trajectory()
        ctx = FakeContext(
            {
                "curobo.plan_to_grasp_poses": {
                    "success": True,
                    "trajectory": traj,
                    "goalset_index": 0,
                },
            }
        )
        plan_grasp.run(
            ctx,
            world_config={"meshes": []},
            observation=_observation(),
            grasp_poses=_pose(0.4, 0.1, 0.06),
            target_name="target",
        )
        sent = ctx.calls_to("curobo.plan_to_grasp_poses")[0].kwargs["grasp_poses"]
        assert isinstance(sent, list) and len(sent) == 1


# ---------------------------------------------------------------------------
# grasping-direct-ik
# ---------------------------------------------------------------------------


class TestGraspingDirectIk:
    # Top face at 0.11, support (bottom face) at -0.01.
    OBB = _obb((0.4, 0.1, 0.05), (0.04, 0.03, 0.06))

    def test_pinned_clearance_keeps_the_numbers_and_makes_no_tool_call(self, skills_registry):
        compute_align = _script(skills_registry, "grasping-direct-ik", "compute_align_pose")
        grasp_pose = _pose(0.41, 0.12, 0.07, (0.0, 0.96, 0.0, 0.28))

        ctx = FakeContext({"robot.go_to_pose": None})
        out = compute_align.run(ctx, grasp_pose=grasp_pose, target_obb=self.OBB, clearance=0.15)
        align = out["align_pose"]

        # Align pose: same XY + rotation as the grasp, Z = OBB top + 0.15 (the
        # grasp sits inside the box, so the top face is the reference).
        assert align["position"]["x"] == pytest.approx(0.41)
        assert align["position"]["y"] == pytest.approx(0.12)
        assert align["position"]["z"] == pytest.approx(0.05 + 0.06 + 0.15)
        assert align["rotation"] == grasp_pose["rotation"]
        # A pinned clearance is pure math: no tool calls.
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

    def test_zero_clearance_asks_the_workspace(self, skills_registry):
        compute_align = _script(skills_registry, "grasping-direct-ik", "compute_align_pose")
        ctx = FakeContext({"robot.describe_workspace": {"align_clearance_m": 0.123}})
        out = compute_align.run(ctx, grasp_pose=_pose(0.41, 0.12, 0.07), target_obb=self.OBB)
        # The hand's own envelope above the fingertips, not a literal.
        assert out["align_pose"]["position"]["z"] == pytest.approx(0.11 + 0.123)
        assert [c.tool for c in ctx.calls] == ["robot.describe_workspace"]

    def test_a_grasp_above_the_obb_top_takes_the_max_branch(self, skills_registry):
        compute_align = _script(skills_registry, "grasping-direct-ik", "compute_align_pose")
        # A thin object whose grasp was clamped to a fingertip floor 3 cm
        # above its top face: the hover stands the full clearance above the
        # fingers, not above the box — max(grasp_z, top) + clearance.
        grasp_pose = _pose(0.41, 0.12, 0.14)
        out = compute_align.run(
            FakeContext({}), grasp_pose=grasp_pose, target_obb=self.OBB, clearance=0.12
        )
        assert out["align_pose"]["position"]["z"] == pytest.approx(0.14 + 0.12)

    # -- refine_top_down_grasp -------------------------------------------------

    # A bar lying along x on a table at z=0: its short horizontal axis is y.
    BAR = _obb((0.4, 0.1, 0.01), (0.10, 0.02, 0.01))
    FRAME = _quat(0.0, 0.7071, 0.7071, 0.0)

    @staticmethod
    def _gripper(stated=True):
        return {
            "finger": {"stated": stated, "reach_m": 0.045, "clearance_m": 0.004},
            "approach_axis": _vec3(0.0, 0.0, 1.0),
            "close_axis": _vec3(0.0, 1.0, 0.0),
        }

    def test_refine_closes_across_the_short_axis_and_lifts_to_the_fingertip_floor(
        self, skills_registry
    ):
        refine = _script(skills_registry, "grasping-direct-ik", "refine_top_down_grasp")
        ctx = FakeContext(
            {
                "robot.describe_gripper": self._gripper(),
                "robot.grasp_frame": {"rotation": self.FRAME},
            }
        )
        out = refine.run(ctx, grasp_pose=_pose(0.4, 0.1, 0.01), target_obb=self.BAR)
        # The jaws close along world y (heading 90 deg), approaching straight
        # down; the connector composes the frame for this hand's axes.
        frame_call = ctx.calls_to("robot.grasp_frame")[0]
        assert frame_call.kwargs["close_heading_deg"] == pytest.approx(90.0)
        assert frame_call.kwargs["approach"] == {"x": 0.0, "y": 0.0, "z": -1.0}
        refined = out["grasp_pose"]
        assert refined["rotation"] == self.FRAME
        # Fingertip floor: support 0.0 + reach 0.045 + clearance 0.004.
        assert refined["position"]["z"] == pytest.approx(0.049)
        assert refined["position"]["x"] == pytest.approx(0.4)
        assert refined["position"]["y"] == pytest.approx(0.1)
        assert [c.tool for c in ctx.calls] == ["robot.describe_gripper", "robot.grasp_frame"]

    def test_refine_follows_a_rotated_bar(self, skills_registry):
        refine = _script(skills_registry, "grasping-direct-ik", "refine_top_down_grasp")
        q = Rotation.from_euler("z", np.radians(30.0)).as_quat()  # xyzw
        bar = _obb((0.4, 0.1, 0.01), (0.10, 0.02, 0.01), (q[3], q[0], q[1], q[2]))
        ctx = FakeContext(
            {
                "robot.describe_gripper": self._gripper(),
                "robot.grasp_frame": {"rotation": self.FRAME},
            }
        )
        refine.run(ctx, grasp_pose=_pose(0.4, 0.1, 0.10), target_obb=bar)
        # Short axis y turned by 30 deg -> heading 120 deg.
        heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
        assert heading == pytest.approx(120.0)

    def test_refine_leaves_z_when_the_hand_states_no_finger_envelope(self, skills_registry):
        refine = _script(skills_registry, "grasping-direct-ik", "refine_top_down_grasp")
        ctx = FakeContext(
            {
                "robot.describe_gripper": self._gripper(stated=False),
                "robot.grasp_frame": {"rotation": self.FRAME},
            }
        )
        out = refine.run(ctx, grasp_pose=_pose(0.4, 0.1, 0.01), target_obb=self.BAR)
        assert out["grasp_pose"]["position"]["z"] == pytest.approx(0.01)

    def test_refine_then_align_reproduces_the_pinned_hover(self, skills_registry):
        refine = _script(skills_registry, "grasping-direct-ik", "refine_top_down_grasp")
        compute_align = _script(skills_registry, "grasping-direct-ik", "compute_align_pose")
        ctx = FakeContext(
            {
                "robot.describe_gripper": self._gripper(),
                "robot.grasp_frame": {"rotation": self.FRAME},
            }
        )
        refined = refine.run(ctx, grasp_pose=_pose(0.4, 0.1, 0.01), target_obb=self.BAR)
        out = compute_align.run(
            FakeContext({}), grasp_pose=refined["grasp_pose"], target_obb=self.BAR, clearance=0.12
        )
        # The refined grasp (0.049) stands above the bar's top (0.02), so the
        # hover is max(0.049 + 0.12, 0.02 + 0.12) at the refined rotation.
        assert out["align_pose"]["position"]["z"] == pytest.approx(0.049 + 0.12)
        assert out["align_pose"]["rotation"] == self.FRAME

    def test_plan_to_pose_failure_raises(self, skills_registry):
        plan_to_pose = _script(skills_registry, "grasping-direct-ik", "plan_to_pose")
        ctx = FakeContext(
            {
                "curobo.plan_to_pose": {"success": False, "trajectory": None},
            }
        )
        with pytest.raises(PlanningFailed):
            plan_to_pose.run(
                ctx,
                world_config={"meshes": []},
                observation=_observation(),
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
            FakeContext({}),
            handle_obb=handle,
            grasp_pose=grasp,
            base_obb=base,
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
        finalize = _script(skills_registry, "grasping-short-axis", "finalize_trajectory")
        traj = _trajectory(n=4)
        ctx = FakeContext({"robot.move_to_joints": None})
        out = finalize.run(ctx, trajectory=traj)
        assert out == {"done": True}
        mtj = ctx.calls_to("robot.move_to_joints")[0].kwargs
        assert mtj["joint_config"]["positions"] == list(traj["waypoints"][-1]["positions"])
        assert mtj["max_steps"] == 120  # full convergence before close

    def test_per_pose_plan_retry_then_failure(self, skills_registry):
        plan_grasp = _script(skills_registry, "grasping-short-axis", "plan_grasp")
        ctx = FakeContext(
            {
                "curobo.plan_to_grasp_poses": {
                    "success": False,
                    "trajectory": None,
                    "goalset_index": 0,
                },
            }
        )
        poses = [_pose(0.4, 0.1, 0.05), _pose(0.4, 0.1, 0.05, (0, 0, 1, 0))]
        with pytest.raises(PlanningFailed):
            plan_grasp.run(
                ctx,
                world_config={"meshes": []},
                observation=_observation(),
                grasp_poses=poses,
                target_name="target",
                retries=2,
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
    # What the connector says about a hand whose TCP sits 0.1029 m along its
    # approach axis (tool-local +z) and whose jaws close along tool-local x.
    ARM = {"arm_id": 0, "tcp_offset": _vec3(0.0, 0.0, 0.1029)}
    GRIPPER = {"approach_axis": _vec3(0.0, 0.0, 1.0), "close_axis": _vec3(1.0, 0.0, 0.0)}
    WORKSPACE = {"transport_z": 0.45, "surface_z": 0.0, "align_clearance_m": 0.15}
    WRIST = _quat(0.0, 0.96, 0.0, 0.28)

    @staticmethod
    def _grasp_frame(approach=None, close_heading_deg=None, **_):
        """That hand's top-down grasp frame at a world heading: R_z(h) · R_x(pi)."""
        yaw = np.radians(close_heading_deg or 0.0)
        q = (Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)).as_quat()
        return {"rotation": _quat(q[3], q[0], q[1], q[2])}

    def _hand_cans(self):
        return {
            "robot.describe_arm": self.ARM,
            "robot.describe_gripper": self.GRIPPER,
            "robot.describe_workspace": self.WORKSPACE,
            "robot.grasp_frame": self._grasp_frame,
        }

    # -- compute_drop_pose --------------------------------------------------------

    def test_compute_drop_pose_z_math(self, skills_registry):
        compute = _script(skills_registry, "transporting-objects", "compute_drop_pose")
        ee_at_grasp = _pose(0.4, 0.1, 0.15)  # top-down, yaw 0
        ctx = FakeContext({**self._hand_cans(), "robot.get_ee_pose": {"pose": ee_at_grasp}})
        out = compute.run(
            ctx,
            container_obb=self.CONTAINER,
            held_obb=self.HELD,
            ee_pose_at_grasp=ee_at_grasp,
        )
        # container top = 0.10; margin = max(0.03, 0.05) = 0.05;
        # desired_obj_z = 0.10 + 0.05 + 0.03 = 0.18 (< ceiling 0.199);
        # ee_to_obj = live ee z 0.15 - 0.03 = 0.12 -> ee_z_at_drop = 0.30;
        # tcp = 0.30 - 0.1029, the wrist-to-TCP distance being the arm's
        # tcp_offset projected on the hand's approach axis (robot.describe_arm
        # / robot.describe_gripper), not a literal. The at-grasp ee height is
        # measured LIVE (robot.get_ee_pose) — the wired ee_pose_at_grasp is
        # the yaw source and the fallback only.
        assert out["drop_position"]["x"] == pytest.approx(0.5)
        assert out["drop_position"]["y"] == pytest.approx(-0.2)
        assert out["drop_position"]["z"] == pytest.approx(0.30 - 0.1029)
        assert out["approach_pose"]["position"]["z"] == pytest.approx(
            out["drop_position"]["z"] + 0.20
        )
        assert ctx.call_count("robot.describe_arm") == 1
        # Yaw-only drop rotation: grasp was yaw-0 top-down, so the frame is
        # asked of the hand at heading 0 and is the canonical top-down.
        assert ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"] == pytest.approx(
            0.0
        )
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
        ctx = FakeContext({**self._hand_cans(), "robot.get_ee_pose": {"pose": ee_at_grasp}})
        out = compute.run(
            ctx,
            container_obb=self.CONTAINER,
            held_obb=self.HELD,
            ee_pose_at_grasp=ee_at_grasp,
        )
        # The heading is measured along the axis the hand says it closes on
        # and handed to robot.grasp_frame rather than composed by hand.
        heading = ctx.calls_to("robot.grasp_frame")[0].kwargs["close_heading_deg"]
        assert heading == pytest.approx(45.0)
        R_drop = _quat_to_matrix(out["drop_pose"]["rotation"])
        # Still top-down (gripper Z down)...
        assert R_drop[:, 2] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
        # ...with the grasp-time yaw preserved on the X axis.
        assert np.arctan2(R_drop[1, 0], R_drop[0, 0]) == pytest.approx(yaw)

    def test_compute_drop_pose_pinned_wrist_to_tcp_skips_the_arm(self, skills_registry):
        compute = _script(skills_registry, "transporting-objects", "compute_drop_pose")
        ee_at_grasp = _pose(0.4, 0.1, 0.15)
        ctx = FakeContext({**self._hand_cans(), "robot.get_ee_pose": {"pose": ee_at_grasp}})
        out = compute.run(
            ctx,
            container_obb=self.CONTAINER,
            held_obb=self.HELD,
            ee_pose_at_grasp=ee_at_grasp,
            wrist_to_tcp=0.0,
        )
        assert out["drop_position"]["z"] == pytest.approx(0.30)
        assert ctx.call_count("robot.describe_arm") == 0

    def test_compute_drop_pose_no_held_geometry_fallback(self, skills_registry):
        compute = _script(skills_registry, "transporting-objects", "compute_drop_pose")
        # No live ee pose scripted: the read fails and is caught.
        ctx = FakeContext(self._hand_cans())
        out = compute.run(ctx, container_obb=self.CONTAINER)
        # Legacy contract: TCP just above the rim by drop_clearance.
        assert out["drop_position"]["z"] == pytest.approx(0.10 + 0.05)
        # No grasp yaw to preserve: the hand's plain top-down frame.
        assert ctx.calls_to("robot.grasp_frame")[0].kwargs == {}

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

    # -- waypoint_move --------------------------------------------------------------

    def _waypoint_ctx(self, cartesian=None, workspace=None):
        return FakeContext(
            {
                "robot.get_observation": _observation(),
                "robot.describe_arm": self.ARM,
                "robot.describe_workspace": workspace or self.WORKSPACE,
                "robot.grasp_frame": self._grasp_frame,
                "robot.go_to_pose_cartesian": cartesian,
            }
        )

    @staticmethod
    def _lateral_heights(ctx):
        return [
            c.kwargs["pose"]["position"]["z"]
            for c in ctx.calls_to("robot.go_to_pose_cartesian")[1:]
        ]

    def test_waypoint_move_two_cartesian_legs(self, skills_registry):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")
        ctx = self._waypoint_ctx()
        out = waypoint.run(ctx, drop_x=0.5, drop_y=-0.2)
        assert out["done"] is True
        assert out["flown_z"] == pytest.approx(0.45)
        assert out["descents"] == 0
        legs = ctx.calls_to("robot.go_to_pose_cartesian")
        assert len(legs) == 2
        # Leg 1: vertical lift at the CURRENT XY to the workspace's transport_z.
        p1 = legs[0].kwargs["pose"]["position"]
        assert p1["z"] == pytest.approx(0.45)
        # Leg 2: lateral translate to the drop XY at constant height.
        p2 = legs[1].kwargs["pose"]["position"]
        assert (p2["x"], p2["y"], p2["z"]) == (0.5, -0.2, pytest.approx(0.45))
        assert p1["x"] != p2["x"] or p1["y"] != p2["y"]
        order = [c.tool for c in ctx.calls]
        assert order == [
            "robot.get_observation",
            "robot.describe_arm",
            "robot.describe_workspace",
            "robot.grasp_frame",
            "robot.go_to_pose_cartesian",
            "robot.go_to_pose_cartesian",
        ]
        # Both legs fly the hand's own top-down frame.
        R = _quat_to_matrix(legs[0].kwargs["pose"]["rotation"])
        assert R == pytest.approx(np.diag([1.0, -1.0, -1.0]), abs=1e-9)
        assert legs[1].kwargs["pose"]["rotation"] == legs[0].kwargs["pose"]["rotation"]

    def test_waypoint_move_keeps_an_upstream_rotation_and_a_pinned_height(self, skills_registry):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")
        ctx = self._waypoint_ctx()
        waypoint.run(ctx, drop_x=0.5, drop_y=-0.2, drop_rotation=self.WRIST, safe_height=0.60)
        assert ctx.call_count("robot.grasp_frame") == 0
        legs = ctx.calls_to("robot.go_to_pose_cartesian")
        assert all(leg.kwargs["pose"]["rotation"] == self.WRIST for leg in legs)
        assert legs[0].kwargs["pose"]["position"]["z"] == pytest.approx(0.60)

    def test_waypoint_move_steps_down_a_rung_when_the_far_corner_is_out_of_reach(
        self, skills_registry
    ):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")

        def _reach_limited(**kwargs):
            p = kwargs["pose"]["position"]
            if p["x"] == pytest.approx(0.5) and p["z"] > 0.43:
                raise RuntimeError("servo stopped 4 cm short")

        ctx = self._waypoint_ctx(cartesian=_reach_limited)
        out = waypoint.run(ctx, drop_x=0.5, drop_y=-0.2)
        assert out["descents"] == 1
        assert out["flown_z"] == pytest.approx(0.42)
        assert self._lateral_heights(ctx) == [pytest.approx(0.45), pytest.approx(0.42)]

    def test_waypoint_move_failure_raises(self, skills_registry):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")

        def _second_leg_fails(**kwargs):
            if kwargs["pose"]["position"]["x"] == pytest.approx(0.5):
                raise RuntimeError("linear plan failed")
            return None

        ctx = self._waypoint_ctx(cartesian=_second_leg_fails)
        with pytest.raises(PlanningFailed):
            waypoint.run(ctx, drop_x=0.5, drop_y=-0.2)
        # The cruise height and three rungs of 3 cm, all above the floor.
        assert self._lateral_heights(ctx) == [pytest.approx(z) for z in (0.45, 0.42, 0.39, 0.36)]

    def test_waypoint_move_ladder_stops_at_the_clearance_floor(self, skills_registry):
        waypoint = _script(skills_registry, "transporting-objects", "waypoint_move")

        def _never(**kwargs):
            if kwargs["pose"]["position"]["x"] == pytest.approx(0.5):
                raise RuntimeError("linear plan failed")

        # A raised work surface: floor = 0.30 + 0.12 = 0.42, so only one rung.
        workspace = {"transport_z": 0.45, "surface_z": 0.30, "align_clearance_m": 0.12}
        ctx = self._waypoint_ctx(cartesian=_never, workspace=workspace)
        with pytest.raises(PlanningFailed, match="clearance floor"):
            waypoint.run(ctx, drop_x=0.5, drop_y=-0.2)
        assert self._lateral_heights(ctx) == [pytest.approx(0.45), pytest.approx(0.42)]

    # -- descend_release / descend_release_linear --------------------------------------

    def test_descend_release_sequencing(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "descend_release")
        drop_position = _vec3(0.5, -0.2, 0.2)
        drop_rotation = _quat(0.0, 1.0, 0.0, 0.0)
        ctx = FakeContext(
            {
                "robot.go_to_pose": None,
                "robot.open_gripper": {"position": 1.0},
                "robot.describe_arm": self.ARM,
                "robot.go_home": None,
            }
        )
        out = release.run(ctx, drop_position=drop_position, drop_rotation=drop_rotation)
        assert out["drop_position"] is drop_position
        # Strict descend -> release -> retract order, with the tuned
        # open-settle so the object lands before the retract, and the
        # retract on the work arm the connector names.
        order = [c.tool for c in ctx.calls]
        assert order == [
            "robot.go_to_pose",
            "robot.open_gripper",
            "robot.describe_arm",
            "robot.go_home",
        ]
        assert ctx.calls_to("robot.go_to_pose")[0].kwargs["pose"]["position"] is drop_position
        assert ctx.calls_to("robot.go_to_pose")[0].kwargs["pose"]["rotation"] is drop_rotation
        assert ctx.calls_to("robot.open_gripper")[0].kwargs["settle_steps"] == 60
        assert ctx.calls_to("robot.go_home")[0].kwargs == {"arm_id": 0}

    def test_descend_release_asks_the_hand_for_top_down(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "descend_release")
        ctx = FakeContext(
            {
                "robot.grasp_frame": self._grasp_frame,
                "robot.go_to_pose": None,
                "robot.open_gripper": {"position": 1.0},
                "robot.describe_arm": self.ARM,
                "robot.go_home": None,
            }
        )
        release.run(ctx, drop_position=_vec3(0.5, -0.2, 0.2))
        assert ctx.calls[0].tool == "robot.grasp_frame"
        R = _quat_to_matrix(ctx.calls_to("robot.go_to_pose")[0].kwargs["pose"]["rotation"])
        assert R == pytest.approx(np.diag([1.0, -1.0, -1.0]), abs=1e-9)

    def test_descend_release_linear_routes_through_connector_cartesian(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "descend_release_linear")
        ctx = FakeContext(
            {
                "robot.go_to_pose_cartesian": None,
                "robot.open_gripper": {"position": 1.0},
                "robot.get_ee_pose": {
                    "pose": {"position": _vec3(0.5, -0.2, 0.2), "rotation": self.WRIST}
                },
                "robot.wait_steps": None,
                "robot.go_home": None,
            }
        )
        release.run(ctx, drop_position=_vec3(0.5, -0.2, 0.2), drop_rotation=self.WRIST)
        order = [c.tool for c in ctx.calls]
        # Linear descent goes through the connector's TCP-aware cartesian
        # tool (the cuRobo→plan_to_pose fallback lives inside the backend);
        # after the open, a short retreat at the live wrist rotation and a
        # settle precede the large go_home swing.
        assert order == [
            "robot.go_to_pose_cartesian",
            "robot.open_gripper",
            "robot.get_ee_pose",
            "robot.go_to_pose_cartesian",
            "robot.wait_steps",
            "robot.go_home",
        ]
        legs = ctx.calls_to("robot.go_to_pose_cartesian")
        assert legs[0].kwargs["pose"]["position"] == _vec3(0.5, -0.2, 0.2)
        assert legs[0].kwargs["pose"]["rotation"] == self.WRIST
        retreat = legs[1].kwargs["pose"]
        assert retreat["position"]["x"] == pytest.approx(0.5)
        assert retreat["position"]["y"] == pytest.approx(-0.2)
        assert retreat["position"]["z"] == pytest.approx(0.25)
        assert retreat["rotation"] == self.WRIST
        assert ctx.calls_to("robot.wait_steps")[0].kwargs == {"steps": 12}
        assert ctx.call_count("robot.execute_trajectory") == 0
        assert ctx.call_count("robot.grasp_frame") == 0

    def test_descend_release_linear_default_rotation_is_the_hands_top_down(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "descend_release_linear")
        ctx = FakeContext(
            {
                "robot.grasp_frame": self._grasp_frame,
                "robot.go_to_pose_cartesian": None,
                "robot.open_gripper": {"position": 1.0},
                "robot.get_ee_pose": {"pose": _pose(0.5, -0.2, 0.2)},
                "robot.wait_steps": None,
                "robot.go_home": None,
            }
        )
        release.run(ctx, drop_position=_vec3(0.5, -0.2, 0.2))
        assert ctx.calls[0].tool == "robot.grasp_frame"
        R = _quat_to_matrix(
            ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]["rotation"]
        )
        assert R == pytest.approx(np.diag([1.0, -1.0, -1.0]), abs=1e-9)

    # -- place_release / transport_descend_linear ----------------------------------------

    def test_place_release_hovers_ten_centimetres_above_the_placement(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "place_release")
        ctx = FakeContext(
            {
                "robot.open_gripper": {"position": 1.0},
                "robot.get_ee_pose": {
                    "pose": {"position": _vec3(0.5, -0.2, 0.04), "rotation": self.WRIST}
                },
                "robot.go_to_pose_cartesian": None,
            }
        )
        out = release.run(ctx, place_position=_vec3(0.5, -0.2, 0.04))
        assert out == {"done": True}
        assert [c.tool for c in ctx.calls] == [
            "robot.open_gripper",
            "robot.get_ee_pose",
            "robot.go_to_pose_cartesian",
        ]
        assert ctx.calls_to("robot.open_gripper")[0].kwargs["settle_steps"] == 60
        pose = ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]
        # Straight up to placement + 0.10 at the wrist rotation the transport
        # arrived with — no canonical yaw forced after the release.
        assert pose["position"] == {"x": 0.5, "y": -0.2, "z": pytest.approx(0.14)}
        assert pose["rotation"] == self.WRIST

    def test_place_release_pinned_hover_wins(self, skills_registry):
        release = _script(skills_registry, "transporting-objects", "place_release")
        ctx = FakeContext(
            {
                "robot.open_gripper": {"position": 1.0},
                "robot.get_ee_pose": {"pose": _pose(0.5, -0.2, 0.04)},
                "robot.go_to_pose_cartesian": None,
            }
        )
        release.run(ctx, place_position=_vec3(0.5, -0.2, 0.04), hover_z=0.353)
        pose = ctx.calls_to("robot.go_to_pose_cartesian")[0].kwargs["pose"]
        assert pose["position"]["z"] == pytest.approx(0.353)

    class _Arm:
        """A wrist that goes wherever the cartesian tool sends it."""

        def __init__(self, pose):
            self.pose = pose
            self.target = None

        def cartesian(self, pose):
            self.pose = pose

        def ee_pose(self):
            return {"pose": self.pose}

    def _transport_ctx(self, arm, **extra):
        return FakeContext(
            {
                "robot.get_ee_pose": lambda **_: arm.ee_pose(),
                "robot.go_to_pose_cartesian": lambda pose: arm.cartesian(pose),
                "robot.get_observation": _observation(),
                "curobo.plan_directed_linear": {"success": True, "trajectory": _trajectory()},
                "robot.execute_trajectory": None,
                **extra,
            }
        )

    @staticmethod
    def _leg_positions(ctx):
        return [
            (
                pytest.approx(c.kwargs["pose"]["position"]["x"]),
                pytest.approx(c.kwargs["pose"]["position"]["y"]),
                pytest.approx(c.kwargs["pose"]["position"]["z"]),
            )
            for c in ctx.calls_to("robot.go_to_pose_cartesian")
        ]

    def test_transport_descend_linear_derives_its_heights(self, skills_registry):
        transport = _script(skills_registry, "transporting-objects", "transport_descend_linear")
        arm = self._Arm({"position": _vec3(0.3, 0.0, 0.30), "rotation": self.WRIST})
        ctx = self._transport_ctx(arm)
        out = transport.run(ctx, container_obb=self.CONTAINER, place_offset=-0.06)
        # rim 0.10, place_offset -0.06 -> place_z 0.16 just above the rim;
        # lift_z = max(0.16 + 0.07, min(0.30, 0.16 + 0.08)) = 0.24;
        # transport_z = 0.16 + 0.08 = 0.24 — from the container, not a literal.
        assert out["place_position"] == {
            "x": pytest.approx(0.5),
            "y": pytest.approx(-0.2),
            "z": pytest.approx(0.16),
        }
        assert self._leg_positions(ctx) == [(0.3, 0.0, 0.24), (0.5, -0.2, 0.24)]
        legs = ctx.calls_to("robot.go_to_pose_cartesian")
        # The wrist keeps the rotation it arrived with.
        assert all(leg.kwargs["pose"]["rotation"] == self.WRIST for leg in legs)
        # The descend is the axis-locked linear leg over the remaining 8 cm.
        plan = ctx.calls_to("curobo.plan_directed_linear")[0].kwargs
        assert plan["distance"] == pytest.approx(0.08)
        assert plan["allowed_axes"] == ["Z"]
        assert plan["orientation_mode"] == "LOCK"
        assert ctx.calls_to("robot.execute_trajectory")[0].kwargs["max_steps_per_waypoint"] == 60
        assert ctx.call_count("robot.go_to_pose") == 0
        assert ctx.call_count("curobo.plan_to_pose") == 0

    def test_transport_descend_linear_pinned_heights_win(self, skills_registry):
        transport = _script(skills_registry, "transporting-objects", "transport_descend_linear")
        arm = self._Arm({"position": _vec3(0.3, 0.0, 0.30), "rotation": self.WRIST})
        ctx = self._transport_ctx(arm)
        out = transport.run(
            ctx, container_obb=self.CONTAINER, transport_z=0.353, lift_z=0.25, place_offset=0.06
        )
        assert out["place_position"]["z"] == pytest.approx(0.04)
        assert self._leg_positions(ctx) == [(0.3, 0.0, 0.25), (0.5, -0.2, 0.353)]
        plan = ctx.calls_to("curobo.plan_directed_linear")[0].kwargs
        assert plan["distance"] == pytest.approx(0.353 - 0.04)

    def test_transport_fallback_that_cannot_plan_raises(self, skills_registry):
        transport = _script(skills_registry, "transporting-objects", "transport_descend_linear")
        def refuse(pose):
            raise RuntimeError("joint limit")

        ctx = FakeContext(
            {
                "robot.go_to_pose_cartesian": refuse,
                "motion.plan_to_pose": {
                    "planned": False,
                    "trajectory": {"waypoints": []},
                    "reason": "no collision-free path to this pose was found",
                },
            }
        )
        with pytest.raises(RuntimeError, match="could not plan"):
            transport._cartesian(ctx, 0.3, 0.0, 0.24, self.WRIST)
        assert ctx.call_count("robot.execute_trajectory") == 0

    def test_transport_leg_that_stops_short_raises(self, skills_registry):
        transport = _script(skills_registry, "transporting-objects", "transport_descend_linear")
        # A wrist that never moves: every leg comes back >3 cm short, so the
        # best-effort lift is skipped and the reachable-interior search
        # exhausts rather than reporting a place that did not happen.
        ctx = FakeContext(
            {
                "robot.get_ee_pose": {
                    "pose": {"position": _vec3(0.3, 0.0, 0.30), "rotation": self.WRIST}
                },
                "robot.go_to_pose_cartesian": None,
            }
        )
        with pytest.raises(RuntimeError, match="no reachable point"):
            transport.run(ctx, container_obb=self.CONTAINER)
        # The preferred centre, then the four interior corners 1.5 cm inside
        # the half-extents (the centre again is deduplicated).
        tried = [(x, y) for x, y, _z in self._leg_positions(ctx)[1:]]
        assert tried == [
            (0.5, -0.2),
            (0.415, -0.285),
            (0.415, -0.115),
            (0.585, -0.285),
            (0.585, -0.115),
        ]
        assert ctx.call_count("curobo.plan_directed_linear") == 0

    def test_transport_falls_back_to_a_planned_leg(self, skills_registry):
        transport = _script(skills_registry, "transporting-objects", "transport_descend_linear")
        arm = self._Arm({"position": _vec3(0.3, 0.0, 0.30), "rotation": self.WRIST})

        def _cartesian(pose):
            # The straight-line lateral leg is refused from this configuration.
            if pose["position"]["x"] == pytest.approx(0.5) and arm.target is None:
                raise RuntimeError("joint-limit basin")
            arm.cartesian(pose)

        def _plan(pose):
            arm.target = pose
            return {"planned": True, "trajectory": _trajectory(), "reason": ""}

        def _execute(trajectory, max_steps_per_waypoint=0):
            if arm.target is not None:
                arm.pose = arm.target

        ctx = self._transport_ctx(
            arm,
            **{
                "robot.go_to_pose_cartesian": _cartesian,
                "motion.plan_to_pose": _plan,
                "robot.execute_trajectory": _execute,
            },
        )
        out = transport.run(ctx, container_obb=self.CONTAINER, place_offset=-0.06)
        # One reconfiguration to the same pose, planned by the robot (which
        # owns its base frame and arm model), then the leg counts as reached
        # and the place proceeds at the container centre.
        assert ctx.call_count("motion.plan_to_pose") == 1
        assert ctx.call_count("curobo.plan_to_pose") == 0
        plan = ctx.calls_to("motion.plan_to_pose")[0].kwargs
        assert plan["pose"]["position"]["x"] == pytest.approx(0.5)
        assert out["place_position"]["x"] == pytest.approx(0.5)
        assert out["place_position"]["z"] == pytest.approx(0.16)


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
    return FakeContext(
        {
            "sam3.tracker_init": {
                "tracker_id": "trk-1" if init_present else "",
                "initial_mask": mask if init_present else None,
                "initial_box": box if init_present else None,
                "score": 0.9 if init_present else 0.0,
                "object_present": init_present,
            },
            "sam3.tracker_update": list(update_responses),
            "sam3.tracker_close": {"closed": True},
        }
    )


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
            ctx,
            observation_stream=stream,
            target_prompt="red cup",
            max_updates=3,
            update_hz=1e6,
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
            ctx,
            observation_stream=stream,
            target_prompt="red cup",
            max_updates=1,
            update_hz=1e6,
            close_on_exit=False,
        )
        skill.run(
            ctx,
            observation_stream=stream,
            target_prompt="red cup",
            max_updates=1,
            update_hz=1e6,
            close_on_exit=False,
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
            ctx,
            observation_stream=stream,
            target_prompt="red cup",
            max_updates=5,
            update_hz=1e6,
            allow_lost_frames=2,
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
            ctx,
            observation_stream=stream,
            target_prompt="unicorn",
            max_updates=5,
            update_hz=1e6,
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
            "tracking-objects.track",
            ctx=ctx,
            observation_stream=stream,
            target_prompt="red cup",
            max_updates=1,
            update_hz=1e6,
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
            skills_registry,
            ctx,
            client,
            max_windows=2,
            replan_every=2,
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
            skills_registry,
            ctx,
            client,
            max_windows=1,
            replan_every=1,
            settle_steps=3,
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
            skills_registry,
            ctx,
            client,
            max_windows=10,
            replan_every=1,
            gripper_cycle_termination=True,
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
            skills_registry,
            ctx,
            client,
            max_windows=3,
            replan_every=1,
            gripper_cycle_termination=True,
        )
        assert out["status"] == "max_windows"

    def test_vlm_termination(self, skills_registry):
        chunk = np.zeros((1, 7), dtype=np.float64)
        client = _StubClient([chunk] * 3)
        ctx = _policy_ctx({"vlm.query_yes_no": {"answer": True, "text": "yes"}})
        out, _, _ = self._run(
            skills_registry,
            ctx,
            client,
            max_windows=10,
            replan_every=1,
            termination_prompt="is the object in the basket?",
            term_period=1,
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
            "pi05-libero.run",
            ctx=ctx,
            observation_stream=_StubStream(_observation()),
            prompt="pick",
            settle_steps=0,
            max_windows=1,
            replan_every=1,
        )
        assert out["status"] == "max_windows"
        assert out["num_steps"] == 1

    def test_molmoact_uses_its_own_preset(self, skills_registry):
        chunk = np.zeros((1, 7), dtype=np.float64)
        client = _StubClient([chunk])
        ctx = _policy_ctx()
        _, executor, _ = self._run(
            skills_registry,
            ctx,
            client,
            bundle="molmoact-libero",
            max_windows=1,
            replan_every=1,
        )
        # Each policy skill resolves the client cache by its own preset
        # (bundle name), proving the model identity is per-skill.
        assert executor.requested == ["molmoact-libero"]
