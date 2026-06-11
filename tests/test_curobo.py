"""curobo bundle: signature/schema + converter units + GPU smoke.

cuRobo itself never imports during collection or the CPU tests — the
converters and validation guards are exercised through the tools module
without touching ``_curobo_impl``.
"""

from __future__ import annotations

import numpy as np
import pytest

EXPECTED_TOOLS = {
    "curobo.plan_to_grasp_poses",
    "curobo.plan_with_grasped_object",
    "curobo.plan_linear",
    "curobo.plan_directed_linear",
    "curobo.plan_grasp_motion",
    "curobo.plan_to_pose",
    "curobo.solve_ik",
    "curobo.batch_grasp_feasibility",
    "curobo.validate_joint_trajectory_robot",
    "curobo.validate_joint_trajectory_grasped",
}

_FRANKA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _pose(x, y, z, w=0.0, qx=1.0, qy=0.0, qz=0.0):
    return {
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"w": w, "x": qx, "y": qy, "z": qz},
    }


def test_all_tools_registered(tool_registry):
    for name in EXPECTED_TOOLS:
        assert name in tool_registry
        assert "planning" in tool_registry.get(name).tags


def test_plan_to_grasp_poses_schema(tool_registry):
    schema = tool_registry.get("curobo.plan_to_grasp_poses").schema
    required = {"world_config", "start_joint_position", "grasp_poses"}
    assert required <= set(schema.inputs)
    for name in required:
        assert schema.inputs[name].required
    # Proto/servicer defaults carried into the signature.
    assert schema.inputs["robot_file"].default == "franka.yml"
    assert schema.inputs["max_attempts"].default == 8
    assert schema.inputs["num_ik_seeds"].default == 128
    assert schema.inputs["robot_collision_sphere_buffer"].default == pytest.approx(-0.01)
    assert schema.inputs["collision_activation_distance"].default == pytest.approx(0.001)
    assert schema.inputs["grasp_pose_is_fingertip"].default is True
    assert schema.inputs["use_world_collision"].default is True
    assert set(schema.outputs) == {"success", "trajectory", "goalset_index"}


def test_validate_schema(tool_registry):
    schema = tool_registry.get("curobo.validate_joint_trajectory_grasped").schema
    assert {"world_config", "trajectory", "object_name"} <= set(schema.inputs)
    assert schema.inputs["link_name"].default == "attached_object"
    assert schema.inputs["surface_sphere_radius"].default == pytest.approx(0.001)
    assert set(schema.outputs) == {
        "success",
        "failure_reason",
        "first_collision_waypoint",
        "collision_status_detail",
    }


def test_trajectory_converters_roundtrip(skills_registry):
    mod = skills_registry.get("curobo").tools_module
    arr = np.arange(21, dtype=np.float64).reshape(3, 7)
    traj = mod._traj_out(arr)
    assert len(traj["waypoints"]) == 3
    back = mod._traj_in(traj)
    np.testing.assert_allclose(back, arr)
    assert mod._traj_out(None) is None
    assert mod._traj_in({"waypoints": []}).shape == (0, 0)


def test_world_converter_builds_mesh_namespace(skills_registry):
    mod = skills_registry.get("curobo").tools_module
    wc = {
        "meshes": [
            {
                "name": "scene",
                "vertices": np.zeros((3, 3), dtype=np.float32),
                "faces": np.array([[0, 1, 2]], dtype=np.int32),
                "pose": _pose(1.0, 2.0, 3.0, w=1.0, qx=0.0),
            }
        ]
    }
    ns = mod._world_ns(wc)
    assert len(ns.mesh) == 1
    mesh = ns.mesh[0]
    assert mesh.name == "scene"
    assert mesh.pose == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    assert len(mesh.vertices) == 3 and len(mesh.faces) == 1
    # Identity pose default when the mesh has no pose.
    ns2 = mod._world_ns({"meshes": [{"name": "m", "vertices": [], "faces": [], "pose": None}]})
    assert ns2.mesh[0].pose == [0, 0, 0, 1, 0, 0, 0]


def test_empty_trajectory_rejected_without_curobo(tool_registry):
    """Argument guards fire before the lazy cuRobo import."""
    from gap.errors import ToolError

    with pytest.raises(ToolError):
        tool_registry.invoke(
            "curobo.validate_joint_trajectory_robot",
            world_config={"meshes": []},
            trajectory={"waypoints": []},
        )
    with pytest.raises(ToolError):
        tool_registry.invoke(
            "curobo.validate_joint_trajectory_grasped",
            world_config={"meshes": []},
            trajectory={"waypoints": [{"positions": _FRANKA_HOME}]},
            object_name="",
        )
    with pytest.raises(ToolError):
        tool_registry.invoke(
            "curobo.batch_grasp_feasibility",
            world_config={"meshes": []},
            start_state={"positions": _FRANKA_HOME},
            grasp_poses=[_pose(0.4, 0.0, 0.2)],
        )


@pytest.mark.gpu
def test_plan_to_pose_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    out = tool_registry.invoke(
        "curobo.plan_to_pose",
        target_pose=_pose(0.45, 0.0, 0.35),  # reachable, gripper down
        start_joint_position={"positions": _FRANKA_HOME},
    )
    assert isinstance(out["success"], bool)
    if out["success"]:
        traj = out["trajectory"]
        assert traj is not None and len(traj["waypoints"]) > 1
        assert len(traj["waypoints"][0]["positions"]) >= 7


@pytest.mark.gpu
def test_plan_to_grasp_poses_gpu_smoke(tool_registry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("curobo")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    # Tiny floor slab as the world; grasp straight down above it.
    verts = np.array(
        [[-0.5, -0.5, -0.02], [0.5, -0.5, -0.02], [0.5, 0.5, -0.02], [-0.5, 0.5, -0.02]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    world = {"meshes": [{"name": "floor", "vertices": verts, "faces": faces, "pose": None}]}

    out = tool_registry.invoke(
        "curobo.plan_to_grasp_poses",
        world_config=world,
        start_joint_position={"positions": _FRANKA_HOME},
        grasp_poses=[_pose(0.45, 0.0, 0.25), _pose(0.45, 0.05, 0.25)],
    )
    assert isinstance(out["success"], bool)
    if out["success"]:
        assert out["goalset_index"] in (0, 1)
        assert out["trajectory"] is not None
