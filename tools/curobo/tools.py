"""curobo tool bundle — collision-aware motion planning, IK, and validation.

Every RPC of the original ``curobo.v1.CuRobo`` service as a typed in-process
tool (snake_case of the RPC name). The planning code lives in
``_curobo_impl.py`` (the dev tree's ``third_party/curobo_api.py`` wrapper layer,
ported verbatim and trimmed); this module is the typed boundary:
:mod:`gap.types` TypedDicts in/out, with the servicer's proto-default
substitution turned into plain Python defaults.

cuRobo is a CUDA-JIT package — ``_curobo_impl`` is imported lazily on first
tool call, so importing this module never pulls torch/curobo. A missing
install raises a ToolError pointing at ``pip install -e "open-robot-skills[curobo]"``
(see SKILL.md for the CUDA_HOME / --no-build-isolation recipe).

GPU access is serialised with a module-level lock (cuRobo is not
thread-safe), and CUDA/graph-capture errors invalidate the cached planners
before surfacing as PlanningFailed — both behaviors carried over from the
gRPC server.
"""

from __future__ import annotations

import logging
import threading
import traceback
from types import SimpleNamespace
from typing import Any, TypedDict

import numpy as np
from gap_core.errors import PlanningFailed, ToolError
from gap_core.tools import tool
from gap_core.types import (
    JointState,
    Quaternion,
    Se3Pose,
    Trajectory,
    Vec3,
    WorldConfig,
)

logger = logging.getLogger(__name__)

# Serialise GPU access -- CuRobo is not thread-safe.
_LOCK = threading.Lock()

_INSTALL_HINT = (
    "cuRobo is not installed (or failed to import). Install the bundle "
    "deps with:  pip install -e \"open-robot-skills[curobo]\" --no-build-isolation  "
    "with CUDA_HOME pointing at your CUDA toolkit — see tools/curobo/SKILL.md."
)


def _impl():
    """Import the cuRobo implementation lazily, with an actionable error."""
    try:
        from gap_skills.tools.curobo import _curobo_impl
        return _curobo_impl
    except ImportError as e:
        raise ToolError("curobo", f"{_INSTALL_HINT} (import error: {e})") from e


def _cuda_cleanup(impl: Any, cache_attr: str, err: Exception) -> None:
    """Invalidate cached planners after CUDA errors (poisoned-state guard)."""
    msg = str(err).lower()
    if "cuda" in msg or "graph capture" in msg:
        try:
            import torch

            getattr(impl, cache_attr).invalidate()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("CUDA state cleaned up after error")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# gap.types <-> impl converters (the former proto decode layer, numpy-fied)
# ---------------------------------------------------------------------------


def _joints(js: JointState) -> np.ndarray:
    """Joint positions -> (dof,) float64, tolerant of the shapes generated
    graphs actually wire in: a gap ``JointState`` dict, a full arm
    observation (``{"joint_state": {...}}``), or a bare positions
    list/array. Guessed-wrong wiring used to surface as an opaque
    ``list indices must be integers`` crash mid-place."""
    if isinstance(js, dict):
        if "positions" in js:
            return np.asarray(js["positions"], dtype=np.float64).flatten()
        if "joint_state" in js:
            return _joints(js["joint_state"])
        raise TypeError(
            "joint state dict has neither 'positions' nor 'joint_state' "
            f"(keys: {sorted(js)})"
        )
    return np.asarray(js, dtype=np.float64).flatten()


def _pose_tuple(pose: Se3Pose) -> tuple[np.ndarray, np.ndarray]:
    """gap Se3Pose -> (position (3,), quaternion_wxyz (4,))."""
    p, r = pose["position"], pose["rotation"]
    pos = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    quat_wxyz = np.array([r["w"], r["x"], r["y"], r["z"]], dtype=np.float64)
    return pos, quat_wxyz


def _vec3(v: Vec3) -> np.ndarray:
    return np.array([v["x"], v["y"], v["z"]], dtype=np.float64)


def _world_ns(wc: WorldConfig | None) -> Any:
    """gap WorldConfig -> lightweight ``.mesh``-bearing object.

    cuRobo v0.8 ("cuRoboV2") deleted ``curobo.geom.types`` (the v0.7
    ``Mesh``/``WorldConfig``). The v0.8 impl consumes the world purely via
    ``getattr(world_config, "mesh", ...)`` with each entry exposing
    ``name``/``pose``/``vertices``/``faces`` (see ``_world_to_v2_scene_cfg``
    / ``_v2_scene_cfg_excluding`` in ``_curobo_impl.py``), so a plain
    SimpleNamespace is the correct, version-agnostic carrier — no curobo
    geom import needed.
    """
    meshes = []
    for m in (wc or {}).get("meshes", []):
        verts = np.asarray(m["vertices"], dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(m["faces"], dtype=np.int32).reshape(-1, 3)
        # Default identity pose: [x, y, z, qw, qx, qy, qz]
        pose = [0, 0, 0, 1, 0, 0, 0]
        if m.get("pose") is not None:
            p = m["pose"]
            pose = [
                p["position"]["x"], p["position"]["y"], p["position"]["z"],
                p["rotation"]["w"], p["rotation"]["x"],
                p["rotation"]["y"], p["rotation"]["z"],
            ]
        meshes.append(SimpleNamespace(
            name=m["name"],
            pose=pose,
            vertices=verts.tolist(),
            faces=faces.tolist(),
        ))
    return SimpleNamespace(mesh=meshes)


def _traj_out(trajectory: np.ndarray | None) -> Trajectory | None:
    """(N, dof) numpy trajectory -> gap Trajectory."""
    if trajectory is None:
        return None
    return {
        "waypoints": [
            {"positions": np.asarray(row, dtype=np.float64)} for row in trajectory
        ]
    }


def _traj_in(traj: Trajectory) -> np.ndarray:
    """gap Trajectory -> (N, dof) float64 array."""
    rows = [np.asarray(wp["positions"], dtype=np.float64) for wp in traj["waypoints"]]
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class PlanGraspResult(TypedDict):
    success: bool
    trajectory: Trajectory | None       # joint trajectory (N waypoints x dof)
    goalset_index: int                  # which grasp pose was reached


class PlanResult(TypedDict):
    success: bool
    trajectory: Trajectory | None


class PlanLinearResult(TypedDict):
    success: bool
    trajectory: Trajectory | None
    failure_reason: str


class PlanGraspMotionResult(TypedDict):
    success: bool
    approach_trajectory: Trajectory | None  # current state -> pre-grasp
    grasp_trajectory: Trajectory | None     # pre-grasp -> grasp (constrained linear)
    lift_trajectory: Trajectory | None      # grasp -> post-grasp (constrained linear)
    failure_reason: str


class SolveIkResult(TypedDict):
    success: bool
    joint_config: JointState | None     # in URDF kinematic order


class BatchFeasibilityResult(TypedDict):
    feasible: list[bool]                       # aligned with input grasp_poses
    grasp_ik_ok: list[bool]
    approach_ik_ok: list[bool]
    corridor_collision_fraction: list[float]   # 0.0=clear, 1.0=fully blocked


class ValidateResult(TypedDict):
    success: bool
    failure_reason: str                 # empty if success
    first_collision_waypoint: int       # -1 if none / success
    collision_status_detail: str        # e.g. MotionGenStatus name


# ---------------------------------------------------------------------------
# Planning tools
# ---------------------------------------------------------------------------


@tool(
    name="curobo.plan_to_grasp_poses",
    summary="Plan a collision-free joint trajectory to reach one of several candidate grasp poses (goalset).",
    tags=("planning",),
)
def plan_to_grasp_poses(
    world_config: WorldConfig,
    start_joint_position: JointState,
    grasp_poses: list[Se3Pose],
    robot_file: str = "franka.yml",
    max_attempts: int = 8,
    use_cuda_graph: bool = False,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.05,
    position_threshold_z: float = 0.01,
    grasp_pose_is_fingertip: bool = True,
    grasp_z_clearance: float = 0.005,
    num_ik_seeds: int = 128,
    relax_orientation: bool = False,
    use_grasp_approach: bool = False,
    grasp_approach_offset: float = 0.03,
    grasp_approach_linear_axis: int = 2,
    grasp_approach_tstep_fraction: float = 0.7,
    use_world_collision: bool = True,
    robot_collision_sphere_buffer: float = -0.01,
    collision_activation_distance: float = 0.001,
    ignore_obstacle_names: list[str] | None = None,
    debug_out_dir: str | None = None,
) -> PlanGraspResult:
    """Grasp poses are in the robot-base frame; with
    ``grasp_pose_is_fingertip=True`` (default) positions are fingertip-centre
    and converted to the panda_hand frame solver-side. Pass the target's mesh
    name in ``ignore_obstacle_names`` so approaching it isn't a collision.
    ``goalset_index`` says which input pose was reached."""
    impl = _impl()
    try:
        with _LOCK:
            success, trajectory, goalset_index = impl.plan_to_grasp_poses(
                _world_ns(world_config),
                _joints(start_joint_position),
                [_pose_tuple(gp) for gp in grasp_poses],
                robot_file=robot_file,
                max_attempts=max_attempts,
                use_cuda_graph=use_cuda_graph,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                position_threshold_z=position_threshold_z,
                grasp_pose_is_fingertip=grasp_pose_is_fingertip,
                grasp_z_clearance=grasp_z_clearance,
                num_ik_seeds=num_ik_seeds,
                relax_orientation=relax_orientation,
                use_grasp_approach=use_grasp_approach,
                grasp_approach_offset=grasp_approach_offset,
                grasp_approach_linear_axis=grasp_approach_linear_axis,
                grasp_approach_tstep_fraction=grasp_approach_tstep_fraction,
                use_world_collision=use_world_collision,
                robot_collision_sphere_buffer=robot_collision_sphere_buffer,
                collision_activation_distance=collision_activation_distance,
                ignore_obstacle_names=ignore_obstacle_names,
                debug_out_dir=debug_out_dir,
            )
    except Exception as e:
        logger.error("plan_to_grasp_poses failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_motion_gen_cache", e)
        raise PlanningFailed(f"plan_to_grasp_poses failed: {e}") from e

    return {
        "success": bool(success),
        "trajectory": _traj_out(trajectory) if success else None,
        "goalset_index": int(goalset_index) if goalset_index is not None else 0,
    }


@tool(
    name="curobo.plan_with_grasped_object",
    summary="Plan a collision-free trajectory while holding a grasped object (attached to the gripper).",
    tags=("planning",),
)
def plan_with_grasped_object(
    world_config: WorldConfig,
    start_joint_position: JointState,
    target_pose: Se3Pose,
    object_name: str,
    robot_file: str = "franka.yml",
    max_attempts: int = 8,
    use_cuda_graph: bool = False,
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    position_threshold_z: float = 0.05,
    num_ik_seeds: int = 128,
    use_world_collision: bool = True,
    robot_collision_sphere_buffer: float = -0.01,
    collision_activation_distance: float = 0.01,
    surface_sphere_radius: float = 0.001,
    link_name: str = "attached_object",
    remove_obstacles_from_world: bool = False,
    debug_out_dir: str | None = None,
) -> PlanResult:
    """``object_name`` must name a mesh in ``world_config``; it is attached
    to the robot at the start configuration and collision-checked against
    the remaining scene during transport."""
    impl = _impl()
    try:
        with _LOCK:
            success, trajectory = impl.plan_with_grasped_object(
                _world_ns(world_config),
                _joints(start_joint_position),
                _pose_tuple(target_pose),
                object_name,
                robot_file=robot_file,
                max_attempts=max_attempts,
                use_cuda_graph=use_cuda_graph,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                position_threshold_z=position_threshold_z,
                num_ik_seeds=num_ik_seeds,
                use_world_collision=use_world_collision,
                robot_collision_sphere_buffer=robot_collision_sphere_buffer,
                collision_activation_distance=collision_activation_distance,
                surface_sphere_radius=surface_sphere_radius,
                link_name=link_name,
                remove_obstacles_from_world=remove_obstacles_from_world,
                debug_out_dir=debug_out_dir,
            )
    except Exception as e:
        logger.error("plan_with_grasped_object failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_motion_gen_cache", e)
        raise PlanningFailed(f"plan_with_grasped_object failed: {e}") from e

    return {
        "success": bool(success),
        "trajectory": _traj_out(trajectory) if success else None,
    }


@tool(
    name="curobo.plan_linear",
    summary="Plan a straight Cartesian-space trajectory between two end-effector poses.",
    tags=("planning",),
)
def plan_linear(
    start_pose: Se3Pose,
    end_pose: Se3Pose,
    start_joint_position: JointState,
    robot_file: str = "franka.yml",
    hold_vec_weight: list[float] | None = None,
) -> PlanLinearResult:
    """``hold_vec_weight`` is the PoseCostMetric 6-vector
    [rx, ry, rz, x, y, z]: 1.0 = hold (constrain), 0.0 = free; e.g.
    [1,1,1,1,0,1] holds orientation + X,Z so motion is along Y only.
    Empty/None = unconstrained planning."""
    impl = _impl()
    try:
        with _LOCK:
            success, trajectory, reason = impl.plan_linear(
                start_pose=_pose_tuple(start_pose),
                end_pose=_pose_tuple(end_pose),
                start_joint_position=_joints(start_joint_position),
                robot_file=robot_file,
                hold_vec_weight=list(hold_vec_weight) if hold_vec_weight else None,
            )
    except Exception as e:
        logger.error("plan_linear failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_motion_gen_cache", e)
        raise PlanningFailed(f"plan_linear failed: {e}") from e

    return {
        "success": bool(success),
        "trajectory": _traj_out(trajectory) if success else None,
        "failure_reason": reason or "",
    }


@tool(
    name="curobo.plan_directed_linear",
    summary="Constrained linear end-effector motion (axis/orientation holds) via MotionGen + PoseCostMetric.",
    tags=("planning",),
)
def plan_directed_linear(
    start_joint_position: JointState,
    start_pose: Se3Pose | None = None,
    target_pose: Se3Pose | None = None,
    allowed_axes: list[str] | None = None,
    explicit_direction: Vec3 | None = None,
    distance: float | None = None,
    endpoint_mode: str = "PROJECT_TO_TARGET",
    orientation_mode: str = "LOCK",
    orientation_target: Quaternion | None = None,
    robot_file: str = "franka.yml",
) -> PlanLinearResult:
    """FK(start_joint_position) is computed internally; ``start_pose`` is a
    hint only. ``allowed_axes`` ⊆ ["X","Y","Z"] are free to move (others held
    at FK values). ``endpoint_mode``: PROJECT_TO_TARGET | DISTANCE |
    ORIENT_IN_PLACE; ``orientation_mode``: LOCK | TARGET_AT_END | SLERP."""
    impl = _impl()
    try:
        with _LOCK:
            success, trajectory, reason = impl.plan_directed_linear(
                start_config=_joints(start_joint_position),
                start_pose=_pose_tuple(start_pose) if start_pose is not None else None,
                target_pose=_pose_tuple(target_pose) if target_pose is not None else None,
                allowed_axes=list(allowed_axes) if allowed_axes else None,
                explicit_direction=(
                    _vec3(explicit_direction).astype(np.float32)
                    if explicit_direction is not None
                    else None
                ),
                distance=distance,
                endpoint_mode=endpoint_mode or "PROJECT_TO_TARGET",
                orientation_mode=orientation_mode or "LOCK",
                orientation_target=(
                    np.array(
                        [
                            orientation_target["w"],
                            orientation_target["x"],
                            orientation_target["y"],
                            orientation_target["z"],
                        ],
                        dtype=np.float32,
                    )
                    if orientation_target is not None
                    else None
                ),
                robot_file=robot_file,
            )
    except Exception as e:
        logger.error("plan_directed_linear failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_motion_gen_cache", e)
        raise PlanningFailed(f"plan_directed_linear failed: {e}") from e

    return {
        "success": bool(success),
        "trajectory": _traj_out(trajectory) if success else None,
        "failure_reason": reason or "",
    }


@tool(
    name="curobo.plan_grasp_motion",
    summary="Plan a full grasp sequence (free-space approach, constrained grasp, constrained lift) via plan_grasp.",
    tags=("planning",),
)
def plan_grasp_motion(
    start_joint_position: JointState,
    grasp_pose: Se3Pose,
    approach_axis: str = "y",
    approach_distance: float = 0.12,
    approach_in_tool_frame: bool = False,
    lift_axis: str = "y",
    lift_distance: float = 0.20,
    lift_in_tool_frame: bool = False,
    robot_file: str = "franka.yml",
) -> PlanGraspMotionResult:
    """Three trajectories come back so the caller can interleave gripper
    commands: execute approach, then grasp, close gripper, then lift.
    Requires curobo v0.8 — returns ``success=False`` on v0.7."""
    impl = _impl()
    try:
        with _LOCK:
            success, approach_traj, grasp_traj, lift_traj, reason = (
                impl.plan_grasp_motion(
                    start_config=_joints(start_joint_position),
                    grasp_pose=_pose_tuple(grasp_pose),
                    approach_axis=approach_axis or "y",
                    approach_distance=approach_distance,
                    approach_in_tool_frame=approach_in_tool_frame,
                    lift_axis=lift_axis or "y",
                    lift_distance=lift_distance,
                    lift_in_tool_frame=lift_in_tool_frame,
                    plan_approach=True,
                    plan_lift=True,
                    robot_file=robot_file,
                )
            )
    except Exception as e:
        logger.error("plan_grasp_motion failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_motion_gen_cache", e)
        raise PlanningFailed(f"plan_grasp_motion failed: {e}") from e

    return {
        "success": bool(success),
        "approach_trajectory": _traj_out(approach_traj) if success else None,
        "grasp_trajectory": _traj_out(grasp_traj) if success else None,
        "lift_trajectory": _traj_out(lift_traj) if success else None,
        "failure_reason": reason or "",
    }


@tool(
    name="curobo.plan_to_pose",
    summary="Plan a collision-aware trajectory to a single end-effector target pose (optional world).",
    tags=("planning",),
)
def plan_to_pose(
    target_pose: Se3Pose,
    start_joint_position: JointState,
    robot_file: str = "franka.yml",
    tcp_offset: Vec3 | None = None,
    world_config: WorldConfig | None = None,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    max_attempts: int = 4,
) -> PlanResult:
    """``target_pose`` is the TCP (tool-tip) pose in world frame;
    ``tcp_offset`` is the EE-link → TCP vector in the EE link's local frame
    and is applied solver-side. With no ``world_config`` this is a free-space
    plan that still respects self-collision."""
    impl = _impl()
    try:
        target_pos, target_quat = _pose_tuple(target_pose)
        world_ns = (
            _world_ns(world_config)
            if world_config is not None and len(world_config.get("meshes", [])) > 0
            else None
        )
        with _LOCK:
            success, traj = impl.plan_to_pose(
                target_pos,
                target_quat,
                _joints(start_joint_position),
                robot_file=robot_file,
                tcp_offset=_vec3(tcp_offset) if tcp_offset is not None else None,
                world_config=world_ns,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
                max_attempts=max_attempts,
            )
    except Exception as e:
        logger.error("plan_to_pose failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_motion_gen_cache", e)
        raise PlanningFailed(f"plan_to_pose failed: {e}") from e

    return {
        "success": bool(success),
        "trajectory": _traj_out(traj) if success else None,
    }


# ---------------------------------------------------------------------------
# IK / feasibility
# ---------------------------------------------------------------------------


@tool(
    name="curobo.solve_ik",
    summary="Solve geometric inverse kinematics for a single TCP pose (no world collision checking).",
    tags=("planning",),
)
def solve_ik(
    target_pose: Se3Pose,
    seed_config: JointState | None = None,
    robot_file: str = "franka.yml",
    tcp_offset: Vec3 | None = None,
    num_seeds: int = 32,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
) -> SolveIkResult:
    """Pure geometric IK (self-collision aware, NOT world-collision aware) —
    use ``curobo.plan_to_pose`` for a collision-aware reach. With a
    ``seed_config`` the solution nearest the seed is returned. NOTE: the
    underlying v0.7 IKSolver API was removed in curobo v0.8 — on a v0.8-only
    install this raises PlanningFailed; plan with plan_to_grasp_poses /
    plan_grasp_motion instead."""
    impl = _impl()
    try:
        pos, quat_wxyz = _pose_tuple(target_pose)
        with _LOCK:
            success, q = impl.solve_ik(
                pos,
                quat_wxyz,
                robot_file=robot_file,
                seed_config=_joints(seed_config) if seed_config is not None else None,
                tcp_offset=_vec3(tcp_offset) if tcp_offset is not None else None,
                num_seeds=num_seeds,
                position_threshold=position_threshold,
                rotation_threshold=rotation_threshold,
            )
    except Exception as e:
        logger.error("solve_ik failed: %s\n%s", e, traceback.format_exc())
        raise PlanningFailed(f"solve_ik failed: {e}") from e

    return {
        "success": bool(success),
        "joint_config": (
            {"positions": np.asarray(q, dtype=np.float64)}
            if success and q is not None
            else None
        ),
    }


@tool(
    name="curobo.batch_grasp_feasibility",
    summary="Per-pose collision-aware feasibility (grasp IK + approach IK + corridor sweep) for a batch of grasps.",
    tags=("planning",),
)
def batch_grasp_feasibility(
    world_config: WorldConfig,
    start_state: JointState,
    grasp_poses: list[Se3Pose],
    grasp_pose_is_fingertip: bool = True,
    approach_offset_m: float = 0.10,
    num_corridor_samples: int = 5,
    num_ik_seeds: int = 32,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    collision_activation_distance: float = 0.01,
    robot_collision_sphere_buffer: float | None = None,
    robot_file: str = "franka.yml",
    ignore_obstacle_names: list[str] | None = None,
) -> BatchFeasibilityResult:
    """All result vectors align with the input ``grasp_poses`` order so
    callers can filter without losing rank. Put the target object in
    ``ignore_obstacle_names`` — the gripper is meant to close on it. NOTE:
    backed by the v0.7 IKSolver API, removed in curobo v0.8 — raises
    PlanningFailed on a v0.8-only install."""
    impl = _impl()
    if len(grasp_poses) == 0:
        raise ToolError("curobo.batch_grasp_feasibility", "empty grasp_poses")
    if len(world_config.get("meshes", [])) == 0:
        raise ToolError("curobo.batch_grasp_feasibility", "world_config has no meshes")

    try:
        with _LOCK:
            feasible, grasp_ok, approach_ok, corridor_frac = (
                impl.batch_grasp_feasibility(
                    _world_ns(world_config),
                    _joints(start_state),
                    [_pose_tuple(gp) for gp in grasp_poses],
                    grasp_pose_is_fingertip=grasp_pose_is_fingertip,
                    approach_offset_m=approach_offset_m,
                    num_corridor_samples=num_corridor_samples,
                    robot_file=robot_file,
                    num_ik_seeds=num_ik_seeds,
                    position_threshold=position_threshold,
                    rotation_threshold=rotation_threshold,
                    collision_activation_distance=collision_activation_distance,
                    robot_collision_sphere_buffer=robot_collision_sphere_buffer,
                    ignore_obstacle_names=ignore_obstacle_names,
                )
            )
    except Exception as e:
        logger.error("batch_grasp_feasibility failed: %s\n%s", e, traceback.format_exc())
        _cuda_cleanup(impl, "_collision_aware_ik_cache", e)
        raise PlanningFailed(f"batch_grasp_feasibility failed: {e}") from e

    return {
        "feasible": [bool(x) for x in feasible],
        "grasp_ik_ok": [bool(x) for x in grasp_ok],
        "approach_ik_ok": [bool(x) for x in approach_ok],
        "corridor_collision_fraction": [float(x) for x in corridor_frac],
    }


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------


@tool(
    name="curobo.validate_joint_trajectory_robot",
    summary="Collision-validate joint-space waypoints (robot vs world + self-collision).",
    tags=("planning",),
)
def validate_joint_trajectory_robot(
    world_config: WorldConfig,
    trajectory: Trajectory,
    robot_file: str = "franka.yml",
    use_cuda_graph: bool = False,
    robot_collision_sphere_buffer: float = -0.01,
    collision_activation_distance: float = 0.01,
    ignore_obstacle_names: list[str] | None = None,
) -> ValidateResult:
    """Checks every waypoint with MotionGen.check_start_state against the
    same WorldConfig used for perception-built meshes. ``use_cuda_graph``
    must stay False (check_start_state requirement)."""
    impl = _impl()
    if not trajectory.get("waypoints"):
        raise ToolError(
            "curobo.validate_joint_trajectory_robot", "trajectory has no waypoints"
        )

    try:
        with _LOCK:
            ok, reason, idx, meta = impl.validate_joint_trajectory_robot_world(
                _world_ns(world_config),
                _traj_in(trajectory),
                robot_file=robot_file,
                use_cuda_graph=use_cuda_graph,
                robot_collision_sphere_buffer=robot_collision_sphere_buffer,
                collision_activation_distance=collision_activation_distance,
                ignore_obstacle_names=(
                    list(ignore_obstacle_names) if ignore_obstacle_names else None
                ),
            )
    except Exception as e:
        logger.error(
            "validate_joint_trajectory_robot failed: %s\n%s", e, traceback.format_exc()
        )
        raise PlanningFailed(f"validate_joint_trajectory_robot failed: {e}") from e

    detail = meta.get("motion_gen_status", "") if meta else ""
    return {
        "success": bool(ok),
        "failure_reason": "" if ok else (reason or "collision"),
        "first_collision_waypoint": -1 if ok or idx is None else int(idx),
        "collision_status_detail": str(detail),
    }


@tool(
    name="curobo.validate_joint_trajectory_grasped",
    summary="Collision-validate joint-space waypoints with a grasped object attached at waypoint 0.",
    tags=("planning",),
)
def validate_joint_trajectory_grasped(
    world_config: WorldConfig,
    trajectory: Trajectory,
    object_name: str,
    robot_file: str = "franka.yml",
    use_cuda_graph: bool = False,
    robot_collision_sphere_buffer: float = -0.01,
    collision_activation_distance: float = 0.01,
    surface_sphere_radius: float = 0.001,
    link_name: str = "attached_object",
    remove_obstacles_from_world: bool = False,
) -> ValidateResult:
    """Attaches ``object_name`` (a mesh in ``world_config``) at the FIRST
    waypoint configuration, validates every row, and always resets the
    planner cache so attachment state cannot leak into later calls."""
    impl = _impl()
    if not object_name:
        raise ToolError(
            "curobo.validate_joint_trajectory_grasped", "object_name is required"
        )
    if not trajectory.get("waypoints"):
        raise ToolError(
            "curobo.validate_joint_trajectory_grasped", "trajectory has no waypoints"
        )

    try:
        with _LOCK:
            ok, reason, idx, meta = impl.validate_joint_trajectory_grasped_object(
                _world_ns(world_config),
                _traj_in(trajectory),
                object_name,
                robot_file=robot_file,
                use_cuda_graph=use_cuda_graph,
                robot_collision_sphere_buffer=robot_collision_sphere_buffer,
                collision_activation_distance=collision_activation_distance,
                surface_sphere_radius=surface_sphere_radius,
                link_name=link_name,
                remove_obstacles_from_world=remove_obstacles_from_world,
            )
    except Exception as e:
        logger.error(
            "validate_joint_trajectory_grasped failed: %s\n%s",
            e, traceback.format_exc(),
        )
        raise PlanningFailed(f"validate_joint_trajectory_grasped failed: {e}") from e

    detail = meta.get("motion_gen_status", "") if meta else ""
    return {
        "success": bool(ok),
        "failure_reason": "" if ok else (reason or "collision"),
        "first_collision_waypoint": -1 if ok or idx is None else int(idx),
        "collision_status_detail": str(detail),
    }
