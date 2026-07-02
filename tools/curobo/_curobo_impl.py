"""CuRobo motion-planning implementation, ported from the dev tree's third_party/curobo_api.py.

Verbatim copy of the planning/IK/validation code paths, trimmed of the
parts the gap tool layer does not use:

- the ``create_curobo_world_from_*`` depth/pointcloud world builders
  (replaced by ``geometry.build_world_config`` in the geometry bundle);
- the dev tree's REPO_ROOT path plumbing (debug dirs resolve against cwd here);
- the dev tree's project-local robot-config registry entries (not part
  of this bundle; register your own via ``_PROJECT_ROBOT_CONFIGS``).

Importing this module requires cuRobo (CUDA-JIT) + torch + trimesh —
import it lazily; ``tools.py`` does.

Original header follows.

CuRobo world and motion planning helpers.

Targeting curobo v0.8.0 (cuRoboV2).  The v0.8.0 release is a major refactor
that breaks the old ``curobo.wrap.reacher.motion_gen`` API.

Migration status
----------------
``plan_directed_linear`` / ``_get_directed_motion_gen``:
    Updated to v0.8.0 (MotionPlanner + GoalToolPose + ToolPoseCriteria).

``plan_to_pose`` / ``plan_grasp_motion`` / ``plan_to_grasp_poses``:
    Updated to v0.8.0 (MotionPlanner.plan_pose / plan_grasp, native goalset,
    SceneCfg collision world, PoseCostMetric).

Remaining functions (plan_with_grasped_object, …):
    NOT YET updated — guarded by a try/except on the v0.7 imports so the
    module still loads cleanly.  They will raise RuntimeError if called.
    TODO: port to v0.8.0 API.
"""

from __future__ import (
    annotations,  # make all annotations strings (lazy) — avoids NameError for v0.7 types
)

import logging
import pathlib
import time
from typing import Any

_log = logging.getLogger(__name__)

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R_scipy

# ---------------------------------------------------------------------------
# curobo v0.7 imports — guarded; only needed by functions NOT yet ported.
# These packages no longer exist in v0.8.0; the try/except lets the module
# import cleanly even when only v0.8.0 is installed.
# TODO: remove once all functions are updated to v0.8.0 API.
# ---------------------------------------------------------------------------
_V1_AVAILABLE = False
try:
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel  # type: ignore
    from curobo.geom.sphere_fit import SphereFitType  # type: ignore
    from curobo.geom.types import Cuboid, Mesh, WorldConfig  # type: ignore
    from curobo.types.base import TensorDeviceType  # type: ignore
    from curobo.types.camera import CameraObservation  # type: ignore
    from curobo.types.math import Pose  # type: ignore
    from curobo.types.robot import JointState, RobotConfig  # type: ignore
    from curobo.types.state import JointState as StateJointState  # type: ignore
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml  # type: ignore
    from curobo.wrap.model.robot_segmenter import RobotSegmenter  # type: ignore
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig  # type: ignore
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # type: ignore
    from curobo.wrap.reacher.motion_gen import (  # type: ignore
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )
    from curobo.wrap.reacher.motion_gen import (
        PoseCostMetric as _V1PoseCostMetric,
    )
    _V1_AVAILABLE = True
    # Stash v0.7-specific callables so they are accessible from v0.7-only functions
    # even if the v0.8 block later rebinds the module-level names (e.g. get_robot_configs_path).
    _v1_get_robot_configs_path = get_robot_configs_path
    _v1_join_path = join_path
    _v1_load_yaml = load_yaml
except ImportError:
    pass  # v0.8 replaced these modules; functions using them will raise RuntimeError.

# ---------------------------------------------------------------------------
# curobo v0.8.0 imports — required by plan_directed_linear (and future ports).
# Imported AFTER the v0.7 block so they override any conflicting names.
# Guarded the same way as v0.7 so the file loads with either version.
# ---------------------------------------------------------------------------
_V2_AVAILABLE = False
try:
    from curobo._src.state.state_joint import JointState as _V2JointState  # type: ignore
    from curobo._src.types.pose import Pose as _V2Pose  # type: ignore
    from curobo._src.types.tool_pose import GoalToolPose  # type: ignore
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore
    # PoseCostMetric only powers the optional use_grasp_approach /
    # relax_orientation knobs; NVlabs upstream removed the API after the
    # dev fork's 4ea7736 state. Gate it separately so the default planning
    # paths work on upstream and the knobs fail loudly when requested.
    try:
        from curobo._src.cost.cost_pose_metric import PoseCostMetric  # noqa: F811
    except ModuleNotFoundError:
        PoseCostMetric = None  # type: ignore[assignment]
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # type: ignore
    from curobo._src.types.device_cfg import DeviceCfg  # type: ignore
    from curobo.content import get_robot_configs_path  # noqa: F811
    _V2_AVAILABLE = True
except ImportError:
    pass  # v0.7 installed — plan_directed_linear will use the MotionGen path.

# --- warp>=1.0 / curobo v0.8 collision-path compat shim ---------------------
# cuRobo v0.8's mesh/collision code (curobo/_src/geom/data/data_mesh.py,
# perception/mapper/mesh_extractor.py) still calls ``wp.torch.device_from_torch``.
# warp-lang>=1.0 (we resolve 1.13) removed the ``warp.torch`` submodule and
# promoted those helpers to the top level, so any world-collision plan
# (the pan-grasp path) raises ``AttributeError: module 'warp' has no
# attribute 'torch'``. The dev deployment never hit this (its only v0.8
# example, drawer,
# uses self_collision_check=False / no scene collision). Re-expose a
# ``warp.torch`` namespace from the promoted top-level helpers. Guarded so
# it's a no-op on warp<1.0 (where the real submodule exists). Same intent
# as the dropped el-refai v0.7 world_mesh.py patch, applied generally at
# the integration layer instead of patching the submodule.
try:
    import types as _types

    import warp as _wp  # type: ignore
    if not hasattr(_wp, "torch"):
        _wp.torch = _types.SimpleNamespace(
            device_from_torch=getattr(_wp, "device_from_torch", None),
            dtype_from_torch=getattr(_wp, "dtype_from_torch", None),
            from_torch=getattr(_wp, "from_torch", None),
            to_torch=getattr(_wp, "to_torch", None),
            stream_from_torch=getattr(_wp, "stream_from_torch", None),
        )
except Exception:  # noqa: BLE001 — warp absent/old: leave as-is
    pass

# Franka panda_hand to gripper fingertip pad center (along hand z).
# From MuJoCo panda_scene.xml:
#   finger body at pos="0 0 0.0584" from panda_hand
#   main fingertip pad (fingertip_pad_collision_1) at pos="0 0.0055 0.0445" in finger body frame
# Total: 0.0584 + 0.0445 = 0.1029 m
FRANKA_HAND_TO_FINGERTIP_Z_M = 0.0584 + 0.0445

# ---------------------------------------------------------------------------
# Robot-specific registries — keyed by CuRobo robot config filename.
# ---------------------------------------------------------------------------

# Number of actuated DOF per robot.
_ROBOT_DOF: dict[str, int] = {
    "franka.yml": 7,
    "yam.yml": 6,
}

# Project-local robot config files (outside curobo's built-in tree).
# Keys match _robot_file() strings; values are absolute paths to the yaml.
# The dev tree shipped a project-local 6-DOF arm config here; it is not
# part of this bundle. Add absolute paths to register robots whose
# configs live outside curobo's packaged robot_configs tree.
_PROJECT_ROBOT_CONFIGS: dict[str, pathlib.Path] = {}


def _load_robot_dict(robot_file: str) -> dict:
    """Return curobo robot config dict for ``robot_file``.

    Built-in curobo robots (e.g. franka.yml) resolve through curobo's packaged
    robot_configs tree. Project-local robots in _PROJECT_ROBOT_CONFIGS load from
    the repo; their ``urdf_path`` / ``asset_root_path`` are absolutized against
    the yaml's directory so curobo's ``join_path`` passes them through unchanged.
    """
    # Prefer the v0.7-stashed loaders: the v0.8 import block rebinds the
    # module-level ``get_robot_configs_path`` (and ``join_path`` / ``load_yaml``
    # only exist in v0.7), so use the aliases stashed at import time.
    _get_cfg_path = globals().get("_v1_get_robot_configs_path", None) or get_robot_configs_path
    _join = globals().get("_v1_join_path", None) or join_path
    _load = globals().get("_v1_load_yaml", None) or load_yaml
    project_path = _PROJECT_ROBOT_CONFIGS.get(robot_file)
    if project_path is None:
        return _load(_join(_get_cfg_path(), robot_file))
    robot_dict = _load(str(project_path))
    kin = robot_dict["robot_cfg"]["kinematics"]
    base_dir = project_path.parent
    if kin.get("urdf_path"):
        kin["urdf_path"] = str((base_dir / kin["urdf_path"]).resolve())
    if kin.get("asset_root_path"):
        kin["asset_root_path"] = str((base_dir / kin["asset_root_path"]).resolve())
    return robot_dict


def _get_robot_dof(robot_file: str) -> int:
    """Return the number of actuated DOF for the given robot config."""
    return _ROBOT_DOF.get(robot_file, 7)


# ---------------------------------------------------------------------------
# MotionGen cache — create once, reuse across requests (HyRL pattern).
# Avoids re-initializing CUDA tensors / graph state on every planning call.
# ---------------------------------------------------------------------------

class _MotionGenCache:
    """Cache MotionGen + RobotConfig to avoid expensive per-request recreation.

    Key insight from HyRL: MotionGen is created once at startup and reused
    via update_world() + reset().  Recreating it per request causes CUDA
    graph state corruption ("Offset increment outside graph capture").
    """

    def __init__(self):
        self._motion_gen: MotionGen | None = None
        self._robot_cfg: RobotConfig | None = None
        self._config_key: tuple | None = None

    def get(
        self,
        robot_file: str,
        robot_collision_sphere_buffer: float | None,
        tensor_args,
        use_cuda_graph: bool,
        position_threshold: float,
        rotation_threshold: float,
        num_ik_seeds: int,
        collision_activation_distance: float | None,
        world_model,
    ) -> tuple[MotionGen, RobotConfig]:
        if not _V1_AVAILABLE:
            raise RuntimeError(
                "_MotionGenCache.get() requires curobo v0.7 (MotionGen API) which is not "
                "available.  This function has not yet been ported to the curobo v0.8 API.  "
                "TODO: update to v0.8 MotionPlanner."
            )
        key = (
            robot_file, robot_collision_sphere_buffer, use_cuda_graph,
            position_threshold, rotation_threshold, num_ik_seeds,
            collision_activation_distance,
        )

        if self._motion_gen is not None and self._config_key == key:
            try:
                self._motion_gen.reset(reset_seed=False)
                if world_model is not None:
                    self._motion_gen.update_world(world_model)
                print("[MotionGenCache] Reusing cached MotionGen (skipping recreation)")
                return self._motion_gen, self._robot_cfg
            except Exception as e:
                print(f"[MotionGenCache] Failed to reuse cached MotionGen: {e}, recreating")
                self.invalidate()

        # Create fresh MotionGen
        if tensor_args is None:
            tensor_args = TensorDeviceType()

        robot_path = join_path(get_robot_configs_path(), robot_file)
        robot_dict = load_yaml(robot_path)
        if robot_collision_sphere_buffer is not None:
            inner = robot_dict.get("robot_cfg", robot_dict)
            if "kinematics" in inner:
                inner["kinematics"] = {
                    **inner["kinematics"],
                    "collision_sphere_buffer": robot_collision_sphere_buffer,
                }
                print(f"[MotionGenCache] collision_sphere_buffer={robot_collision_sphere_buffer}")
        robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)

        print(f"[MotionGenCache] Creating new MotionGen (robot={robot_file})")
        motion_gen_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_model=world_model,
            tensor_args=tensor_args,
            use_cuda_graph=use_cuda_graph,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            num_ik_seeds=num_ik_seeds,
            collision_activation_distance=collision_activation_distance,
            store_debug_in_result=True,
            interpolation_dt=1.0 / 15.0,
        )
        motion_gen = MotionGen(motion_gen_cfg)

        self._motion_gen = motion_gen
        self._robot_cfg = robot_cfg
        self._config_key = key
        print("[MotionGenCache] MotionGen created and cached")
        return motion_gen, robot_cfg

    def invalidate(self):
        """Clear cache (e.g. after CUDA error)."""
        self._motion_gen = None
        self._robot_cfg = None
        self._config_key = None


_motion_gen_cache = _MotionGenCache()


class _IKSolverCache:
    """Cache IKSolver instances per (robot_file, num_seeds, thresholds).

    SolveIK is hit hundreds of times by the prefilter step on every
    grasp leg; recreating an IKSolver per call is GPU-allocation heavy
    and causes CUDA-graph state corruption identical to MotionGen.
    """

    def __init__(self):
        self._cache: dict[tuple, Any] = {}

    def get(
        self,
        robot_file: str,
        *,
        num_seeds: int = 32,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        tensor_args=None,
    ):
        if tensor_args is None:
            tensor_args = TensorDeviceType()
        key = (robot_file, num_seeds, position_threshold, rotation_threshold)
        solver = self._cache.get(key)
        if solver is not None:
            return solver
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        robot_dict = _load_robot_dict(robot_file)
        robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)
        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,                                  # no world model — pure geometric IK
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        solver = IKSolver(ik_cfg)
        self._cache[key] = solver
        print(f"[IKSolverCache] Created IKSolver(robot={robot_file}, seeds={num_seeds})")
        return solver

    def invalidate(self) -> None:
        self._cache.clear()


_ik_solver_cache = _IKSolverCache()


def _robot_joint_names(robot_cfg: RobotConfig) -> list[str]:
    """Resolve actuated joint names for :class:`JointState` construction."""
    joint_names = None
    try:
        joint_names = getattr(robot_cfg.kinematics, "joint_names", None)
    except Exception:
        joint_names = None
    if joint_names is None:
        try:
            joint_names = robot_cfg.cspace.joint_names
        except Exception:
            joint_names = None
    if joint_names is None:
        try:
            joint_names = robot_cfg.kinematics.cspace.joint_names
        except Exception:
            joint_names = None
    if joint_names is None:
        raise AttributeError(
            "Could not determine robot joint_names from RobotConfig for trajectory validation."
        )
    return list(joint_names)


def validate_joint_trajectory_robot_world(
    world_config: Any,
    joint_waypoints: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    use_cuda_graph: bool = False,
    robot_collision_sphere_buffer: float | None = -0.01,
    collision_activation_distance: float | None = 0.01,
    ignore_obstacle_names: list[str] | None = None,
) -> tuple[bool, str, int | None, dict[str, Any]]:
    """CuRobo collision check for each joint waypoint (robot vs world + self-collision).

    Uses :meth:`MotionGen.check_start_state` per configuration. Intended for validating
    PyRoKI ``PlanLinear`` (or similar) joint trajectories against the same ``WorldConfig``
    used for perception-built meshes.

    :return: ``(success, failure_reason, first_collision_index, debug_dict)``.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    q = np.atleast_2d(np.asarray(joint_waypoints, dtype=np.float64))
    if q.shape[1] < 7:
        return False, "need_at_least_7_joint_values", None, {"shape": list(q.shape)}
    q = q[:, :7]
    n = q.shape[0]

    motion_gen, robot_cfg = _motion_gen_cache.get(
        robot_file=robot_file,
        robot_collision_sphere_buffer=robot_collision_sphere_buffer,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
        position_threshold=0.05,
        rotation_threshold=0.1,
        num_ik_seeds=32,
        collision_activation_distance=collision_activation_distance,
        world_model=world_config,
    )
    jnames = _robot_joint_names(robot_cfg)

    if ignore_obstacle_names:
        for name in ignore_obstacle_names:
            try:
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=name)
            except Exception as e:
                print(f"[validate_joint_trajectory_robot_world] could not disable '{name}': {e}")

    try:
        for i in range(n):
            row = q[i]
            js = JointState.from_position(
                tensor_args.to_device(torch.from_numpy(row).unsqueeze(0).float()),
                joint_names=jnames,
            )
            ok, status = motion_gen.check_start_state(js)
            if not ok:
                reason = (
                    getattr(status, "name", str(status))
                    if status is not None
                    else "infeasible"
                )
                meta = {
                    "motion_gen_status": str(status) if status is not None else None,
                    "joint_preview": row.tolist(),
                }
                return False, reason, i, meta
        return True, "", None, {"num_waypoints": n}
    finally:
        if ignore_obstacle_names:
            for name in ignore_obstacle_names:
                try:
                    motion_gen.world_coll_checker.enable_obstacle(enable=True, name=name)
                except Exception:
                    pass


def validate_joint_trajectory_grasped_object(
    world_config: Any,
    joint_waypoints: np.ndarray,
    object_name: str,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    use_cuda_graph: bool = False,
    robot_collision_sphere_buffer: float | None = -0.01,
    collision_activation_distance: float | None = 0.01,
    surface_sphere_radius: float = 0.001,
    link_name: str = "attached_object",
    remove_obstacles_from_world: bool = False,
) -> tuple[bool, str, int | None, dict[str, Any]]:
    """Collision check for each waypoint with the grasped object attached to the robot.

    Attaches ``object_name`` at the **first** waypoint configuration, then runs
    :meth:`MotionGen.check_start_state` for every row. Always invalidates the MotionGen
    cache afterward so attachment state does not leak.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    q = np.atleast_2d(np.asarray(joint_waypoints, dtype=np.float64))
    if q.shape[1] < 7:
        return False, "need_at_least_7_joint_values", None, {"shape": list(q.shape)}
    q = q[:, :7]
    n = q.shape[0]

    _motion_gen_cache.invalidate()
    try:
        motion_gen, robot_cfg = _motion_gen_cache.get(
            robot_file=robot_file,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            tensor_args=tensor_args,
            use_cuda_graph=use_cuda_graph,
            position_threshold=0.05,
            rotation_threshold=0.1,
            num_ik_seeds=32,
            collision_activation_distance=collision_activation_distance,
            world_model=world_config,
        )
        jnames = _robot_joint_names(robot_cfg)
        q0 = q[0].flatten()
        start_state = JointState.from_position(
            tensor_args.to_device(torch.from_numpy(q0).unsqueeze(0).float()),
            joint_names=jnames,
        )
        try:
            attach_ok = motion_gen.attach_objects_to_robot(
                start_state,
                [object_name],
                surface_sphere_radius=surface_sphere_radius,
                link_name=link_name,
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                remove_obstacles_from_world_config=remove_obstacles_from_world,
            )
        except ValueError as e:
            return False, f"attach_object_failed:{e}", 0, {"object_name": object_name}
        if not attach_ok:
            return False, "attach_objects_to_robot_failed", 0, {"object_name": object_name}

        for i in range(n):
            row = q[i]
            js = JointState.from_position(
                tensor_args.to_device(torch.from_numpy(row).unsqueeze(0).float()),
                joint_names=jnames,
            )
            ok, status = motion_gen.check_start_state(js)
            if not ok:
                reason = (
                    getattr(status, "name", str(status))
                    if status is not None
                    else "infeasible"
                )
                meta = {
                    "motion_gen_status": str(status) if status is not None else None,
                    "joint_preview": row.tolist(),
                    "object_name": object_name,
                }
                return False, reason, i, meta
        return True, "", None, {"num_waypoints": n, "object_name": object_name}
    finally:
        _motion_gen_cache.invalidate()
# Read by the gRPC server for vlog debug logging.
_last_planning_debug: dict[str, Any] | None = None

# Last post-attach collision spheres — populated after plan_with_grasped_object.
# Read by the gRPC server for PLY debug visualization.
_last_post_attach_spheres: list[tuple[np.ndarray, float]] | None = None


def _resolve_debug_dir(trace_dir: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the CuRobo debug output directory.

    When trace_dir is provided (the tools' ``debug_out_dir``), outputs go to
    <trace_dir>/data/gap/skills/curobo/.  Otherwise falls back to ./curobo_debug.

    Relative paths are resolved against the current working directory (the
    dev tree resolved them against its REPO_ROOT).
    """
    if trace_dir:
        p = pathlib.Path(trace_dir)
        if not p.is_absolute():
            # The dev tree resolved relative paths against its REPO_ROOT; here we
            # resolve against the current working directory.
            p = pathlib.Path.cwd() / p
        out = p / "data" / "gap" / "skills" / "curobo"
    else:
        out = pathlib.Path("curobo_debug")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _log_curobo_event(event_type: str, data: dict[str, Any], override_dir: str | pathlib.Path | None = None) -> None:
    """Append an event to the CuRobo debug log file in the trial trace directory."""
    import json as _json
    try:
        debug_dir = _resolve_debug_dir(override_dir)
        log_file = debug_dir / "events.jsonl"
        entry = {"timestamp": time.time(), "event": event_type, **data}
        with open(log_file, "a") as f:
            f.write(_json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi] range."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _pick_nearest_solution(q_solutions: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Pick the IK solution closest to the reference joint configuration.
    
    Args:
        q_solutions: (K, 7) array of IK solutions
        q_ref: (7,) reference joint configuration
        
    Returns:
        (7,) array of the nearest solution
    """
    # q_solutions: (K, 7), q_ref: (7,)
    dq = _wrap_to_pi(q_solutions - q_ref[None, :])
    score = np.sum(dq * dq, axis=-1)
    return q_solutions[int(np.argmin(score))]


def _grasp_pose_fingertip_to_hand(
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    hand_to_fingertip_z: float = FRANKA_HAND_TO_FINGERTIP_Z_M,
) -> np.ndarray:
    """Convert grasp position from gripper fingertip frame to panda_hand (EE) frame.
    Grasp nets typically output pose for the fingertip center; CuRobo plans for panda_hand.
    Returns new position (3,) in world frame; orientation is unchanged."""
    R = R_scipy.from_quat(np.roll(np.asarray(quat_wxyz), -1)).as_matrix()  # wxyz -> xyzw for scipy
    offset_world = R @ np.array([0.0, 0.0, hand_to_fingertip_z])
    return np.asarray(position, dtype=np.float64) - offset_world


def save_world_and_robot_spheres_debug(
    world_config,
    joint_position: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    out_dir: str | pathlib.Path = "curobo_debug",
    tag: str = "debug",
    attached_obstacle: Any = None,
    attached_obstacle_pose: tuple[np.ndarray, np.ndarray] | None = None,
    exclude_obstacle_names: list[str] | None = None,
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Save CuRobo world + robot collision spheres at a given joint configuration.

    This is a lightweight debug helper to visualize why IK / planning failed. It saves:

      - world_and_robot_spheres_{tag}.obj: single OBJ containing both the CuRobo world
        geometry (all obstacles in the WorldConfig) and the robot collision spheres at
        the given joint configuration. Optionally includes an attached object at a given pose.
      - robot_spheres_{tag}.npz: centers (N, 3) and radii (N,) of robot collision spheres.

    Use a separate script (e.g. Plotly, Open3D, or trimesh) to visualize the mesh+spheres.

    Args:
        world_config: CuRobo WorldConfig or None.
        joint_position: Joint configuration (7,) or (8,); only first 7 are used.
        robot_file: Robot config YAML under CuRobo's robot configs (default franka.yml).
        out_dir: Directory to save debug artifacts into (default ./curobo_debug).
        tag: String tag to distinguish multiple saves (e.g. "grasp_3_IK_FAIL").
        attached_obstacle: Optional CuRobo Obstacle (e.g. grasped object) to add at attached_obstacle_pose.
        attached_obstacle_pose: (position (3,), quat_wxyz (4,)) world frame; required if attached_obstacle is set.
        exclude_obstacle_names: When collecting world geometry, skip obstacles with these names (use to avoid
            drawing the attached object at its old world pose when adding it at attached_obstacle_pose).

    Returns:
        (world_path, spheres_path): Paths to saved files; either may be None on error.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    world_path: pathlib.Path | None = None
    spheres_path: pathlib.Path | None = None
    exclude_set = set(exclude_obstacle_names or [])

    # Collect world geometry for combined OBJ (skip tiny/noise meshes that appear as floating points)
    MIN_WORLD_MESH_VERTICES = 20
    MIN_WORLD_MESH_EXTENT_M = 0.025
    geometries: list[trimesh.Trimesh] = []
    if world_config is not None:
        try:
            # WorldConfig/obstacles expose get_trimesh_mesh()
            for obj in getattr(world_config, "objects", []) or []:
                if getattr(obj, "name", None) in exclude_set:
                    continue
                try:
                    m = obj.get_trimesh_mesh()
                    if m is None:
                        continue
                    n_verts = len(m.vertices) if hasattr(m.vertices, "__len__") else 0
                    if n_verts < MIN_WORLD_MESH_VERTICES:
                        continue
                    extents = getattr(m, "extents", None)
                    if extents is None and hasattr(m, "bounds") and m.bounds is not None:
                        extents = np.ptp(m.bounds, axis=0)
                    if extents is not None and float(np.max(extents)) < MIN_WORLD_MESH_EXTENT_M:
                        continue
                    geometries.append(m)
                except Exception as e:  # pragma: no cover - debug/log only
                    print(f"[curobo_debug] Warning: failed to convert world obstacle to mesh for tag={tag}: {e}")
        except Exception as e:  # pragma: no cover - debug/log only
            print(f"[curobo_debug] Warning: failed to iterate world objects for tag={tag}: {e}")

    # Add attached object: draw at its position in the world (no EE transform).
    # When the world is built with object_pose_override=EE, the object is already at the grasped pose.
    if attached_obstacle is not None:
        try:
            tensor_args = TensorDeviceType()
            mesh_added = False
            try:
                m = attached_obstacle.get_trimesh_mesh()
                if m is not None:
                    m_copy = m.copy()
                    verts = np.asarray(m_copy.vertices, dtype=np.float64)
                    pose_list = getattr(attached_obstacle, "pose", None)
                    if pose_list is not None and len(pose_list) >= 7:
                        obj_pose = Pose.from_list(pose_list, tensor_args)
                        verts_t = tensor_args.to_device(torch.from_numpy(verts).float())
                        verts = obj_pose.transform_points(verts_t).cpu().numpy()
                        m_copy.vertices = verts
                    geometries.append(m_copy)
                    mesh_added = True
            except Exception:
                pass
            if not mesh_added:
                sph_list = attached_obstacle.get_bounding_spheres(
                    n_spheres=min(128, 64),
                    surface_sphere_radius=0.002,
                    pre_transform_pose=None,
                    tensor_args=tensor_args,
                )
                for s in sph_list:
                    center = np.asarray(s.pose[:3], dtype=np.float64)
                    r = float(getattr(s, "radius", 0.01))
                    if r <= 0.0:
                        r = 0.01
                    sph_mesh = trimesh.creation.icosphere(radius=r)
                    sph_mesh.apply_translation(center)
                    geometries.append(sph_mesh)
        except Exception as e:  # pragma: no cover - debug/log only
            print(f"[curobo_debug] Warning: failed to add attached obstacle for tag={tag}: {e}")

    # Save robot collision spheres at the given joint configuration
    try:
        tensor_args = TensorDeviceType()
        robot_path = join_path(get_robot_configs_path(), robot_file)
        robot_cfg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
        robot_model = CudaRobotModel(robot_cfg.kinematics)

        q = np.asarray(joint_position, dtype=np.float64).flatten()[:7]
        q_tensor = tensor_args.to_device(torch.from_numpy(q).unsqueeze(0).float())

        spheres_batch = robot_model.get_robot_as_spheres(q_tensor)
        if not spheres_batch:
            print(f"[curobo_debug] No robot spheres found for tag={tag}.")
            centers = np.zeros((0, 3), dtype=np.float64)
            radii = np.zeros((0,), dtype=np.float64)
        else:
            # get_robot_as_spheres returns list over batch; we use the first batch element
            spheres = spheres_batch[0]
            centers = np.array(
                [np.asarray(s.position, dtype=np.float64) for s in spheres],
                dtype=np.float64,
            )
            radii = np.array([float(s.radius) for s in spheres], dtype=np.float64)

        spheres_path = out_path / f"robot_spheres_{tag}.npz"
        np.savez(spheres_path, centers=centers, radii=radii)
        print(f"[curobo_debug] Saved robot spheres to {spheres_path}")

        # Add robot spheres to combined geometry
        for center, radius in zip(centers, radii):
            if radius <= 0.0:
                continue
            try:
                sph_mesh = trimesh.creation.icosphere(radius=float(radius))
                sph_mesh.apply_translation(center.astype(np.float64))
                geometries.append(sph_mesh)
            except Exception as e:  # pragma: no cover - debug/log only
                print(f"[curobo_debug] Warning: failed to create sphere mesh for tag={tag}: {e}")

        # Export combined world + robot spheres OBJ if we have any geometry
        if geometries:
            try:
                scene = trimesh.Scene(geometries)
                world_path = out_path / f"world_and_robot_spheres_{tag}.obj"
                scene.export(str(world_path))
                print(f"[curobo_debug] Saved combined world+spheres OBJ to {world_path}")
            except Exception as e:  # pragma: no cover - debug/log only
                print(f"[curobo_debug] Failed to save combined OBJ for tag={tag}: {e}")
                world_path = None
    except Exception as e:  # pragma: no cover - debug/log only
        print(f"[curobo_debug] Failed to save robot spheres for tag={tag}: {e}")
        spheres_path = None

    return world_path, spheres_path


def extract_planning_debug_trajectories(
    result,
    robot_file: str = "franka.yml",
    tensor_args=None,
) -> dict[str, Any]:
    """Extract intermediate trajectories from a MotionGenResult for debug logging.

    Returns a dict with keys for each planning stage, each containing:
      - 'joint_traj': (T, dof) numpy array of joint positions (or None)
      - 'ee_positions': (T, 3) numpy array of EE Cartesian positions (or None)
      - 'available': bool

    Stages: graph_plan, trajopt_result, finetune_trajopt_result, final_trajectory.
    """
    if tensor_args is None:
        # v0.8 ships no TensorDeviceType (the FK helpers below are v0.7-only):
        # keep the joint trajectories and skip EE FK instead of aborting the
        # whole extraction with a NameError.
        tensor_args = TensorDeviceType() if _V1_AVAILABLE else None

    stages: dict[str, Any] = {}

    def _joint_state_to_numpy(js) -> np.ndarray | None:
        """Extract (T, dof) numpy from a JointState, handling various shapes."""
        if js is None:
            return None
        pos = js.position if hasattr(js, "position") else js
        if isinstance(pos, torch.Tensor):
            arr = pos.detach().cpu().numpy()
        else:
            arr = np.asarray(pos)
        # Squeeze batch dims: common shapes are (B, T, D) or (T, D)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _to_ee(joint_traj_np: np.ndarray | None) -> np.ndarray | None:
        if joint_traj_np is None or joint_traj_np.size == 0:
            return None
        if tensor_args is None:
            return None  # v0.8: no v0.7 FK stack — joint trajs only
        try:
            arr = joint_traj_np
            # Ensure 2D (T, dof)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[-1])
            if arr.shape[-1] < 7:
                return None
            return joint_trajectory_to_ee_positions(
                arr[:, :7],
                robot_file=robot_file,
                tensor_args=tensor_args,
            )
        except Exception as e:
            print(f"[extract_planning_debug] FK failed: {e}")
            return None

    # 1. Graph planner path
    graph_np = _joint_state_to_numpy(getattr(result, "graph_plan", None))
    stages["graph_plan"] = {
        "joint_traj": graph_np,
        "ee_positions": _to_ee(graph_np),
        "available": graph_np is not None,
        "used_graph": getattr(result, "used_graph", False),
    }

    # 2-4. From debug_info (populated when store_debug_in_result=True)
    debug_info = getattr(result, "debug_info", None) or {}

    # 2. Trajectory optimization result (pre-finetune)
    trajopt_res = debug_info.get("trajopt_result")
    trajopt_np = None
    if trajopt_res is not None:
        # TrajOptResult.solution is a JointState with the optimized trajectory
        trajopt_np = _joint_state_to_numpy(getattr(trajopt_res, "solution", None))
    stages["trajopt_result"] = {
        "joint_traj": trajopt_np,
        "ee_positions": _to_ee(trajopt_np),
        "available": trajopt_np is not None,
    }

    # 3. Finetune trajectory optimization result
    finetune_res = debug_info.get("finetune_trajopt_result")
    finetune_np = None
    if finetune_res is not None:
        finetune_np = _joint_state_to_numpy(getattr(finetune_res, "solution", None))
    stages["finetune_trajopt_result"] = {
        "joint_traj": finetune_np,
        "ee_positions": _to_ee(finetune_np),
        "available": finetune_np is not None,
    }

    # 4. Final interpolated trajectory
    final_np = None
    if result.interpolated_plan is not None:
        try:
            traj_js = result.get_interpolated_plan()
            if traj_js is not None and traj_js.position is not None:
                final_np = np.squeeze(traj_js.position.detach().cpu().numpy())
                if final_np.ndim == 1:
                    final_np = final_np.reshape(1, -1)
        except Exception:
            final_np = _joint_state_to_numpy(result.interpolated_plan)
    stages["final_trajectory"] = {
        "joint_traj": final_np,
        "ee_positions": _to_ee(final_np),
        "available": final_np is not None,
    }

    # Metadata
    stages["metadata"] = {
        "status": getattr(result.status, "name", str(result.status)) if result.status else None,
        "success": bool(result.success.item()) if result.success is not None else False,
        "used_graph": getattr(result, "used_graph", False),
        "ik_time": getattr(result, "ik_time", 0.0),
        "graph_time": getattr(result, "graph_time", 0.0),
        "trajopt_time": getattr(result, "trajopt_time", 0.0),
        "finetune_time": getattr(result, "finetune_time", 0.0),
        "total_time": getattr(result, "total_time", 0.0),
    }

    return stages


def joint_trajectory_to_ee_positions(
    joint_traj: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    ee_link_name: str = "panda_hand",
) -> np.ndarray:
    """Compute end-effector positions (T, 3) from joint trajectory (T, 7) using CuRobo FK.
    joint_traj: (T, 7) arm joint positions in radians.
    Returns (T, 3) Cartesian positions in meters (robot base frame)."""
    if tensor_args is None:
        if not _V1_AVAILABLE:
            raise RuntimeError(
                "joint_trajectory_to_ee_positions needs the curobo v0.7 FK "
                "stack (CudaRobotModel); not available on curobo v0.8"
            )
        tensor_args = TensorDeviceType()
    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_cfg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
    model = CudaRobotModel(robot_cfg.kinematics)
    q = tensor_args.to_device(torch.from_numpy(np.asarray(joint_traj, dtype=np.float32)))
    if q.ndim == 1:
        q = q.unsqueeze(0)
    state = model.forward(q, link_name=ee_link_name)
    ee_pos = state[0].detach().cpu().numpy()
    return np.squeeze(ee_pos).astype(np.float64)


def robot_joint_position_to_ee_pose(
    joint_position: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    ee_link_name: str = "panda_hand",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute EE pose (position, quat_wxyz) for a single joint config using CuRobo FK.
    joint_position: (7,) or (8,) arm joint positions in radians.
    Returns (position (3,), quaternion_wxyz (4,)) in robot base frame."""
    if tensor_args is None:
        if not _V1_AVAILABLE:
            raise RuntimeError(
                "robot_joint_position_to_ee_pose needs the curobo v0.7 FK "
                "stack (CudaRobotModel); not available on curobo v0.8"
            )
        tensor_args = TensorDeviceType()
    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_cfg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
    model = CudaRobotModel(robot_cfg.kinematics)
    q = np.asarray(joint_position, dtype=np.float32).flatten()[:7]
    q_t = tensor_args.to_device(torch.from_numpy(q).unsqueeze(0))
    state = model.forward(q_t, link_name=ee_link_name)
    ee_pos = state[0].detach().cpu().numpy().squeeze()
    ee_quat = state[1].detach().cpu().numpy().squeeze()  # wxyz
    return ee_pos.astype(np.float64), ee_quat.astype(np.float64)


def plan_to_grasp_poses(
    world_config,
    start_joint_position: np.ndarray,
    grasp_poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    max_attempts: int = 8,
    use_cuda_graph: bool = False,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.05,
    position_threshold_z: float | None = 0.01,
    grasp_pose_is_fingertip: bool = True,
    grasp_z_clearance: float = 0.005,
    num_ik_seeds: int = 128,
    relax_orientation: bool = False,
    use_grasp_approach: bool = False,
    grasp_approach_offset: float = 0.03,
    grasp_approach_linear_axis: int = 2,
    grasp_approach_tstep_fraction: float = 0.7,
    use_world_collision: bool = True,
    robot_collision_sphere_buffer: float | None = None,
    collision_activation_distance: float | None = 0.01,
    ignore_obstacle_names: list[str] | None = None,
    debug_out_dir: str | pathlib.Path | None = None,
) -> tuple[bool, np.ndarray | None, int | None]:
    """
    Plan a collision-free trajectory from the current joint configuration to one of
    the given grasp poses (end-effector position + quaternion wxyz) in the given
    CuRobo world.

    Frame convention: grasp_poses must be in the robot base frame (CuRobo uses the
    robot base as world origin). Camera extrinsics and any saved grasp poses must
    be expressed in this same frame.

    :param world_config: CuRobo WorldConfig (e.g. from create_curobo_world_from_depth). Ignored if use_world_collision=False.
    :param start_joint_position: Current joint positions (7,) or (8,); only first 7 (arm) are used.
    :param grasp_poses: List of (position (3,), quaternion_wxyz (4,)) in robot base (world) frame.
    :param robot_file: Robot config filename under curobo robot configs (default franka.yml).
    :param tensor_args: Device/dtype; default TensorDeviceType().
    :param max_attempts: Max planning attempts per grasp.
    :param use_cuda_graph: Whether to use CUDA graph (disable for varying world/start).
    :param position_threshold: Success threshold for position (m); larger allows more variance (default 0.02).
    :param rotation_threshold: Success threshold for orientation; larger allows more variance (default 0.12).
    :param position_threshold_z: If set, use max(position_threshold, position_threshold_z) so final reached
        position can have more variance (e.g. on z). Default 0.05 when collision checking. Set None to use only position_threshold.
    :param grasp_pose_is_fingertip: If True (default), treat grasp position as gripper fingertip center and
        convert to panda_hand (EE) frame using FRANKA_HAND_TO_FINGERTIP_Z_M before planning.
    :param grasp_z_clearance: Extra clearance (m) added to the fingertip z position before converting
        to panda_hand frame. Raises the whole gripper slightly to avoid table surface collisions.
        Default 0.005 (5 mm).
    :param num_ik_seeds: Number of IK seeds (default 128). Higher can fix IK_FAIL at cost of time.
    :param relax_orientation: If True (default), only require reaching goal position; orientation is relaxed
        to improve IK success. Set False to require full pose. Ignored if use_grasp_approach=True.
    :param use_grasp_approach: If True, use PoseCostMetric.create_grasp_approach_metric to bias the
        trajectory towards a two-phase motion: move towards an offset (pre-grasp) and then linearly
        approach the final grasp along a single axis (no hard stop at the offset; blended path).
    :param grasp_approach_offset: Offset (m) along the linear axis for grasp-approach cost.
    :param grasp_approach_linear_axis: Linear axis index for grasp-approach cost (0=x, 1=y, 2=z).
    :param grasp_approach_tstep_fraction: Timestep fraction in [0,1] at which to start activating the
        grasp-approach constraint (later part of the trajectory).
    :param use_world_collision: If True (default), use world_config for collision checking. If False, pass
        world_model=None to MotionGen (no obstacle collision checking).
    :param robot_collision_sphere_buffer: If set, override the robot's collision_sphere_buffer (m). Negative
        values shrink the robot's collision spheres and can reduce IK_FAIL when the world mesh is close.
        E.g. -0.01 or -0.02. Default None uses the value from the robot config file.
    :param collision_activation_distance: Distance (m) to activate collision cost; smaller is less
        conservative and can reduce IK_FAIL with dense meshes. Default 0.01. Set None to use CuRobo default.
    :param ignore_obstacle_names: If provided, these world obstacles are disabled for collision during
        planning (e.g. the object to grasp so the robot can approach it). Re-enabled before return.
    :return: (success, trajectory, goalset_index). trajectory is (T, 7) joint positions or None.

    v0.8 implementation
    -------------------
    Ported from the (removed) v0.7 MotionGen/IKSolver path onto the proven
    in-file v0.8 stack used by ``plan_to_pose`` / ``plan_directed_linear``:

    * Collision-aware ``MotionPlanner`` from :func:`_get_pose_planner`
      (self-collision always on; mesh ``collision_cache`` so a scene can be
      loaded). ``plan_grasp_motion`` was NOT reused because it builds on
      :func:`_get_directed_planner`, which is created with
      ``self_collision_check=False`` and **no collision cache** — it cannot do
      the world-collision-aware planning that is the entire purpose of this
      function (``use_world_collision=True`` by default), takes a single pose
      (not a goalset with reachability fallback + ``goalset_index``), and
      returns a 5-tuple of approach/grasp/lift segments rather than this
      function's ``(success, traj[T,7], idx)`` contract.
    * **Native goalset**: all candidate poses are packed into a single
      ``GoalToolPose`` with ``num_goalset = len(grasp_poses)`` and one
      ``MotionPlanner.plan_pose`` call.  v0.8 ``plan_pose`` auto-detects
      ``num_goalset > 1`` (motion_planner.py:224) and returns the reached
      candidate via ``result.goalset_index`` — this is the native, robust
      replacement for the v0.7 per-pose Python loop and directly yields the
      "first reachable candidate + which index" semantics.  A per-candidate
      ``try/except`` fallback loop is still kept for reachability robustness if
      the goalset call yields nothing.
    * World collision via :func:`_world_to_v2_scene_cfg` +
      ``planner.clear_scene_cache()`` / ``planner.update_world(SceneCfg)``
      (mirrors ``plan_to_pose``).  ``ignore_obstacle_names`` meshes are
      excluded when building the ``SceneCfg`` (the v0.8 scene checker is
      rebuilt every call, so there is no obstacle re-enable bookkeeping).
    * ``relax_orientation`` → ``PoseCostMetric.reach_position_metric``;
      ``use_grasp_approach`` → ``PoseCostMetric.create_grasp_approach_metric``;
      applied/reset via ``ik_solver``/``trajopt_solver``
      ``update_pose_cost_metric`` exactly as ``plan_directed_linear`` does.

    ``tensor_args`` is accepted for signature compatibility but is NOT used:
    the v0.7 ``TensorDeviceType`` no longer exists, so device/dtype come from
    the planner's v0.8 ``DeviceCfg`` (same as every other v0.8 function here).
    """
    if not _V2_AVAILABLE:
        raise RuntimeError(
            "plan_to_grasp_poses: curobo v0.8 (cuRoboV2) is required.  The v0.7 "
            "MotionGen/IKSolver API was removed in the v0.8.0 refactor.  "
            "Re-install curobo v0.8.0 (third_party/curobo at the v0.8.0 commit) "
            "and run `uv sync`."
        )

    dof = _get_robot_dof(robot_file)

    n = len(grasp_poses)

    if n > _GRASP_MAX_GOALSET:

        print(

            f"[plan_to_grasp_poses] trimming {n} candidates to the planner "

            f"goalset capacity ({_GRASP_MAX_GOALSET})."

        )

        grasp_poses = grasp_poses[:_GRASP_MAX_GOALSET]

        n = _GRASP_MAX_GOALSET
    if n == 0:
        print("[plan_to_grasp_poses] No grasp poses; returning False, None, None.")
        return False, None, None

    # ── Threshold mapping (v0.7 -> v0.8) ────────────────────────────────────
    # v0.7 used max(position_threshold, position_threshold_z) as the IK/plan
    # success position threshold.  v0.8's MotionPlanner has a single
    # ``position_tolerance`` (no separate per-axis z tolerance — there is no
    # v0.8 equivalent, so the conservative max is used, matching v0.7).
    effective_position_threshold = position_threshold
    if position_threshold_z is not None:
        effective_position_threshold = max(position_threshold, position_threshold_z)
        if effective_position_threshold > position_threshold:
            print(
                f"[plan_to_grasp_poses] Using position_threshold="
                f"{effective_position_threshold:.3f} m (z allowance: "
                f"position_threshold_z={position_threshold_z}; v0.8 has no "
                f"separate z tolerance, applying the max)."
            )
    print(
        "[plan_to_grasp_poses] v0.8 thresholds: "
        f"position_threshold={position_threshold}, "
        f"rotation_threshold={rotation_threshold}, "
        f"position_threshold_z={position_threshold_z}, "
        f"effective_position_threshold={effective_position_threshold}"
    )
    # robot_collision_sphere_buffer / collision_activation_distance have no
    # direct knob on MotionPlannerCfg.create (verified motion_planner_cfg.py:
    # create() exposes self_collision_check / num_ik_seeds / position_tolerance
    # / orientation_tolerance / use_cuda_graph / collision_cache only).  v0.7
    # used them to relax collision near dense meshes; on v0.8 they are ignored
    # gracefully rather than failing the call.
    if robot_collision_sphere_buffer is not None:
        print(
            f"[plan_to_grasp_poses] NOTE: robot_collision_sphere_buffer="
            f"{robot_collision_sphere_buffer} has no v0.8 MotionPlannerCfg "
            f"equivalent; ignored."
        )
    if collision_activation_distance is not None:
        print(
            f"[plan_to_grasp_poses] NOTE: collision_activation_distance="
            f"{collision_activation_distance} has no v0.8 MotionPlannerCfg "
            f"equivalent; ignored."
        )

    # ── World collision setup ───────────────────────────────────────────────
    # Build the v0.8 SceneCfg, excluding any ``ignore_obstacle_names`` meshes
    # (e.g. the object being grasped) so the robot may approach them.  Done at
    # SceneCfg-build time because the v0.8 scene checker is rebuilt per call —
    # equivalent to v0.7's enable_obstacle(False)/re-enable, with no bookkeeping.
    has_world = bool(
        use_world_collision
        and world_config is not None
        and len(list(getattr(world_config, "mesh", None) or [])) > 0
    )
    if not use_world_collision:
        print(
            "[plan_to_grasp_poses] use_world_collision=False: planning "
            "without obstacle collision (self-collision still enforced)."
        )
    n_mesh_total = (
        len(list(getattr(world_config, "mesh", None) or []))
        if world_config is not None else 0
    )

    planner = _get_pose_planner(
        robot_file,
        with_collision=has_world,
        # Generous floor so update_world() never overflows the mesh cache and
        # the same planner survives scenes with slightly different mesh counts.
        mesh_cache=max(n_mesh_total + 4, 32),
        position_threshold=effective_position_threshold,
        rotation_threshold=rotation_threshold,
        num_ik_seeds=num_ik_seeds,
        use_cuda_graph=use_cuda_graph,
        # Fixed goalset capacity: one cached planner serves every call; the
        # candidate list is trimmed to this cap below.
        max_goalset=_GRASP_MAX_GOALSET,
    )
    planner.reset_seed()

    device_cfg = planner.config.device_cfg
    joint_names = planner.joint_names
    n_dof = len(joint_names)

    if has_world:
        try:
            scene_cfg = _v2_scene_cfg_excluding(
                world_config, device_cfg, ignore_obstacle_names
            )
            if scene_cfg is not None:
                planner.clear_scene_cache()
                planner.update_world(scene_cfg)
                print(
                    f"[plan_to_grasp_poses] Loaded v0.8 collision scene "
                    f"(meshes={n_mesh_total}, "
                    f"ignored={list(ignore_obstacle_names or [])})."
                )
            else:
                # use_world_collision=True with a world that supplied meshes
                # but none survived (e.g. all ignored / malformed): a scene
                # was requested but could not be built — do NOT silently plan
                # collision-free.
                raise RuntimeError(
                    "world had meshes but the v0.8 SceneCfg is empty after "
                    "filtering ignore_obstacle_names / dropping malformed "
                    "meshes."
                )
        except Exception as e:
            raise RuntimeError(
                f"plan_to_grasp_poses: failed to load collision world into "
                f"v0.8 MotionPlanner: {e}"
            ) from e

    # ── Start JointState (v0.8) — shape [1, DOF], mirrors plan_to_pose ───────
    q_start = np.asarray(start_joint_position, dtype=np.float64)[:7].flatten()
    print(f"[plan_to_grasp_poses] start_joint_position (7): {q_start.tolist()}")
    print(
        f"[plan_to_grasp_poses] start valid: "
        f"no_nan={not np.any(np.isnan(q_start))}, "
        f"no_inf={not np.any(np.isinf(q_start))}, "
        f"in_range=[{q_start.min():.3f}, {q_start.max():.3f}]"
    )
    cfg = np.asarray(q_start, dtype=np.float32).flatten()
    if len(cfg) > n_dof:
        cfg = cfg[:n_dof]
    elif len(cfg) < n_dof:
        cfg = np.concatenate([cfg, np.zeros(n_dof - len(cfg), dtype=np.float32)])
    start_state = _V2JointState.from_numpy(
        joint_names=joint_names,
        position=np.expand_dims(cfg, 0),
        device_cfg=device_cfg,
    )

    # ── Convert fingertip -> panda_hand for every candidate (same as v0.7) ───
    tool_frame = _FRANKA_TOOL_FRAME
    if planner.tool_frames:
        tool_frame = planner.tool_frames[0]

    hand_positions: list[np.ndarray] = []
    hand_quats: list[np.ndarray] = []
    for idx in range(n):
        pos = np.asarray(grasp_poses[idx][0], dtype=np.float64).reshape(3)
        quat = np.asarray(grasp_poses[idx][1], dtype=np.float64).reshape(4)
        if grasp_pose_is_fingertip:
            if grasp_z_clearance != 0.0:
                pos = pos.copy()
                pos[2] += grasp_z_clearance
            pos = _grasp_pose_fingertip_to_hand(pos, quat)
            print(
                f"[plan_to_grasp_poses] grasp {idx}/{n}: position "
                f"(panda_hand after offset, z_clearance={grasp_z_clearance}): "
                f"{np.round(pos, 4).tolist()}, quat_wxyz={np.round(quat, 4).tolist()}"
            )
        else:
            print(
                f"[plan_to_grasp_poses] grasp {idx}/{n}: "
                f"position={np.round(pos, 4).tolist()}, "
                f"quat_wxyz={np.round(quat, 4).tolist()}"
            )
        hand_positions.append(np.asarray(pos, dtype=np.float32).reshape(3))
        hand_quats.append(np.asarray(quat, dtype=np.float32).reshape(4))

    # ── Pose cost metric (relax_orientation / use_grasp_approach) ───────────
    # Same precedence as v0.7: grasp-approach wins over relax_orientation.
    pose_metric = None
    if (use_grasp_approach or relax_orientation) and PoseCostMetric is None:
        raise RuntimeError(
            "use_grasp_approach/relax_orientation need curobo's PoseCostMetric "
            "API, which this curobo build does not ship (NVlabs upstream "
            "removed it). Install the dev-era fork SHA or drop the knob."
        )
    if use_grasp_approach:
        pose_metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=grasp_approach_offset,
            linear_axis=grasp_approach_linear_axis,
            tstep_fraction=grasp_approach_tstep_fraction,
            device_cfg=device_cfg,
        )
        print(
            "[plan_to_grasp_poses] v0.8 grasp-approach metric: "
            f"offset={grasp_approach_offset}, "
            f"axis={grasp_approach_linear_axis}, "
            f"tstep_fraction={grasp_approach_tstep_fraction}."
        )
    elif relax_orientation:
        # v0.8 built-in position-only metric (reach_vec_weight zeros out
        # position-error axes, keeps orientation free) — cost_pose_metric.py:
        # reach_position_metric L93-100.  Replaces v0.7's manual
        # reach_partial_pose + reach_vec_weight=[..,0.2].
        pose_metric = PoseCostMetric.reach_position_metric(device_cfg=device_cfg)
        print(
            "[plan_to_grasp_poses] v0.8 position-only reach "
            "(relax_orientation=True) to improve IK success."
        )

    def _apply_metric(metric) -> None:
        if metric is None:
            return
        planner.ik_solver.update_pose_cost_metric({tool_frame: metric})
        planner.trajopt_solver.update_pose_cost_metric({tool_frame: metric})

    def _reset_metric() -> None:
        if pose_metric is None:
            return
        reset = PoseCostMetric.reset_metric()
        planner.ik_solver.update_pose_cost_metric({tool_frame: reset})
        planner.trajopt_solver.update_pose_cost_metric({tool_frame: reset})

    global _last_planning_debug

    def _result_status(result) -> str:
        return (
            getattr(result, "status", None)
            or getattr(result, "failure_reason", None)
            or "N/A"
        )

    def _result_success(result) -> bool:
        return bool(
            result is not None
            and result.success is not None
            and torch.any(result.success).item()
        )

    def _traj_from_result(result) -> np.ndarray | None:
        try:
            traj_js = result.get_interpolated_plan()
        except Exception as e:  # pragma: no cover - debug/log only
            print(f"[plan_to_grasp_poses] trajectory extraction failed: {e}")
            return None
        if traj_js is None or traj_js.position is None:
            return None
        arr = np.squeeze(traj_js.position.detach().cpu().numpy())
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr[:, :dof].astype(np.float64)

    def _goalset_idx(result, fallback: int) -> int:
        # v0.8 trajopt result stores goalset_index [batch, num_seeds, n_links]
        # (solver_trajopt_result.py L241-249); plan_grasp itself reads it via
        # .view(-1)[0] (motion_planner.py L461-462) — mirror that.
        gi = getattr(result, "goalset_index", None)
        if gi is None:
            return fallback
        try:
            return int(gi.view(-1)[0].item())
        except Exception:
            return fallback

    _apply_metric(pose_metric)
    try:
        # ── Primary path: single native-goalset plan_pose call ──────────────
        # GoalToolPose.from_poses with num_goalset=N expects the per-link Pose
        # to have position [N,3] / quaternion [N,4] (tool_pose.py L264-273:
        # .view(batch=total//num_goalset, num_goalset, 3)).  Stack all
        # candidates into one Pose, batch == 1.
        goalset_pos = device_cfg.to_device(
            torch.tensor(np.stack(hand_positions, axis=0), dtype=torch.float32)
        )  # [N, 3]
        goalset_quat = device_cfg.to_device(
            torch.tensor(np.stack(hand_quats, axis=0), dtype=torch.float32)
        )  # [N, 4]
        goalset_v2pose = _V2Pose(
            position=goalset_pos, quaternion=goalset_quat, name=tool_frame
        )
        goalset_tool_poses = GoalToolPose.from_poses(
            {tool_frame: goalset_v2pose},
            ordered_tool_frames=[tool_frame],
            num_goalset=n,
        )

        print(
            f"[plan_to_grasp_poses] v0.8 goalset plan_pose | tool={tool_frame} "
            f"n_candidates={n} world={'on(%d mesh)' % n_mesh_total if has_world else 'off'} "
            f"relax_orientation={relax_orientation} "
            f"use_grasp_approach={use_grasp_approach} max_attempts={max_attempts}"
        )

        _t0 = time.monotonic()
        result = planner.plan_pose(
            goalset_tool_poses, start_state, max_attempts=max_attempts
        )
        _t_ms = (time.monotonic() - _t0) * 1000.0
        _log.info(
            "[TIMING] v0.8 plan_to_grasp_poses goalset n=%d world=%s plan_ms=%.1f t=%.3f",
            n, has_world, _t_ms, time.time(),
        )

        if _result_success(result):
            idx = _goalset_idx(result, 0)
            if idx < 0 or idx >= n:
                idx = 0
            traj = _traj_from_result(result)
            if traj is not None:
                try:
                    _last_planning_debug = extract_planning_debug_trajectories(
                        result, robot_file=robot_file, tensor_args=None,
                    )
                    _last_planning_debug["metadata"]["grasp_index"] = idx
                    _last_planning_debug["metadata"]["planning_function"] = (
                        "plan_to_grasp_poses"
                    )
                except Exception as e:
                    print(
                        f"[plan_to_grasp_poses] Warning: failed to extract "
                        f"debug trajectories: {e}"
                    )
                    _last_planning_debug = None
                _log_curobo_event("grasp_plan_result", {
                    "success": True,
                    "status": str(_result_status(result)),
                    "grasp_index": idx,
                    "grasp_pos": hand_positions[idx].tolist(),
                    "start_joint": q_start.tolist(),
                    "use_world_collision": use_world_collision,
                    "function": "plan_to_grasp_poses",
                    "mode": "goalset",
                }, override_dir=debug_out_dir)
                print(
                    f"[plan_to_grasp_poses] v0.8 goalset SUCCESS: reached "
                    f"candidate idx={idx}, trajectory shape={traj.shape}."
                )
                return True, traj, idx
            print(
                "[plan_to_grasp_poses] goalset plan reported success but no "
                "interpolated trajectory; falling back to per-candidate."
            )
        else:
            print(
                f"[plan_to_grasp_poses] v0.8 goalset plan_pose did not "
                f"succeed (status={_result_status(result)}); falling back to "
                f"per-candidate planning for reachability robustness."
            )

        # ── Fallback path: per-candidate single-goal plan_pose ──────────────
        # Preserves the v0.7 "one bad pose must not abort the rest" semantics:
        # try each candidate independently; first reachable wins.
        #
        # Order by descending goal height: for thin/flat objects most
        # low-clearance candidates are table-colliding (each burning a full
        # max_attempts plan, ~10-15s), while the high-clearance variants are
        # the ones that succeed — trying them first cuts the ladder from
        # minutes to seconds without dropping any candidate.
        fallback_order = sorted(
            range(n), key=lambda i: float(hand_positions[i][2]), reverse=True
        )
        for idx in fallback_order:
            try:
                g_pos = device_cfg.to_device(
                    torch.tensor(hand_positions[idx], dtype=torch.float32).unsqueeze(0)
                )
                g_quat = device_cfg.to_device(
                    torch.tensor(hand_quats[idx], dtype=torch.float32).unsqueeze(0)
                )
                g_v2pose = _V2Pose(
                    position=g_pos, quaternion=g_quat, name=tool_frame
                )
                g_tool_poses = GoalToolPose.from_poses(
                    {tool_frame: g_v2pose},
                    ordered_tool_frames=[tool_frame],
                    num_goalset=1,
                )
                _ts = time.monotonic()
                result = planner.plan_pose(
                    g_tool_poses, start_state, max_attempts=max_attempts
                )
                _tms = (time.monotonic() - _ts) * 1000.0
                ok = _result_success(result)
                status_str = _result_status(result)
                print(
                    f"[plan_to_grasp_poses] grasp {idx}: v0.8 plan_pose "
                    f"success={ok} status={status_str} plan_ms={_tms:.1f}"
                )

                try:
                    _last_planning_debug = extract_planning_debug_trajectories(
                        result, robot_file=robot_file, tensor_args=None,
                    )
                    _last_planning_debug["metadata"]["grasp_index"] = idx
                    _last_planning_debug["metadata"]["planning_function"] = (
                        "plan_to_grasp_poses"
                    )
                except Exception as e:
                    print(
                        f"[plan_to_grasp_poses] Warning: failed to extract "
                        f"debug trajectories: {e}"
                    )
                    _last_planning_debug = None

                _log_curobo_event("grasp_plan_result", {
                    "success": ok,
                    "status": str(status_str),
                    "grasp_index": idx,
                    "grasp_pos": hand_positions[idx].tolist(),
                    "start_joint": q_start.tolist(),
                    "use_world_collision": use_world_collision,
                    "function": "plan_to_grasp_poses",
                    "mode": "per_candidate",
                }, override_dir=debug_out_dir)

                if ok:
                    traj = _traj_from_result(result)
                    if traj is not None:
                        print(
                            f"[plan_to_grasp_poses] Returning first success "
                            f"at grasp index {idx} (per-candidate), "
                            f"trajectory shape={traj.shape}."
                        )
                        return True, traj, idx
                    print(
                        f"[plan_to_grasp_poses] Warning: grasp {idx} "
                        f"succeeded but trajectory extraction returned None."
                    )
            except Exception as e:
                # Reachability robustness: a single bad pose must not abort
                # the remaining candidates.
                print(
                    f"[plan_to_grasp_poses] Warning: grasp {idx} raised "
                    f"{type(e).__name__}: {e}; trying next candidate."
                )
                continue

        print(
            "[plan_to_grasp_poses] No grasp succeeded; returning "
            "False, None, None."
        )
        return False, None, None
    finally:
        # Always reset the pose cost metric so a cached planner starts clean
        # next call (same discipline as plan_directed_linear).
        try:
            _reset_metric()
        except Exception as e:  # pragma: no cover - cleanup only
            print(
                f"[plan_to_grasp_poses] Warning: failed to reset pose cost "
                f"metric: {e}"
            )


def _v2_scene_cfg_excluding(
    world_config, device_cfg, ignore_obstacle_names: list[str] | None
) -> Any | None:
    """Build a v0.8 ``SceneCfg`` from ``world_config`` excluding named meshes.

    Thin wrapper over the proven :func:`_world_to_v2_scene_cfg` (used by the
    working v0.8 ``plan_to_pose``).  When ``ignore_obstacle_names`` is set, the
    matching meshes are dropped from the v0.7 ``WorldConfig`` *before*
    conversion so they are never loaded into the v0.8 scene checker — the v0.8
    equivalent of v0.7's ``world_coll_checker.enable_obstacle(False, name)``.
    The scene checker is rebuilt on every call (``clear_scene_cache`` +
    ``update_world``), so there is nothing to re-enable afterwards.

    Returns the ``SceneCfg`` (or ``None`` if nothing remains to add).
    """
    if world_config is None:
        return None
    if not ignore_obstacle_names:
        return _world_to_v2_scene_cfg(world_config, device_cfg)

    ignore = set(ignore_obstacle_names)
    meshes_in = list(getattr(world_config, "mesh", None) or [])
    kept = [m for m in meshes_in if getattr(m, "name", None) not in ignore]
    dropped = [
        getattr(m, "name", None) for m in meshes_in
        if getattr(m, "name", None) in ignore
    ]
    if dropped:
        print(
            f"[plan_to_grasp_poses] Excluding {len(dropped)} obstacle(s) from "
            f"v0.8 collision scene: {dropped}"
        )

    # Build a lightweight stand-in exposing only ``.mesh`` (all
    # _world_to_v2_scene_cfg reads); avoids mutating the caller's WorldConfig.
    class _FilteredWorld:
        pass

    fw = _FilteredWorld()
    fw.mesh = kept
    return _world_to_v2_scene_cfg(fw, device_cfg)


def _build_transport_debug_ply(
    world_config, q_start, robot_file, tensor_args,
    attached_obstacle, object_name, target_pos,
    planning_debug, out_dir, tag,
    post_attach_spheres=None,
):
    """Build a combined PLY with world mesh, robot spheres, object at EE, and all trajectories.

    Returns the combined trimesh mesh, or None if no geometry was built.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()

    scene_parts = []

    def _tube(points, radius, color):
        parts = []
        for i in range(len(points) - 1):
            seg = trimesh.creation.cylinder(radius=radius, segment=[points[i], points[i + 1]])
            seg.visual.vertex_colors = np.tile(color, (len(seg.vertices), 1))
            parts.append(seg)
        for p in points:
            sp = trimesh.creation.icosphere(radius=radius * 1.5)
            sp.apply_translation(p)
            sp.visual.vertex_colors = np.tile(color, (len(sp.vertices), 1))
            parts.append(sp)
        return parts

    def _marker(pos, radius, color):
        sp = trimesh.creation.icosphere(radius=radius)
        sp.apply_translation(pos)
        sp.visual.vertex_colors = np.tile(color, (len(sp.vertices), 1))
        return sp

    def _joint_to_ee(joint_traj):
        """Convert (T, 7) or (N, T, 7) joint traj to list of (T, 3) EE paths."""
        arr = np.array(joint_traj)
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (1, T, 7)
        results = []
        for i in range(arr.shape[0]):
            try:
                seed = arr[i]
                if seed.shape[-1] > 7:
                    seed = seed[:, :7]
                ee = joint_trajectory_to_ee_positions(seed, robot_file=robot_file, tensor_args=tensor_args)
                if ee.ndim == 1:
                    ee = ee.reshape(1, 3)
                if ee.shape[0] > 1:
                    results.append(ee)
            except Exception:
                pass
        return results

    # 1. World mesh (gray, basket orange)
    if world_config is not None:
        exclude_set = {object_name} if object_name else set()
        for obj in getattr(world_config, "objects", []) or []:
            if getattr(obj, "name", None) in exclude_set:
                continue
            try:
                m = obj.get_trimesh_mesh()
                if m is None or len(m.vertices) < 20:
                    continue
                m_copy = m.copy()
                verts = np.array(m_copy.vertices)
                colors = np.tile([180, 180, 180, 60], (len(verts), 1))
                basket_mask = (
                    (verts[:, 0] > 0.40) & (verts[:, 0] < 0.80) &
                    (verts[:, 1] > 0.10) & (verts[:, 1] < 0.45) &
                    (verts[:, 2] > 0.0) & (verts[:, 2] < 0.20)
                )
                colors[basket_mask] = [255, 140, 0, 180]
                m_copy.visual.vertex_colors = colors
                scene_parts.append(m_copy)
            except Exception:
                pass

    # 2. Robot + attached object collision spheres
    # Robot + attached object collision spheres
    # Get pre-attach spheres (fresh model, no attachment) to identify which are robot body
    robot_path = join_path(get_robot_configs_path(), robot_file)
    robot_cfg_dbg = RobotConfig.from_dict(load_yaml(robot_path), tensor_args)
    robot_model_dbg = CudaRobotModel(robot_cfg_dbg.kinematics)
    q_t = tensor_args.to_device(torch.from_numpy(q_start.astype(np.float32)).unsqueeze(0))
    pre_batch = robot_model_dbg.get_robot_as_spheres(q_t, filter_valid=False)
    # Build set of pre-attach sphere positions (robot body only)
    pre_attach_positions = set()
    if pre_batch:
        for s in pre_batch[0]:
            if float(s.radius) > 0.001:
                p = tuple(np.round(np.array(s.position), 5))
                pre_attach_positions.add(p)

    if post_attach_spheres is not None:
        for center, radius in post_attach_spheres:
            if radius < 0.001:
                continue
            sph = trimesh.creation.icosphere(radius=radius)
            sph.apply_translation(center)
            p_key = tuple(np.round(center, 5))
            if p_key in pre_attach_positions:
                # Robot body sphere (blue)
                sph.visual.vertex_colors = np.tile([70, 130, 200, 80], (len(sph.vertices), 1))
            else:
                # Attached object collision sphere (red, opaque)
                sph.visual.vertex_colors = np.tile([255, 50, 50, 200], (len(sph.vertices), 1))
            scene_parts.append(sph)
    else:
        if pre_batch:
            for s in pre_batch[0]:
                r = float(s.radius)
                if r < 0.005:
                    continue
                sph = trimesh.creation.icosphere(radius=r)
                sph.apply_translation(np.array(s.position))
                sph.visual.vertex_colors = np.tile([70, 130, 200, 80], (len(sph.vertices), 1))
                scene_parts.append(sph)

    # 3. Attached object at EE position (yellow)
    # The mesh from get_trimesh_mesh() has its world pose baked in (table position).
    # We undo that pose, then apply the current EE pose.
    if attached_obstacle is not None:
        try:
            m = attached_obstacle.get_trimesh_mesh()
            if m is not None:
                m_copy = m.copy()
                from scipy.spatial.transform import Rotation as _R

                # Undo the obstacle's world pose (baked into vertices)
                pose_list = getattr(attached_obstacle, "pose", None)
                if pose_list is not None and len(pose_list) >= 7:
                    # pose_list = [x, y, z, qw, qx, qy, qz]
                    obj_pos = np.array(pose_list[:3], dtype=np.float64)
                    obj_quat_wxyz = np.array(pose_list[3:7], dtype=np.float64)
                    obj_rot = _R.from_quat(np.roll(obj_quat_wxyz, -1)).as_matrix()
                    obj_inv = np.eye(4)
                    obj_inv[:3, :3] = obj_rot.T
                    obj_inv[:3, 3] = -obj_rot.T @ obj_pos
                    m_copy.apply_transform(obj_inv)

                # Apply EE pose
                ee_pos_dbg, ee_quat_dbg = robot_joint_position_to_ee_pose(
                    q_start, robot_file=robot_file, tensor_args=tensor_args
                )
                ee_rot = _R.from_quat(np.roll(ee_quat_dbg, -1)).as_matrix()  # wxyz -> xyzw
                ee_transform = np.eye(4)
                ee_transform[:3, :3] = ee_rot
                ee_transform[:3, 3] = ee_pos_dbg
                m_copy.apply_transform(ee_transform)
                m_copy.visual.vertex_colors = np.tile([255, 255, 0, 200], (len(m_copy.vertices), 1))
                scene_parts.append(m_copy)
        except Exception as e:
            print(f"[PLY debug] Warning: failed to place object at EE: {e}")

    # 4. Trajectories — all candidates in light orange, final (shortest) in green
    def _pick_shortest_idx(ee_list):
        """Return index of shortest path by Cartesian path length."""
        if not ee_list:
            return -1
        best_idx, best_len = 0, float("inf")
        for i, ee in enumerate(ee_list):
            plen = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=-1)))
            if plen < best_len:
                best_idx, best_len = i, plen
        return best_idx

    if planning_debug is not None:
        # Stage colors (distinct, easy to tell apart):
        #   Graph seeds  = light gray    (thin)
        #   Trajopt      = orange        (thin)
        #   Finetune     = cyan/teal     (medium)
        #   Final (best) = bright green  (thick)

        # 1. Graph seeds — light gray
        gj = planning_debug.get("graph_plan", {}).get("joint_traj")
        if gj is not None:
            for ee in _joint_to_ee(gj):
                scene_parts.extend(_tube(ee, 0.002, [180, 180, 180, 120]))

        # 2. Trajopt — orange
        tj = planning_debug.get("trajopt_result", {}).get("joint_traj")
        if tj is not None:
            for ee in _joint_to_ee(tj):
                scene_parts.extend(_tube(ee, 0.003, [255, 140, 0, 180]))

        # 3. Finetune — cyan
        fj_fine = planning_debug.get("finetune_trajopt_result", {}).get("joint_traj")
        if fj_fine is not None:
            for ee in _joint_to_ee(fj_fine):
                scene_parts.extend(_tube(ee, 0.004, [0, 220, 220, 220]))
        else:
            fe = planning_debug.get("finetune_trajopt_result", {}).get("ee_positions")
            if fe is not None and fe.ndim >= 2 and fe.shape[0] > 1:
                scene_parts.extend(_tube(fe, 0.004, [0, 220, 220, 220]))

        # 4. Final — pick shortest candidate, draw in bright green (thick)
        fj_final = planning_debug.get("final_trajectory", {}).get("joint_traj")
        final_candidates = []
        if fj_final is not None:
            final_candidates = _joint_to_ee(fj_final)
        fin_ee = None
        if final_candidates:
            best_idx = _pick_shortest_idx(final_candidates)
            fin_ee = final_candidates[best_idx]
        if fin_ee is None:
            fe = planning_debug.get("final_trajectory", {}).get("ee_positions")
            if fe is not None and fe.ndim >= 2 and fe.shape[0] > 1:
                fin_ee = fe

        if fin_ee is not None and fin_ee.ndim >= 2 and fin_ee.shape[0] > 1:
            scene_parts.extend(_tube(fin_ee, 0.006, [0, 230, 0, 255]))
            scene_parts.append(_marker(fin_ee[0], 0.012, [0, 230, 0, 255]))
            scene_parts.append(_marker(fin_ee[-1], 0.012, [0, 150, 0, 255]))
            min_idx = np.argmin(fin_ee[:, 2])
            scene_parts.append(_marker(fin_ee[min_idx], 0.01, [255, 0, 255, 255]))

    # 5. Target (red)
    scene_parts.append(_marker(target_pos, 0.015, [255, 0, 0, 255]))

    if scene_parts:
        combined = trimesh.util.concatenate(scene_parts)
        ply_path = pathlib.Path(out_dir) / f"curobo_transport_debug_{tag}.ply"
        combined.export(str(ply_path))
        print(f"[plan_with_grasped_object] PLY debug saved to {ply_path} ({ply_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return combined
    return None


def plan_with_grasped_object(
    world_config: Any,
    start_joint_position: np.ndarray,
    target_pose: tuple[np.ndarray, np.ndarray],
    object_name: str,
    *,
    robot_file: str = "franka.yml",
    tensor_args=None,
    max_attempts: int = 8,
    use_cuda_graph: bool = False,
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    position_threshold_z: float | None = 0.05,
    num_ik_seeds: int = 128,
    use_world_collision: bool = True,
    robot_collision_sphere_buffer: float | None = None,
    collision_activation_distance: float | None = 0.01,
    surface_sphere_radius: float = 0.001,
    link_name: str = "attached_object",
    remove_obstacles_from_world: bool = False,
    debug_out_dir: str | pathlib.Path | None = "curobo_debug2",
) -> tuple[bool, np.ndarray | None]:
    """
    Plan a collision-free trajectory to move a grasped object to a target pose.

    This function:
    1. Attaches the object (by name) from world_config to the robot at the current joint state
    2. Computes IK for the goal pose, selecting the solution closest to the start joint state
    3. Plans a trajectory in joint space from start to the IK solution
    4. Returns the joint trajectory

    The object must exist in world_config with the given object_name. After attaching,
    the object moves with the robot and is checked for collisions with the remaining
    scene obstacles.

    If planning fails, debug world meshes are saved to debug_out_dir (default: "curobo_debug2"):
    - start_joint_state_{tag}.obj: World + robot at start configuration
    - end_joint_state_{tag}.obj: World + robot at IK goal configuration (if IK succeeded)

    :param world_config: CuRobo WorldConfig containing the object (by object_name) and scene obstacles.
    :param start_joint_position: Current joint positions (7,) or (8,); object is assumed grasped at this state.
    :param target_pose: (position (3,), quaternion_wxyz (4,)) target pose in robot base (world) frame.
    :param object_name: Name of the object in world_config to attach to the robot.
    :param robot_file: Robot config filename under curobo robot configs (default franka.yml).
    :param tensor_args: Device/dtype; default TensorDeviceType().
    :param max_attempts: Max planning attempts.
    :param use_cuda_graph: Whether to use CUDA graph (disable for varying world/start).
    :param position_threshold: Success threshold for position (m).
    :param rotation_threshold: Success threshold for orientation.
    :param position_threshold_z: If set, use max(position_threshold, position_threshold_z).
    :param num_ik_seeds: Number of IK seeds (default 128).
    :param use_world_collision: If True, use world_config for collision checking.
    :param robot_collision_sphere_buffer: Override robot collision_sphere_buffer (m).
    :param collision_activation_distance: Distance (m) to activate collision cost.
    :param surface_sphere_radius: Radius (m) for surface spheres when attaching object.
    :param link_name: Link name to attach object to (default "attached_object").
    :param remove_obstacles_from_world: Remove attached object from world after attaching.
    :param debug_out_dir: Directory to save debug meshes on planning failure (default "curobo_debug2"). Set to None to disable.
    :return: (success, joint_trajectory). joint_trajectory is (T, 7) or None.
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    device = tensor_args.device
    dtype = tensor_args.dtype

    # Always provide world_config to MotionGen so attach_objects_to_robot can
    # look up object geometry.  When use_world_collision is False we still pass
    # the world model (needed for attach) but log that collision avoidance is
    # effectively bypassed via collision_activation_distance / sphere_buffer.
    world_model = world_config
    if not use_world_collision:
        print("[plan_with_grasped_object] use_world_collision=False: world model still provided for object attachment, but collision avoidance is relaxed.")

    # For pose planning with z variance: use position_threshold for x, y and a large value for z
    # Set position_threshold_z to a large value to allow any z position
    if position_threshold_z is None:
        position_threshold_z = 10.0  # Allow any z by default
    effective_position_threshold = max(position_threshold, position_threshold_z)
    if effective_position_threshold > position_threshold:
        print(f"[plan_with_grasped_object] Using position_threshold={effective_position_threshold:.3f} m (z allowance: position_threshold_z={position_threshold_z:.3f}).")

    # plan_with_grasped_object uses attach_objects_to_robot which mutates
    # MotionGen state, so we always need a fresh MotionGen for this path.
    # Invalidate cache to prevent stale attachment state leaking.
    _motion_gen_cache.invalidate()
    motion_gen, robot_cfg = _motion_gen_cache.get(
        robot_file=robot_file,
        robot_collision_sphere_buffer=robot_collision_sphere_buffer,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
        position_threshold=effective_position_threshold,
        rotation_threshold=rotation_threshold,
        num_ik_seeds=num_ik_seeds,
        collision_activation_distance=collision_activation_distance,
        world_model=world_model,
    )

    q_start = np.asarray(start_joint_position, dtype=np.float64)
    q_start = q_start[:7].flatten()
    print(f"[plan_with_grasped_object] start_joint_position (7): {q_start.tolist()}")
    start_state = JointState.from_position(
        tensor_args.to_device(torch.from_numpy(q_start).unsqueeze(0).float())
    )

    # Capture attached object reference before attach (so we can add it to debug meshes at EE pose)
    attached_obstacle = world_config.get_obstacle(object_name) if world_config else None

    # Attach object to robot
    print(f"[plan_with_grasped_object] Attaching object '{object_name}' to robot at link '{link_name}'...")
    try:
        attach_success = motion_gen.attach_objects_to_robot(
            start_state,
            [object_name],
            surface_sphere_radius=surface_sphere_radius,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            remove_obstacles_from_world_config=remove_obstacles_from_world,
        )
    except ValueError as e:
        # CuRobo raises ValueError when the object is not found in the world
        msg = f"Object '{object_name}' not found in world: {e}"
        print(f"[plan_with_grasped_object] {msg}")
        world_names = [getattr(m, "name", "?") for m in (getattr(world_config, "objects", []) or [])] if world_config else []
        _log_curobo_event("attach_object_not_found", {
            "object_name": object_name,
            "world_objects": world_names,
            "error": str(e),
            "function": "plan_with_grasped_object",
        }, override_dir=debug_out_dir)
        return False, None
    if not attach_success:
        msg = f"Failed to attach object '{object_name}'. Check that it exists in world_config."
        print(f"[plan_with_grasped_object] {msg}")
        world_names = [getattr(m, "name", "?") for m in (getattr(world_config, "objects", []) or [])] if world_config else []
        _log_curobo_event("attach_failed", {
            "object_name": object_name,
            "world_objects": world_names,
            "function": "plan_with_grasped_object",
        }, override_dir=debug_out_dir)
        return False, None
    print(f"[plan_with_grasped_object] Successfully attached object '{object_name}' to robot.")

    # Extract post-attach collision spheres, separating robot body vs attached object
    # Before attach: attached_object link spheres have radius <= 0 (pre-allocated slots)
    # After attach: those slots get positive radii from the fitted surface spheres
    _post_attach_spheres = None
    _pre_attach_radii = None
    try:
        _kin_model = motion_gen.kinematics
        q_t = tensor_args.to_device(torch.from_numpy(q_start.astype(np.float32)).unsqueeze(0))
        _sph_batch = _kin_model.get_robot_as_spheres(q_t, filter_valid=False)
        if _sph_batch:
            all_spheres = _sph_batch[0]
            _post_attach_spheres = []
            n_robot = 0
            n_attached = 0
            for s in all_spheres:
                r = float(s.radius)
                if r <= 0.0:
                    continue
                pos = np.array(s.position, dtype=np.float64)
                # Spheres belonging to the attached_object link are at the EE
                # and were added by attach_objects_to_robot
                _post_attach_spheres.append((pos, r))
            print(f"[plan_with_grasped_object] Post-attach collision spheres: {len(_post_attach_spheres)}")
    except Exception as e:
        print(f"[plan_with_grasped_object] Warning: failed to get post-attach spheres: {e}")

    # Resolve debug output directory
    _debug_out = _resolve_debug_dir(debug_out_dir)
    _debug_tag = f"transport_{int(time.time())}"

    # Save pre-planning debug: world meshes + robot spheres + attached object
    # This shows exactly what CuRobo sees for collision checking
    try:
        # Get EE pose for attached object visualization
        ee_pos_dbg, ee_quat_dbg = robot_joint_position_to_ee_pose(
            q_start, robot_file=robot_file, tensor_args=tensor_args
        )
        save_world_and_robot_spheres_debug(
            world_config=world_config,
            joint_position=q_start,
            robot_file=robot_file,
            out_dir=_debug_out,
            tag=_debug_tag,
            attached_obstacle=attached_obstacle,
            attached_obstacle_pose=(ee_pos_dbg, ee_quat_dbg),
            exclude_obstacle_names=[object_name],
        )
        # Also log world mesh names and counts
        if world_config is not None:
            mesh_names = [getattr(m, "name", "?") for m in (getattr(world_config, "objects", []) or [])]
            print(f"[plan_with_grasped_object] World meshes for collision: {mesh_names}")
            print(f"[plan_with_grasped_object] use_world_collision={use_world_collision}")
        print(f"[plan_with_grasped_object] Debug saved to {_debug_out / f'world_and_robot_spheres_{_debug_tag}.obj'}")
    except Exception as e:
        print(f"[plan_with_grasped_object] Warning: failed to save pre-planning debug: {e}")

    # Convert target pose to CuRobo Pose
    pos = np.asarray(target_pose[0], dtype=np.float64)
    quat = np.asarray(target_pose[1], dtype=np.float64)
    print(f"[plan_with_grasped_object] Target pose: position={pos.tolist()}, quat_wxyz={quat.tolist()}")
    print(f"[plan_with_grasped_object] Allowing z variance (position_threshold_z={position_threshold_z:.3f} m), matching x, y, orientation")

    goal_pose = Pose(
        position=tensor_args.to_device(torch.from_numpy(pos).unsqueeze(0).float()),
        quaternion=tensor_args.to_device(torch.from_numpy(quat).unsqueeze(0).float()),
    )

    # Plan directly to pose with z variance enabled
    # The effective_position_threshold allows z variance while still matching x, y, orientation
    plan_cfg = MotionGenPlanConfig(
        pose_cost_metric=PoseCostMetric.reset_metric(),
        max_attempts=max_attempts,
        enable_graph_attempt=True,
    )

    print(f"[plan_with_grasped_object] Planning to pose with z variance (position_threshold={effective_position_threshold:.3f} m)...")
    result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)

    success = bool(result.success.item() if result.success is not None else False)
    status_str = getattr(result.status, "name", str(result.status)) if hasattr(result, "status") and result.status is not None else "N/A"
    print(f"[plan_with_grasped_object] Planning result: success={success}, status={status_str}")

    # Log planning result
    _log_curobo_event("transport_plan_result", {
        "success": success,
        "status": status_str,
        "object_name": object_name,
        "start_joint": q_start.tolist(),
        "target_pos": pos.tolist(),
        "use_world_collision": use_world_collision,
        "surface_sphere_radius": surface_sphere_radius,
        "collision_activation_distance": collision_activation_distance,
        "robot_collision_sphere_buffer": robot_collision_sphere_buffer,
        "n_post_attach_spheres": len(_post_attach_spheres) if _post_attach_spheres else 0,
        "function": "plan_with_grasped_object",
    }, override_dir=debug_out_dir)

    # Expose post-attach spheres for server-side PLY debug
    global _last_post_attach_spheres
    _last_post_attach_spheres = _post_attach_spheres

    # Extract intermediate trajectories for debug logging
    global _last_planning_debug
    try:
        _last_planning_debug = extract_planning_debug_trajectories(
            result, robot_file=robot_file, tensor_args=tensor_args,
        )
        _last_planning_debug["metadata"]["planning_function"] = "plan_with_grasped_object"
        _last_planning_debug["metadata"]["object_name"] = object_name
    except Exception as e:
        print(f"[plan_with_grasped_object] Warning: failed to extract debug trajectories: {e}")
        _last_planning_debug = None

    # Save all intermediate trajectories (joint + EE) + combined PLY
    try:
        traj_save = {}
        if _last_planning_debug is not None:
            for stage in ["graph_plan", "trajopt_result", "finetune_trajopt_result", "final_trajectory"]:
                sd = _last_planning_debug.get(stage, {})
                if sd.get("available"):
                    if sd.get("joint_traj") is not None:
                        traj_save[f"{stage}_joint"] = sd["joint_traj"]
                    if sd.get("ee_positions") is not None:
                        traj_save[f"{stage}_ee"] = sd["ee_positions"]
            traj_save["start_joint"] = q_start
            traj_save["target_pos"] = pos
            traj_save["target_quat"] = quat
        if traj_save:
            npz_path = _debug_out / f"trajectories_{_debug_tag}.npz"
            np.savez(str(npz_path), **traj_save)
            print(f"[plan_with_grasped_object] Trajectory debug saved to {npz_path}")

        # Build combined PLY: world mesh + robot spheres + all trajectories
        _build_transport_debug_ply(
            world_config, q_start, robot_file, tensor_args,
            attached_obstacle, object_name, pos,
            _last_planning_debug, _debug_out, _debug_tag,
            post_attach_spheres=_post_attach_spheres,
        )
    except Exception as e:
        print(f"[plan_with_grasped_object] Warning: failed to save trajectory debug: {e}")

    if success and result.interpolated_plan is not None:
        try:
            traj_js = result.get_interpolated_plan()
            if traj_js is not None and traj_js.position is not None:
                trajectory = np.squeeze(traj_js.position.detach().cpu().numpy())
                print(f"[plan_with_grasped_object] Planning succeeded; trajectory shape: {trajectory.shape}")
                return True, trajectory
        except Exception as e:
            print(f"[plan_with_grasped_object] Warning: failed to extract trajectory despite success: {e}")

    # Planning failed - save debug meshes if requested
    if debug_out_dir is not None:
        tag = f"plan_fail_{int(time.time())}"
        print(f"[plan_with_grasped_object] Planning failed; saving debug meshes to {debug_out_dir}...")
        
        # Save start joint state (include attached object at EE pose)
        try:
            ee_pos, ee_quat = robot_joint_position_to_ee_pose(
                q_start, robot_file=robot_file, tensor_args=tensor_args
            )
            save_world_and_robot_spheres_debug(
                world_config=world_config,
                joint_position=q_start,
                robot_file=robot_file,
                out_dir=debug_out_dir,
                tag=f"{tag}_start",
                attached_obstacle=attached_obstacle,
                attached_obstacle_pose=(ee_pos, ee_quat),
                exclude_obstacle_names=[object_name],
            )
        except Exception as e:
            print(f"[plan_with_grasped_object] Warning: failed to save start debug mesh: {e}")
        
        # Try to compute IK for goal pose to save end joint state debug mesh
        try:
            ik_world = world_model
            ik_cfg = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_model=ik_world,
                tensor_args=tensor_args,
                num_seeds=num_ik_seeds,
                position_threshold=effective_position_threshold,  # Use same threshold as planning
                rotation_threshold=rotation_threshold,
                self_collision_check=True,
            )
            ik_solver = IKSolver(ik_cfg)
            retract_cfg = tensor_args.to_device(torch.from_numpy(q_start).unsqueeze(0).float())
            ik_result = ik_solver.solve_single(goal_pose, retract_config=retract_cfg)
            
            if bool(ik_result.success.item() if ik_result.success is not None else False):
                q_sols = ik_result.solution.detach().cpu().numpy()
                q_sols = np.atleast_2d(q_sols)[:, :7]
                q_goal_debug = _pick_nearest_solution(q_sols, q_start)
                ee_pos_end, ee_quat_end = robot_joint_position_to_ee_pose(
                    q_goal_debug, robot_file=robot_file, tensor_args=tensor_args
                )
                save_world_and_robot_spheres_debug(
                    world_config=world_config,
                    joint_position=q_goal_debug,
                    robot_file=robot_file,
                    out_dir=debug_out_dir,
                    tag=f"{tag}_end_ik",
                    attached_obstacle=attached_obstacle,
                    attached_obstacle_pose=(ee_pos_end, ee_quat_end),
                    exclude_obstacle_names=[object_name],
                )
            else:
                print("[plan_with_grasped_object] IK failed for debug mesh; skipping end state debug mesh.")
        except Exception as e:
            print(f"[plan_with_grasped_object] Warning: failed to compute IK and save end debug mesh: {e}")

    print("[plan_with_grasped_object] Planning failed; returning False, None.")
    return False, None


# ---------------------------------------------------------------------------
# v0.8 MotionPlanner cache for plan_directed_linear.
# Keyed by robot_file string; replaces the old _directed_mg / _directed_rcfg.
# ---------------------------------------------------------------------------

_directed_planner_cache: dict[str, MotionPlanner] = {}

_AXIS_IDX = {"X": 0, "Y": 1, "Z": 2}

# Franka tool frame name in the v0.8 robot YAML.
_FRANKA_TOOL_FRAME = "panda_hand"

# ---------------------------------------------------------------------------
# v0.7 (MotionGen) cache for plan_directed_linear — used when _V2_AVAILABLE=False.
# ---------------------------------------------------------------------------
_directed_mg_v1: MotionGen | None = None
_directed_rcfg_v1: RobotConfig | None = None


def _get_directed_motion_gen_v1(robot_file: str = "franka.yml"):
    """v0.7: lazily create a collision-free MotionGen for constrained linear planning."""
    global _directed_mg_v1, _directed_rcfg_v1
    if _directed_mg_v1 is not None:
        _directed_mg_v1.reset(reset_seed=False)
        return _directed_mg_v1, _directed_rcfg_v1

    tensor_args = TensorDeviceType()
    robot_path = _v1_join_path(_v1_get_robot_configs_path(), robot_file)
    robot_dict = _v1_load_yaml(robot_path)
    rcfg = RobotConfig.from_dict(robot_dict, tensor_args)
    print("[plan_directed_linear] v1: Creating MotionGen (no collision)")
    mg_cfg = MotionGenConfig.load_from_robot_config(
        rcfg,
        world_model=None,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        self_collision_check=False,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_ik_seeds=32,
        interpolation_dt=1.0 / 15.0,
    )
    _directed_mg_v1 = MotionGen(mg_cfg)
    _directed_rcfg_v1 = rcfg
    print("[plan_directed_linear] v1: MotionGen ready")
    return _directed_mg_v1, _directed_rcfg_v1


def _plan_directed_linear_v1(
    start_config,
    start_pose=None,
    *,
    target_pose=None,
    allowed_axes=None,
    explicit_direction=None,
    distance=None,
    endpoint_mode="PROJECT_TO_TARGET",
    orientation_mode="LOCK",
    orientation_target=None,
    robot_file="franka.yml",
):
    """v0.7 (MotionGen) implementation of plan_directed_linear."""
    motion_gen, robot_cfg = _get_directed_motion_gen_v1(robot_file)
    motion_gen.reset(reset_seed=False)

    tensor_args = motion_gen.tensor_args
    joint_names = _robot_joint_names(robot_cfg)
    n_dof = len(joint_names)

    cfg = np.array(start_config, dtype=np.float32)
    if len(cfg) > n_dof:
        cfg = cfg[:n_dof]
    elif len(cfg) < n_dof:
        cfg = np.concatenate([cfg, np.zeros(n_dof - len(cfg), dtype=np.float32)])

    start_state = StateJointState.from_numpy(
        joint_names=joint_names,
        position=np.expand_dims(cfg, 0),
        tensor_args=tensor_args,
    )

    fk = motion_gen.compute_kinematics(start_state)
    fk_pos = fk.ee_pose.position.squeeze().cpu().numpy()
    fk_quat = fk.ee_pose.quaternion.squeeze().cpu().numpy()

    mode = endpoint_mode.upper()

    if mode == "ORIENT_IN_PLACE":
        free_idx: set = set()
        held_idx: set = {0, 1, 2}
        goal_pos = fk_pos.copy()
    else:
        if allowed_axes is None:
            allowed_axes = ["X", "Y", "Z"]
        free_idx = {_AXIS_IDX[a.upper()] for a in allowed_axes if a.upper() in _AXIS_IDX}
        held_idx = {0, 1, 2} - free_idx
        goal_pos = fk_pos.copy()
        if mode == "DISTANCE":
            if explicit_direction is not None and distance is not None:
                d = np.array(explicit_direction, dtype=np.float32)
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    d = d / norm
                goal_pos = fk_pos + distance * d
            else:
                return False, None, "DISTANCE mode requires explicit_direction + distance"
        else:
            if target_pose is None:
                return False, None, "PROJECT_TO_TARGET needs target_pose"
            tgt_pos = np.array(target_pose[0], dtype=np.float32)
            for i in free_idx:
                goal_pos[i] = tgt_pos[i]

    if orientation_mode.upper() == "LOCK":
        goal_quat = fk_quat.copy()
    else:
        if orientation_target is not None:
            goal_quat = np.array(orientation_target, dtype=np.float32)
        elif target_pose is not None:
            goal_quat = np.array(target_pose[1], dtype=np.float32)
        else:
            goal_quat = fk_quat.copy()

    hvw = [0.0] * 6
    if orientation_mode.upper() in ("LOCK", "SLERP"):
        hvw[0] = hvw[1] = hvw[2] = 1.0
    for i in held_idx:
        hvw[3 + i] = 1.0

    hvw_t = tensor_args.to_device(torch.tensor(hvw, dtype=torch.float32))

    pose_metric = None
    if any(v > 0 for v in hvw):
        pose_metric = _V1PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=hvw_t,
            project_to_goal_frame=False,
        )

    goal = Pose(
        position=tensor_args.to_device(torch.tensor(goal_pos, dtype=torch.float32).unsqueeze(0)),
        quaternion=tensor_args.to_device(torch.tensor(goal_quat, dtype=torch.float32).unsqueeze(0)),
    )
    plan_cfg = MotionGenPlanConfig(
        max_attempts=10,
        enable_graph=False,
        enable_opt=True,
        enable_finetune_trajopt=True,
        pose_cost_metric=pose_metric,
    )

    print(
        f"[plan_directed_linear] v1 | axes={allowed_axes} orient={orientation_mode} "
        f"hvw={hvw} goal_pos={np.round(goal_pos, 4).tolist()} fk_pos={np.round(fk_pos, 4).tolist()}"
    )

    _t_plan_start = time.monotonic()
    result = motion_gen.plan_single(start_state, goal, plan_cfg)
    _t_plan_ms = (time.monotonic() - _t_plan_start) * 1000.0
    _log.info(
        "[TIMING] v1 plan_directed_linear axes=%s mode=%s orient=%s plan_ms=%.1f t=%.3f",
        allowed_axes, mode, orientation_mode, _t_plan_ms, time.time(),
    )
    with open("/tmp/curobo_timing.log", "a") as _tf:
        _tf.write(
            f"v1\taxes={allowed_axes}\tmode={mode}\torient={orientation_mode}"
            f"\tplan_ms={_t_plan_ms:.1f}\tt={time.time():.3f}\n"
        )
    success = bool(result.success.item() if result.success is not None else False)
    status_str = getattr(result.status, "name", str(result.status)) if hasattr(result, "status") else "N/A"
    print(f"[plan_directed_linear] v1 success={success} status={status_str}")

    if not success:
        return False, None, f"motion_gen_failed_{status_str}"
    try:
        traj_js = result.get_interpolated_plan()
        if traj_js is not None and traj_js.position is not None:
            traj = np.squeeze(traj_js.position.detach().cpu().numpy())
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            traj = traj[:, :7]
            print(f"[plan_directed_linear] v1 trajectory shape: {traj.shape}")
            return True, traj, ""
    except Exception as e:
        return False, None, f"trajectory_extraction: {e}"
    return False, None, "no_interpolated_plan"


def _build_linear_tool_pose_criteria(
    held_idx: set,
    orientation_mode: str,
    mode: str,
    device_cfg: DeviceCfg,
) -> ToolPoseCriteria:
    """Build ToolPoseCriteria that penalises intermediate-waypoint deviations.

    In curobo v0.8, ``ToolPoseCriteria.non_terminal_pose_axes_weight_factor``
    is the only mechanism that constrains the **path** (not just the terminal
    pose).  Its default is all-zeros, which means the optimiser can swing
    through any arc as long as it arrives at the goal — causing the visible
    orientation drift mid-move.  Setting non-zero weights here forces the
    intermediate waypoints to stay on the desired straight/constrained line.

    Format: ``[x, y, z, roll, pitch, yaw]``, 1.0 = held, 0.0 = free.
    """
    if mode == "ORIENT_IN_PLACE":
        # Position fully fixed; only orientation changes.
        pos_non_term = [1.0, 1.0, 1.0]
        rot_non_term = [0.0, 0.0, 0.0]  # orientation free along path
    else:
        # Held position axes stay fixed at all intermediate waypoints.
        pos_non_term = [1.0 if i in held_idx else 0.0 for i in range(3)]
        if orientation_mode.upper() in ("LOCK", "SLERP"):
            rot_non_term = [1.0, 1.0, 1.0]  # orientation locked along path
        else:  # TARGET_AT_END — orientation interpolates freely
            rot_non_term = [0.0, 0.0, 0.0]

    non_terminal = [*pos_non_term, *rot_non_term]
    terminal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # full pose at terminal

    # project_distance_to_goal helps the optimiser measure progress along the
    # correct axis when exactly one position axis is free.
    use_projection = (len(held_idx) == 2) or (mode == "ORIENT_IN_PLACE")

    return ToolPoseCriteria(
        terminal_pose_axes_weight_factor=terminal,
        non_terminal_pose_axes_weight_factor=non_terminal,
        project_distance_to_goal=use_projection,
        device_cfg=device_cfg,
    )


def _get_directed_planner(robot_file: str = "franka.yml") -> MotionPlanner:
    """Lazily create (and cache) a v0.8 MotionPlanner for constrained linear planning.

    No collision environment is attached — the planner is purely kinematic,
    matching the v0.7 behaviour of ``_get_directed_motion_gen``.
    """
    global _directed_planner_cache
    if robot_file in _directed_planner_cache:
        print(f"[plan_directed_linear] Reusing cached MotionPlanner for {robot_file}")
        return _directed_planner_cache[robot_file]

    print(f"[plan_directed_linear] Creating MotionPlanner (v0.8, no collision) for {robot_file}")
    device_cfg = DeviceCfg()
    cfg = MotionPlannerCfg.create(
        robot=robot_file,
        self_collision_check=False,
        device_cfg=device_cfg,
        num_ik_seeds=32,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        use_cuda_graph=True,
    )
    planner = MotionPlanner(cfg)
    _directed_planner_cache[robot_file] = planner
    print(f"[plan_directed_linear] MotionPlanner ready (tool_frames={planner.tool_frames})")
    return planner


def plan_directed_linear(
    start_config: np.ndarray,
    start_pose: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    target_pose: tuple[np.ndarray, np.ndarray] | None = None,
    allowed_axes: list[str] | None = None,
    explicit_direction: np.ndarray | None = None,
    distance: float | None = None,
    endpoint_mode: str = "PROJECT_TO_TARGET",
    orientation_mode: str = "LOCK",
    orientation_target: np.ndarray | None = None,
    robot_file: str = "franka.yml",
) -> tuple[bool, np.ndarray | None, str]:
    """Constrained linear trajectory via curobo v0.8 MotionPlanner + PoseCostMetric.

    Computes FK(start_config) internally to derive the true start EE pose, then
    builds a GoalToolPose and PoseCostMetric that enforce the requested
    axis/orientation constraints during trajectory optimisation.

    API contract is identical to the v0.7 implementation — the server.py caller
    does not need any changes.

    Args:
        start_config: current joint positions (7-DOF arm).
        start_pose:   hint from GetEEPose — used only as fallback, not for planning.
        target_pose:  (position (3,), quaternion_wxyz (4,)) — desired target.
        allowed_axes: subset of ["X","Y","Z"] that may change. Others are held
                      at the FK(start_config) value.
        explicit_direction: unit Vec3 for DISTANCE mode.
        distance:     metres to travel along explicit_direction.
        endpoint_mode:
            "PROJECT_TO_TARGET" — free-axis values come from target_pose,
                                  held-axis values come from FK(start).
            "DISTANCE"          — goal = FK(start) + distance * direction.
            "ORIENT_IN_PLACE"   — position stays exactly at FK(start);
                                  only orientation changes (toward
                                  orientation_target or target_pose.quat).
                                  allowed_axes is ignored.
        orientation_mode:
            "LOCK"          — goal orientation = FK(start) orientation.
            "TARGET_AT_END" — goal orientation = orientation_target (or
                              target_pose quaternion).
            "SLERP"         — same as TARGET_AT_END for goal, but with
                              rotation held throughout the path.
        orientation_target: explicit wxyz quaternion for TARGET_AT_END/SLERP.
        robot_file:   curobo robot YAML (e.g. "franka.yml").

    Returns:
        (success, trajectory_Tx7, failure_reason)
    """
    if not _V2_AVAILABLE:
        raise RuntimeError(
            "plan_directed_linear: curobo v0.8 (cuRoboV2) is required.  "
            "v0.7 fallback is disabled for the drawer-angle-sweep experiment.  "
            "Re-install curobo v0.8.0 (third_party/curobo at the v0.8.0 commit) and run `uv sync`."
        )

    planner = _get_directed_planner(robot_file)
    planner.reset_seed()

    device_cfg = planner.config.device_cfg
    joint_names = planner.joint_names
    n_dof = len(joint_names)

    cfg = np.array(start_config, dtype=np.float32)
    if len(cfg) > n_dof:
        cfg = cfg[:n_dof]
    elif len(cfg) < n_dof:
        cfg = np.concatenate([cfg, np.zeros(n_dof - len(cfg), dtype=np.float32)])

    # v0.8 JointState — shape [1, DOF]
    start_state = _V2JointState.from_numpy(
        joint_names=joint_names,
        position=np.expand_dims(cfg, 0),
        device_cfg=device_cfg,
    )

    # ── FK to get true start EE pose ────────────────────────────────────────
    kin = planner.compute_kinematics(start_state)
    # ToolPose position shape: [B, H, num_links, 3] → squeeze to (3,)
    fk_pos = kin.tool_poses.position[0, 0, 0, :].cpu().numpy()
    fk_quat = kin.tool_poses.quaternion[0, 0, 0, :].cpu().numpy()  # wxyz

    # ── Resolve endpoint mode ────────────────────────────────────────────────
    mode = endpoint_mode.upper()

    if mode == "ORIENT_IN_PLACE":
        free_idx: set[int] = set()
        held_idx = {0, 1, 2}
        goal_pos = fk_pos.copy()
    else:
        if allowed_axes is None:
            allowed_axes = ["X", "Y", "Z"]
        free_idx = {_AXIS_IDX[a.upper()] for a in allowed_axes if a.upper() in _AXIS_IDX}
        held_idx = {0, 1, 2} - free_idx
        goal_pos = fk_pos.copy()

        if mode == "DISTANCE":
            if explicit_direction is not None and distance is not None:
                d = np.array(explicit_direction, dtype=np.float32)
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    d = d / norm
                goal_pos = fk_pos + distance * d
            else:
                return False, None, "DISTANCE mode requires explicit_direction + distance"
        else:  # PROJECT_TO_TARGET
            if target_pose is None:
                return False, None, "PROJECT_TO_TARGET needs target_pose"
            tgt_pos = np.array(target_pose[0], dtype=np.float32)
            for i in free_idx:
                goal_pos[i] = tgt_pos[i]

    # ── Build goal orientation ──────────────────────────────────────────────
    if orientation_mode.upper() == "LOCK":
        goal_quat = fk_quat.copy()
    else:  # TARGET_AT_END or SLERP
        if orientation_target is not None:
            goal_quat = np.array(orientation_target, dtype=np.float32)
        elif target_pose is not None:
            goal_quat = np.array(target_pose[1], dtype=np.float32)
        else:
            goal_quat = fk_quat.copy()

    # ── Build hold_vec_weight [rx, ry, rz, x, y, z] and PoseCostMetric ──────
    # Same semantics as v0.7: 1.0 = held (constrained), 0.0 = free.
    hvw = [0.0] * 6
    if orientation_mode.upper() in ("LOCK", "SLERP"):
        hvw[0] = hvw[1] = hvw[2] = 1.0
    for i in held_idx:
        hvw[3 + i] = 1.0

    tool_frame = _FRANKA_TOOL_FRAME
    if planner.tool_frames:
        tool_frame = planner.tool_frames[0]

    pose_metric: PoseCostMetric | None = None
    if any(v > 0 for v in hvw):
        hvw_t = device_cfg.to_device(torch.tensor(hvw, dtype=torch.float32))
        pose_metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=hvw_t,
        )

    # ── Build GoalToolPose ────────────────────────────────────────────────────
    goal_pos_t = device_cfg.to_device(
        torch.tensor(goal_pos, dtype=torch.float32).unsqueeze(0)
    )
    goal_quat_t = device_cfg.to_device(
        torch.tensor(goal_quat, dtype=torch.float32).unsqueeze(0)
    )
    goal_v2pose = _V2Pose(position=goal_pos_t, quaternion=goal_quat_t, name=tool_frame)
    goal_tool_poses = GoalToolPose.from_poses({tool_frame: goal_v2pose})

    print(
        f"[plan_directed_linear] v0.8 | axes={allowed_axes} orient={orientation_mode} "
        f"hvw={hvw} goal_pos={np.round(goal_pos, 4).tolist()} fk_pos={np.round(fk_pos, 4).tolist()}"
    )

    # ── Apply PoseCostMetric (goal-side axis holding) ─────────────────────────
    if pose_metric is not None:
        planner.ik_solver.update_pose_cost_metric({tool_frame: pose_metric})
        planner.trajopt_solver.update_pose_cost_metric({tool_frame: pose_metric})

    # ── Apply ToolPoseCriteria (path-side intermediate waypoint constraints) ──
    # This is what actually prevents orientation/position drift mid-move.
    # PoseCostMetric only constrains axis values relative to the start pose;
    # ToolPoseCriteria.non_terminal_pose_axes_weight_factor penalises each
    # intermediate optimisation step for deviating from the straight line.
    path_criteria = _build_linear_tool_pose_criteria(held_idx, orientation_mode, mode, device_cfg)
    planner.update_tool_pose_criteria({tool_frame: path_criteria})

    _t_plan_start = time.monotonic()
    try:
        result = planner.plan_pose(goal_tool_poses, start_state, max_attempts=10)
    finally:
        # Always reset both constraints so subsequent calls start clean.
        reset_metric = PoseCostMetric.reset_metric()
        planner.ik_solver.update_pose_cost_metric({tool_frame: reset_metric})
        planner.trajopt_solver.update_pose_cost_metric({tool_frame: reset_metric})
        # Reset ToolPoseCriteria to default (all-zero non-terminal = unconstrained path).
        planner.update_tool_pose_criteria({tool_frame: ToolPoseCriteria(device_cfg=device_cfg)})
        _t_plan_ms = (time.monotonic() - _t_plan_start) * 1000.0
        _log.info(
            "[TIMING] v0.8 plan_directed_linear axes=%s mode=%s orient=%s plan_ms=%.1f t=%.3f",
            allowed_axes, mode, orientation_mode, _t_plan_ms, time.time(),
        )
        with open("/tmp/curobo_timing.log", "a") as _tf:
            _tf.write(
                f"v0.8\taxes={allowed_axes}\tmode={mode}\torient={orientation_mode}"
                f"\tplan_ms={_t_plan_ms:.1f}\tt={time.time():.3f}\n"
            )

    if result is None:
        return False, None, "motion_planner_returned_none"

    success = bool(
        result.success is not None and torch.any(result.success).item()
    )
    status_str = (
        getattr(result, "status", None) or
        getattr(result, "failure_reason", None) or
        "N/A"
    )
    print(f"[plan_directed_linear] v0.8 success={success} status={status_str}")

    if not success:
        return False, None, f"motion_gen_failed_{status_str}"

    try:
        traj_js = result.get_interpolated_plan()
        if traj_js is not None and traj_js.position is not None:
            traj = np.squeeze(traj_js.position.detach().cpu().numpy())
            if traj.ndim == 1:
                traj = traj.reshape(1, -1)
            traj = traj[:, :7]
            print(f"[plan_directed_linear] v0.8 trajectory shape: {traj.shape}")
            return True, traj, ""
    except Exception as e:
        return False, None, f"trajectory_extraction: {e}"

    return False, None, "no_interpolated_plan"


def _extract_traj_v2(js) -> np.ndarray | None:
    """Extract a (T, 7) float32 numpy array from a curobo v0.8 JointState."""
    if js is None or js.position is None:
        return None
    traj = np.squeeze(js.position.detach().cpu().numpy())
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    return traj[:, :7]


def plan_grasp_motion(
    start_config,
    grasp_pose,
    approach_axis: str = "y",
    approach_distance: float = 0.12,
    approach_in_tool_frame: bool = False,
    lift_axis: str = "y",
    lift_distance: float = 0.20,
    lift_in_tool_frame: bool = False,
    plan_approach: bool = True,
    plan_lift: bool = True,
    robot_file: str = "franka.yml",
):
    """Plan a grasp sequence using curobo v0.8 ``MotionPlanner.plan_grasp``.

    Internally ``plan_grasp`` runs four planning steps:

    1. Free-space plan to one of the ``grasp_poses`` (goalset selection).
    2. Free-space plan to the *approach* pose (grasp offset by approach_axis *
       approach_distance away from the grasp pose).
    3. Constrained linear plan from approach → grasp (along approach_axis).
    4. Constrained linear plan from grasp → lift pose (along lift_axis *
       lift_distance).

    The returned trajectories correspond to steps 2, 3, and 4.  Execute them
    in order with a gripper close between steps 3 and 4.

    Args:
        start_config: Current joint configuration, length ≥ 7.
        grasp_pose: ``([x, y, z], [w, x, y, z])`` EE pose in robot-base frame.
        approach_axis: World-frame axis to approach along ('x', 'y', or 'z').
        approach_distance: Pre-grasp offset along approach_axis (positive =
            move *away* from the grasp along that axis to place pre-grasp).
        approach_in_tool_frame: If True, offset is in tool frame.
        lift_axis: Axis to move along after grasping.
        lift_distance: Distance to travel in lift phase (positive).
        lift_in_tool_frame: If True, offset is in tool frame.
        plan_approach: Whether to plan the approach (pre-grasp → grasp) segment.
        plan_lift: Whether to plan the lift (grasp → post-grasp) segment.
        robot_file: CuRobo robot YAML name.

    Returns:
        ``(success, approach_traj, grasp_traj, lift_traj, failure_reason)``
        where each trajectory is ``(T, 7)`` float32 or ``None`` if not planned.
    """
    if not _V2_AVAILABLE:
        return False, None, None, None, "plan_grasp_motion requires curobo v0.8"

    planner = _get_directed_planner(robot_file)
    planner.reset_seed()

    device_cfg = planner.config.device_cfg
    tool_frame = _FRANKA_TOOL_FRAME
    joint_names = planner.joint_names
    n_dof = len(joint_names)

    cfg = np.array(start_config, dtype=np.float32)
    if len(cfg) > n_dof:
        cfg = cfg[:n_dof]
    elif len(cfg) < n_dof:
        cfg = np.concatenate([cfg, np.zeros(n_dof - len(cfg), dtype=np.float32)])

    start_state = _V2JointState.from_numpy(
        joint_names=joint_names,
        position=np.expand_dims(cfg, 0),
        device_cfg=device_cfg,
    )

    # Build GoalToolPose from the single grasp pose.
    g_pos, g_quat = grasp_pose  # (xyz), (wxyz)
    g_pos_t = torch.tensor(np.array(g_pos, dtype=np.float32).reshape(1, 3), device=device_cfg.device)
    g_quat_t = torch.tensor(np.array(g_quat, dtype=np.float32).reshape(1, 4), device=device_cfg.device)
    grasp_v2pose = _V2Pose(position=g_pos_t, quaternion=g_quat_t, name=tool_frame)
    grasp_tool_poses = GoalToolPose.from_poses({tool_frame: grasp_v2pose})

    print(
        f"[plan_grasp_motion] v0.8 | grasp_pos={np.round(g_pos, 4).tolist()} "
        f"approach={approach_axis}±{approach_distance} lift={lift_axis}±{lift_distance}"
    )

    _t0 = time.monotonic()
    result = planner.plan_grasp(
        grasp_poses=grasp_tool_poses,
        current_state=start_state,
        grasp_approach_axis=approach_axis,
        grasp_approach_offset=approach_distance,   # positive = pre-grasp behind the grasp
        grasp_approach_in_tool_frame=approach_in_tool_frame,
        grasp_lift_axis=lift_axis,
        grasp_lift_offset=lift_distance,
        grasp_lift_in_tool_frame=lift_in_tool_frame,
        plan_approach_to_grasp=plan_approach,
        plan_grasp_to_lift=plan_lift,
    )
    _t_ms = (time.monotonic() - _t0) * 1000.0
    _log.info("[TIMING] v0.8 plan_grasp_motion plan_ms=%.1f t=%.3f", _t_ms, time.time())

    if result is None:
        return False, None, None, None, "plan_grasp returned None"

    success = bool(result.success is not None and torch.any(result.success).item())
    status_str = getattr(result, "status", None) or "N/A"
    print(f"[plan_grasp_motion] v0.8 success={success} status={status_str}")

    if not success:
        return False, None, None, None, f"plan_grasp_failed_{status_str}"

    approach_traj = _extract_traj_v2(result.approach_interpolated_trajectory)
    grasp_traj = _extract_traj_v2(result.grasp_interpolated_trajectory)
    lift_traj = _extract_traj_v2(result.lift_interpolated_trajectory)

    print(
        f"[plan_grasp_motion] approach={approach_traj.shape if approach_traj is not None else None} "
        f"grasp={grasp_traj.shape if grasp_traj is not None else None} "
        f"lift={lift_traj.shape if lift_traj is not None else None}"
    )
    return True, approach_traj, grasp_traj, lift_traj, ""


def plan_linear(
    start_pose: tuple[np.ndarray, np.ndarray],
    end_pose: tuple[np.ndarray, np.ndarray],
    start_joint_position: np.ndarray,
    **kwargs: Any,
) -> tuple[bool, np.ndarray | None, str]:
    """Legacy shim — delegates to plan_directed_linear."""
    return plan_directed_linear(
        start_config=start_joint_position,
        start_pose=start_pose,
        target_pose=end_pose,
        allowed_axes=["X", "Y", "Z"],
        orientation_mode="TARGET_AT_END",
    )


# ---------------------------------------------------------------------------
# Geometric IK + collision-aware planning (ported from curobo v0.7 wrapper).
# These still use the v0.7 IKSolver / MotionGen API via the caches above.
# ---------------------------------------------------------------------------
def _apply_tcp_offset_inverse(
    target_position: np.ndarray,
    target_quat_wxyz: np.ndarray,
    tcp_offset: np.ndarray | None,
) -> np.ndarray:
    """Convert a desired TCP world position to the EE-link world position.

    cuRobo's IK / MotionGen primitives reach the EE link, but callers
    typically want a target expressed at the tool tip (TCP). With the
    EE link rotated to ``target_quat_wxyz``, the TCP sits at
    ``ee_position + R(quat) @ tcp_offset`` in world frame; inverting gives
    ``ee_position = target_position − R(quat) @ tcp_offset``.

    ``tcp_offset`` is expressed in the EE link's local frame (e.g. for
    the Franka panda hand → fingertip pads it is ``(0, 0, 0.1029)``).
    Returns ``target_position`` unchanged when ``tcp_offset`` is None /
    all-zero so callers can pass through pure EE-link targets.
    """
    pos = np.asarray(target_position, dtype=np.float64).reshape(3)
    if tcp_offset is None:
        return pos.astype(np.float64)
    off = np.asarray(tcp_offset, dtype=np.float64).reshape(3)
    if not np.any(off):
        return pos.astype(np.float64)
    quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
    R = R_scipy.from_quat(np.roll(quat_wxyz, -1)).as_matrix()  # wxyz → xyzw
    return pos - R @ off


def solve_ik(
    target_position: np.ndarray,
    target_quat_wxyz: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    seed_config: np.ndarray | None = None,
    tcp_offset: np.ndarray | None = None,
    num_seeds: int = 32,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    tensor_args=None,
) -> tuple[bool, np.ndarray | None]:
    """Solve geometric IK for a single target TCP pose.

    target_position, target_quat_wxyz: desired TCP pose in world frame.
    tcp_offset: optional EE-link → TCP offset in EE link local frame. If given,
        we subtract the rotated offset from target_position so that FK(EE-link)
        + rotated-offset reaches the requested TCP.
    seed_config: optional warm start. When provided, the solution that
        minimizes joint-space distance to the seed is returned.
    Returns (success, joint_config (dof,)) or (False, None).

    NOT SUPPORTED on cuRobo v0.8: the v0.7 ``curobo.wrap.reacher.ik_solver``
    API (IKSolver / IKSolverConfig / Pose / TensorDeviceType) was removed in
    the v0.8.0 refactor.  This function is not on the active pan-grasp path;
    use ``plan_to_grasp_poses`` / ``plan_grasp_motion`` (v0.8 MotionPlanner)
    instead.  Raising here turns the otherwise-cryptic ``NameError`` (unbound
    v0.7 symbols) into an honest, actionable signal.
    """
    raise RuntimeError(
        "solve_ik is not supported on cuRobo v0.8; use PlanToGraspPoses / "
        "PlanGraspMotion"
    )


# ---------------------------------------------------------------------------
# v0.8 single-pose planner cache + helpers for plan_to_pose.
#
# plan_to_pose is the single-pose, non-grasp-specific subset of the proven
# v0.8 path used by plan_directed_linear / plan_grasp_motion (MotionPlanner +
# GoalToolPose + plan_pose).  It cannot reuse _get_directed_planner because
# that planner is built with self_collision_check=False and NO collision
# environment (scene_collision_checker is None), so update_world() would
# fail.  We build a separate, self-collision-aware planner here, and (only
# when a world is supplied) one with a pre-allocated mesh collision cache so
# MotionPlanner.update_world() can load the scene.
# ---------------------------------------------------------------------------
# Goalset capacity for the cached grasp planner (cuRoboV2 creation-time cap).
_GRASP_MAX_GOALSET = 32

_pose_planner_cache: dict[tuple, MotionPlanner] = {}


def _get_pose_planner(
    robot_file: str,
    *,
    with_collision: bool,
    mesh_cache: int,
    position_threshold: float,
    rotation_threshold: float,
    num_ik_seeds: int,
    use_cuda_graph: bool,
    max_goalset: int = 1,
) -> MotionPlanner:
    """Lazily create (and cache) a v0.8 MotionPlanner for single-pose planning.

    Mirrors ``_get_directed_planner`` (the proven v0.8 builder) but:

    * ``self_collision_check=True`` so the plan always respects self-collision
      (matches plan_to_pose's "still respects self-collisions" contract);
    * when ``with_collision`` is True, a ``collision_cache={"mesh": M}`` is
      passed so ``MotionPlannerCfg.create`` allocates a SceneCollisionCfg
      (scene_model=None) and the resulting planner exposes a
      ``scene_collision_checker`` that ``update_world(SceneCfg)`` can populate
      (confirmed: motion_planner_cfg.MotionPlannerCfg.create L139-146 and
      motion_planner.MotionPlanner.update_world L598-601).

    Cached by (robot, with_collision, mesh_cache, thresholds, seeds) so the
    no-collision path (the active pan-grasp path) and the collision path do
    not thrash one cache slot.
    """
    key = (
        robot_file,
        with_collision,
        int(mesh_cache) if with_collision else 0,
        round(float(position_threshold), 6),
        round(float(rotation_threshold), 6),
        int(num_ik_seeds),
        bool(use_cuda_graph),
        int(max_goalset),
    )
    cached = _pose_planner_cache.get(key)
    if cached is not None:
        print(f"[plan_to_pose] Reusing cached MotionPlanner key={key}")
        return cached

    print(
        f"[plan_to_pose] Creating MotionPlanner (v0.8, "
        f"collision={'mesh:%d' % mesh_cache if with_collision else 'self-only'}) "
        f"for {robot_file}"
    )
    device_cfg = DeviceCfg()
    create_kwargs = dict(
        robot=robot_file,
        self_collision_check=True,
        device_cfg=device_cfg,
        num_ik_seeds=num_ik_seeds,
        position_tolerance=position_threshold,
        orientation_tolerance=rotation_threshold,
        use_cuda_graph=use_cuda_graph,
    )
    if max_goalset > 1:
        # cuRoboV2 caps goalset size at planner creation (default 1); its own
        # goalset tests pass max_goalset explicitly (test_motion_planner.py).
        create_kwargs["max_goalset"] = int(max_goalset)
    if with_collision:
        # Pre-allocate a mesh cache so update_world() can later load the scene.
        create_kwargs["collision_cache"] = {"mesh": int(mesh_cache)}
    cfg = MotionPlannerCfg.create(**create_kwargs)
    planner = MotionPlanner(cfg)
    _pose_planner_cache[key] = planner
    print(
        f"[plan_to_pose] MotionPlanner ready (tool_frames={planner.tool_frames}, "
        f"scene_collision={'on' if with_collision else 'off'})"
    )
    return planner


def _world_to_v2_scene_cfg(world_config, device_cfg) -> Any | None:
    """Convert a (v0.7-style) WorldConfig of meshes into a v0.8 ``SceneCfg``.

    ``server.py`` builds the world via ``curobo.geom.types.WorldConfig`` whose
    ``.mesh`` entries carry ``name`` / ``pose`` ([x,y,z,qw,qx,qy,qz]) /
    ``vertices`` / ``faces``.  The v0.8 ``curobo._src.geom.types.Mesh``
    dataclass accepts exactly those fields (confirmed in
    third_party/curobo/curobo/_src/geom/types.py:449-485), so we map each
    entry onto a v0.8 ``Mesh`` and wrap them in a v0.8 ``SceneCfg``
    (``SceneCfg(mesh=[...])`` — confirmed in the same file L870-918 and used
    by the canonical test test_motion_planner.py:699-715).

    Returns the SceneCfg, or None if there is nothing to add.  Imported
    lazily / defensively (same pattern as the rest of this module) so the
    file still loads if the geom layout differs.
    """
    if world_config is None:
        return None
    meshes_in = list(getattr(world_config, "mesh", None) or [])
    if not meshes_in:
        return None
    from curobo._src.geom.types import Mesh as _V2Mesh
    from curobo._src.geom.types import SceneCfg as _V2SceneCfg

    v2_meshes = []
    for i, m in enumerate(meshes_in):
        name = getattr(m, "name", None) or f"mesh_{i}"
        pose = getattr(m, "pose", None)
        if pose is None:
            pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        else:
            pose = [float(v) for v in list(pose)]
        verts = getattr(m, "vertices", None)
        faces = getattr(m, "faces", None)
        if verts is None or faces is None:
            continue
        v2_meshes.append(
            _V2Mesh(
                name=str(name),
                pose=pose,
                vertices=np.asarray(verts, dtype=np.float32).reshape(-1, 3).tolist(),
                faces=np.asarray(faces, dtype=np.int32).reshape(-1, 3).tolist(),
                device_cfg=device_cfg,
            )
        )
    if not v2_meshes:
        return None
    return _V2SceneCfg(mesh=v2_meshes)


def plan_to_pose(
    target_position: np.ndarray,
    target_quat_wxyz: np.ndarray,
    start_joint_position: np.ndarray,
    *,
    robot_file: str = "franka.yml",
    tcp_offset: np.ndarray | None = None,
    world_config=None,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    max_attempts: int = 4,
    num_ik_seeds: int = 32,
    use_cuda_graph: bool = False,
    tensor_args=None,
) -> tuple[bool, np.ndarray | None]:
    """Plan a collision-aware trajectory to a single TCP pose (curobo v0.8).

    v0.8 implementation: this is the single-pose, non-grasp-specific subset of
    the proven v0.8 path used by ``plan_directed_linear`` /
    ``plan_grasp_motion`` (``MotionPlanner`` + ``GoalToolPose`` +
    ``MotionPlanner.plan_pose``).

    Steps (mirroring ``plan_directed_linear``):

    1. Build/get a cached v0.8 ``MotionPlanner`` via :func:`_get_pose_planner`
       (self-collision always on; mesh collision cache only when a world is
       supplied).
    2. When ``world_config`` is provided, convert it to a v0.8 ``SceneCfg``
       (:func:`_world_to_v2_scene_cfg`) and ``planner.update_world(...)`` it so
       the trajectory respects the scene geometry.  With no world this is a
       free-space plan that still respects self-collisions.
    3. Apply :func:`_apply_tcp_offset_inverse` to get the EE-link world target.
    4. Build a v0.8 start ``JointState`` (``_V2JointState.from_numpy``) and a
       single goal ``GoalToolPose`` (``_V2Pose`` + ``GoalToolPose.from_poses``)
       on the planner's tool frame — exactly as ``plan_directed_linear`` does.
    5. ``planner.plan_pose(...)`` and convert ``result.get_interpolated_plan()``
       to an ``(N_waypoints, dof)`` numpy array (same extraction as
       ``_extract_traj_v2`` / ``plan_directed_linear``).

    ``tensor_args`` is accepted for signature compatibility but is NOT used:
    the v0.7 ``TensorDeviceType`` no longer exists, so device/dtype come from
    the v0.8 ``DeviceCfg`` owned by the planner (same as every other v0.8
    function in this module).

    Returns ``(success, trajectory)`` where ``trajectory`` is a numpy array of
    shape ``[N_waypoints, dof]`` (the server wraps it via
    ``_trajectory_to_proto``), or ``(False, None)`` on failure.
    """
    if not _V2_AVAILABLE:
        raise RuntimeError(
            "plan_to_pose: curobo v0.8 (cuRoboV2) is required.  The v0.7 "
            "MotionGen/IKSolver API was removed in the v0.8.0 refactor.  "
            "Re-install curobo v0.8.0 and run `uv sync`."
        )

    dof = _get_robot_dof(robot_file)

    # EE-link world target (cuRobo reaches the tool frame, not the TCP).
    ee_position = _apply_tcp_offset_inverse(
        target_position, target_quat_wxyz, tcp_offset
    )
    ee_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float32).reshape(4)

    has_world = (
        world_config is not None
        and len(list(getattr(world_config, "mesh", None) or [])) > 0
    )
    n_mesh = (
        len(list(getattr(world_config, "mesh", None) or [])) if has_world else 0
    )

    planner = _get_pose_planner(
        robot_file,
        with_collision=has_world,
        # Generous floor so update_world() never overflows the mesh cache and
        # the same planner survives scenes with slightly different mesh counts.
        mesh_cache=max(n_mesh + 4, 32),
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
        num_ik_seeds=num_ik_seeds,
        use_cuda_graph=use_cuda_graph,
    )
    planner.reset_seed()

    device_cfg = planner.config.device_cfg
    joint_names = planner.joint_names
    n_dof = len(joint_names)

    # Load the collision scene (v0.8 SceneCfg) if a world was supplied.
    if has_world:
        try:
            scene_cfg = _world_to_v2_scene_cfg(world_config, device_cfg)
            if scene_cfg is not None:
                planner.clear_scene_cache()
                planner.update_world(scene_cfg)
                print(
                    f"[plan_to_pose] Loaded v0.8 collision scene "
                    f"(meshes={n_mesh})."
                )
        except Exception as e:
            # A scene-load failure must not silently downgrade to a
            # collision-free plan; surface it as an actionable error.
            raise RuntimeError(
                f"plan_to_pose: failed to load collision world into v0.8 "
                f"MotionPlanner: {e}"
            ) from e

    # ── Start JointState (v0.8) — shape [1, DOF], mirrors plan_directed_linear
    cfg = np.asarray(start_joint_position, dtype=np.float32).flatten()
    if len(cfg) > n_dof:
        cfg = cfg[:n_dof]
    elif len(cfg) < n_dof:
        cfg = np.concatenate([cfg, np.zeros(n_dof - len(cfg), dtype=np.float32)])
    start_state = _V2JointState.from_numpy(
        joint_names=joint_names,
        position=np.expand_dims(cfg, 0),
        device_cfg=device_cfg,
    )

    # ── Single goal pose on the planner's tool frame (mirrors
    #    plan_directed_linear's GoalToolPose construction).
    tool_frame = _FRANKA_TOOL_FRAME
    if planner.tool_frames:
        tool_frame = planner.tool_frames[0]

    goal_pos_t = device_cfg.to_device(
        torch.tensor(
            np.asarray(ee_position, dtype=np.float32).reshape(3),
            dtype=torch.float32,
        ).unsqueeze(0)
    )
    goal_quat_t = device_cfg.to_device(
        torch.tensor(ee_quat_wxyz, dtype=torch.float32).unsqueeze(0)
    )
    goal_v2pose = _V2Pose(
        position=goal_pos_t, quaternion=goal_quat_t, name=tool_frame
    )
    goal_tool_poses = GoalToolPose.from_poses({tool_frame: goal_v2pose})

    print(
        f"[plan_to_pose] v0.8 | tool={tool_frame} "
        f"ee_pos={np.round(ee_position, 4).tolist()} "
        f"ee_quat_wxyz={np.round(ee_quat_wxyz, 4).tolist()} "
        f"world={'on(%d mesh)' % n_mesh if has_world else 'off'} "
        f"max_attempts={max_attempts}"
    )

    _t_plan_start = time.monotonic()
    result = planner.plan_pose(
        goal_tool_poses, start_state, max_attempts=max_attempts
    )
    _t_plan_ms = (time.monotonic() - _t_plan_start) * 1000.0
    _log.info(
        "[TIMING] v0.8 plan_to_pose world=%s plan_ms=%.1f t=%.3f",
        has_world, _t_plan_ms, time.time(),
    )

    if result is None:
        print("[plan_to_pose] v0.8 planner returned None.")
        return False, None

    success = bool(
        result.success is not None and torch.any(result.success).item()
    )
    status_str = (
        getattr(result, "status", None)
        or getattr(result, "failure_reason", None)
        or "N/A"
    )
    print(f"[plan_to_pose] v0.8 success={success} status={status_str}")
    if not success:
        return False, None

    # Trajectory extraction: identical to _extract_traj_v2 /
    # plan_directed_linear (result.get_interpolated_plan() -> JointState).
    try:
        traj_js = result.get_interpolated_plan()
    except Exception as e:
        print(f"[plan_to_pose] trajectory extraction failed: {e}")
        return False, None
    if traj_js is None or traj_js.position is None:
        print("[plan_to_pose] no interpolated plan on a successful result.")
        return False, None

    arr = np.squeeze(traj_js.position.detach().cpu().numpy())
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[:, :dof].astype(np.float64)
    print(f"[plan_to_pose] v0.8 trajectory shape: {arr.shape}")
    return True, arr


# ---------------------------------------------------------------------------
# Batch grasp feasibility (collision-aware IK + corridor sweep)
# ---------------------------------------------------------------------------
class _CollisionAwareIKCache:
    """Single cached IKSolver + RobotWorld pair for batch feasibility queries.

    Both consume the same WorldConfig per request via update_world() so we
    don't pay the kernel-build cost on every call. use_cuda_graph is left
    False on the IKSolver because the cache may see worlds with different
    obstacle counts across requests, and changing obstacle count breaks
    captured graphs (per IKSolver.update_world docstring).
    """

    def __init__(self) -> None:
        self._ik: IKSolver | None = None
        self._world: RobotWorld | None = None
        self._key: tuple | None = None

    def get(
        self,
        world_config,
        robot_file: str,
        *,
        num_seeds: int,
        position_threshold: float,
        rotation_threshold: float,
        collision_activation_distance: float,
        robot_collision_sphere_buffer: float | None,
        tensor_args,
    ) -> tuple[IKSolver, RobotWorld]:
        key = (
            robot_file,
            num_seeds,
            position_threshold,
            rotation_threshold,
            collision_activation_distance,
            robot_collision_sphere_buffer,
        )
        if self._ik is not None and self._key == key:
            try:
                self._ik.update_world(world_config)
                self._world.update_world(world_config)
                return self._ik, self._world
            except Exception as e:
                print(f"[CollisionAwareIKCache] update_world failed ({e}); rebuilding")
                self.invalidate()

        robot_dict = _load_robot_dict(robot_file)
        if robot_collision_sphere_buffer is not None:
            inner = robot_dict.get("robot_cfg", robot_dict)
            if "kinematics" in inner:
                inner["kinematics"] = {
                    **inner["kinematics"],
                    "collision_sphere_buffer": robot_collision_sphere_buffer,
                }
        robot_cfg = RobotConfig.from_dict(robot_dict, tensor_args)

        ik_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_config,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            use_cuda_graph=False,
            collision_activation_distance=collision_activation_distance,
        )
        ik = IKSolver(ik_cfg)

        rw_cfg = RobotWorldConfig.load_from_config(
            robot_cfg,
            world_config,
            collision_activation_distance=collision_activation_distance,
            tensor_args=tensor_args,
        )
        rw = RobotWorld(rw_cfg)

        self._ik = ik
        self._world = rw
        self._key = key
        print(
            f"[CollisionAwareIKCache] Built IKSolver+RobotWorld "
            f"(robot={robot_file}, seeds={num_seeds}, world_meshes={len(world_config.mesh)})"
        )
        return ik, rw

    def invalidate(self) -> None:
        self._ik = None
        self._world = None
        self._key = None


_collision_aware_ik_cache = _CollisionAwareIKCache()


def batch_grasp_feasibility(
    world_config,
    start_state: np.ndarray,
    grasp_poses: list[tuple[np.ndarray, np.ndarray]],
    *,
    grasp_pose_is_fingertip: bool = True,
    approach_offset_m: float = 0.10,
    num_corridor_samples: int = 5,
    robot_file: str = "franka.yml",
    num_ik_seeds: int = 32,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
    collision_activation_distance: float = 0.01,
    robot_collision_sphere_buffer: float | None = None,
    ignore_obstacle_names: list[str] | None = None,
    tensor_args=None,
) -> tuple[list[bool], list[bool], list[bool], list[float]]:
    """Per-pose scene-collision feasibility for a batch of grasp candidates.

    Returns ``(feasible, grasp_ik_ok, approach_ik_ok, corridor_collision_fraction)``
    each of length ``len(grasp_poses)``, aligned with input order.

    feasible[i] is True iff IK solves at both the grasp and the pre-grasp
    approach pose (offset by ``approach_offset_m`` along the grasp's local
    -Z) WITHOUT world or self collision, AND a joint-space linear interp
    between the two IK solutions has zero in-collision waypoints.

    See ``proto/curobo/v1/curobo.proto`` BatchGraspFeasibility for the
    semantic contract.
    """
    raise RuntimeError(
        "batch_grasp_feasibility is not supported on cuRobo v0.8; use "
        "PlanToGraspPoses / PlanGraspMotion"
    )


if __name__ == "__main__":
    print("_curobo_impl is a library; call it through the curobo tool bundle.")
    raise SystemExit(0)
