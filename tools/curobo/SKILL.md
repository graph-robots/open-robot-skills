---
name: curobo
description: NVIDIA cuRobo motion planning — collision-free trajectories to
  grasp goalsets, transport with an attached object, constrained linear
  moves, single-pose planning, geometric IK, batch grasp feasibility, and
  joint-trajectory collision validation. Use when a workflow needs
  GPU-accelerated, collision-aware arm motion plans.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: planning, tags: [planning, motion, gpu, cuda]}
gap:
  tools:
    - curobo.plan_to_grasp_poses: Collision-free trajectory to one of several grasp poses (goalset).
    - curobo.plan_with_grasped_object: Transport plan with the grasped object attached to the gripper.
    - curobo.plan_linear: Straight Cartesian trajectory between two EE poses.
    - curobo.plan_directed_linear: Constrained linear motion with axis/orientation holds.
    - curobo.plan_grasp_motion: Approach + grasp + lift sequence (three trajectories).
    - curobo.plan_to_pose: Collision-aware plan to a single TCP pose.
    - curobo.solve_ik: Geometric IK for a single TCP pose (no world collision).
    - curobo.batch_grasp_feasibility: Per-pose grasp/approach IK + corridor feasibility for a grasp batch.
    - curobo.validate_joint_trajectory_robot: Collision-validate joint waypoints (robot vs world + self).
    - curobo.validate_joint_trajectory_grasped: Same, with a grasped object attached at waypoint 0.
---

# curobo

Collision-aware cuRobo motion planning as in-process tools. Trajectories
in/out are gap `Trajectory` dicts (`waypoints: [{positions:
float64[dof]}]`); worlds are gap `WorldConfig` dicts (build one with
`geometry.build_world_config`).

## When to use

- Tabletop pick: `geometry.top_down_grasp_candidates` →
  `curobo.plan_to_grasp_poses` (pass the whole fan as the goalset; check
  `goalset_index` for which one was reached).
- Transport after grasping: `curobo.plan_with_grasped_object` with the
  object's mesh name from `build_world_config`.
- Drawers/doors: `curobo.plan_grasp_motion` (approach → grasp → lift/pull
  with gripper commands interleaved), or `curobo.plan_directed_linear` for
  a pull along one axis with orientation locked.

## Install

cuRobo JIT-compiles CUDA extensions at install time — build isolation must
be off and `CUDA_HOME` must point at a toolkit matching your torch build:

```bash
export CUDA_HOME=/usr/local/cuda     # toolkit matching torch's CUDA version
uv sync --extra curobo               # (pip: pip install -e ".[curobo]" --no-build-isolation)
```

If the import fails at tool-call time the tools raise a `ToolError` with
this recipe. First planner call per process pays JIT/warmup latency; the
MotionGen/planner instances are cached and reused (HyRL pattern — recreating
them per call corrupts CUDA graph state).

## Gotchas (carried over from the service + curobo_api)

- **Frames**: grasp/target poses are in the robot-base frame (cuRobo treats
  the robot base as world origin). With `grasp_pose_is_fingertip=True`
  (default) grasp positions are fingertip-pad centers and converted to the
  `panda_hand` frame solver-side (offset 0.1029 m along hand Z).
- **Ignore the grasp target**: pass its mesh name in
  `ignore_obstacle_names` for `plan_to_grasp_poses` /
  `batch_grasp_feasibility` — closing on the target is not a collision.
- `robot_collision_sphere_buffer` default −0.01 shrinks robot collision
  spheres 1 cm; reduces IK_FAIL against dense perception meshes. Negative
  is intentional.
- GPU access is serialised by a module lock (cuRobo is not thread-safe);
  CUDA/"graph capture" errors invalidate the cached planners automatically
  before raising `PlanningFailed`.
- `use_cuda_graph` must stay False for the validators
  (`check_start_state` requirement) and for varying world/start setups.
- **curobo version split**: `plan_to_grasp_poses`, `plan_grasp_motion`,
  `plan_directed_linear`, `plan_linear`, `plan_to_pose`,
  `plan_with_grasped_object` target curobo v0.8 (MotionPlanner API);
  `solve_ik` and `batch_grasp_feasibility` are built on the v0.7
  IKSolver API that v0.8 removed — on a v0.8-only install they raise
  `PlanningFailed` ("not supported on cuRobo v0.8"); plan via the goalset
  tools instead. The validators use the v0.7 MotionGen path too.
- `validate_joint_trajectory_grasped` always invalidates the planner cache
  afterward so the attachment cannot leak; expect the next planning call to
  re-create the planner.
- Planning failures return `success=False` (with `failure_reason` where the
  RPC had one); infrastructure errors raise `PlanningFailed` / `ToolError`.
- Set `debug_out_dir` on the grasp/transport planners to dump world + robot
  sphere OBJ/PLY debug artifacts on failure (default `./curobo_debug*`).
