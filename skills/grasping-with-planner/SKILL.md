---
name: grasping-with-planner
description: Collision-aware grasping using cuRobo trajectory planning over a
  per-observation collision world. The planner picks a collision-free
  trajectory to one of several candidate top-down grasp poses, the arm
  executes it, and the gripper closes. Use when the curobo tool bundle is
  installed and a perceived object (OBB + mask) must be grasped — the default
  grasping skill whenever cuRobo is deployed, especially in cluttered scenes.
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, manipulation, collision-aware, curobo]}
gap:
  allowed_tools:
    - geometry.top_down_grasp_candidates
    - geometry.build_world_config
    - robot.get_observation
    - curobo.plan_to_grasp_poses
    - robot.execute_trajectory
    - robot.open_gripper
    - robot.close_gripper
  exit_conditions:
    grasped: Object held in the gripper after `close`.
    failed: Any failure during the grasp attempt — collision-aware planning failure or trajectory execution error (a raise propagating to `on_error`). Coordinator routes to abort.
  required_inputs:
    target_obb: OrientedBoundingBox
    target_mask: Mask
  produces_outputs:
    ee_pose_at_grasp: Se3Pose
    grasp_pose: Se3Pose
  hard_rules:
    - >
      ALWAYS begin with `robot.open_gripper` (settle_steps=40). Skipping this
      is the #1 silent grasp-failure mode when the gripper starts closed from
      a previous task step.
    - >
      When building the collision world, ALWAYS pass `target_mask = Ref("in.target_mask")` AND `target_obb = Ref("in.target_obb")` AND a
      `target_name`. The mask is the authoritative target silhouette
      (pixel-accurate); the OBB is a safety fallback. The same `target_name`
      must be passed to `plan_grasp` so the planner knows which mesh to
      exclude from self-collision.
    - >
      Use `geometry.top_down_grasp_candidates` (returns
      `candidates: {poses: list[Se3Pose]}`), not
      `geometry.top_down_grasp_from_obb` (single bare pose). The planner
      needs a goalset to pick a reachable candidate from.
    - >
      ALWAYS include an `approach` state between `compute_grasp` and `observe`.
      `plan_grasp` disables CuRobo's `use_grasp_approach` flag on the
      assumption the gripper is already oriented over the target — skipping
      `approach` causes top-down grasps to descend with the wrong wrist
      rotation and slip. See references/design_grasp_curobo.md.
    - >
      Do NOT add a `verify_gripper_grasp` node — there is no such skill;
      postcondition verification is a `validate=True` checkpoint. Express "the gripper is actually holding the target after close" as a
      `validate=True` checkpoint (`target_held`). See `## Checkpoints` below.
  canonical_scripts:
    - approach_above: scripts/approach_above.py
    - build_world: scripts/build_world.py
    - plan_grasp: scripts/plan_grasp.py
    - compute_align_pose: scripts/compute_align_pose.py
    - select_short_axis: scripts/select_short_axis.py
  references:
    - title: Why disable use_grasp_approach?
      path: references/design_grasp_curobo.md
    - title: Gripper settle-step constants
      path: references/gripper_settle_constants.md
  streaming: false
---

# grasping-with-planner

Collision-aware grasping with CuRobo. The subgraph builds a per-observation
collision world (with the target excluded), asks CuRobo to plan a
trajectory to one of several candidate top-down grasps, executes the plan,
and closes the gripper.

## Install

This skill depends on the **curobo tool bundle** (`curobo.plan_to_grasp_poses`).
cuRobo JIT-compiles CUDA extensions at install time — build isolation must be
off and `CUDA_HOME` must point at a toolkit matching your torch build:

```bash
export CUDA_HOME=/usr/local/cuda
uv sync --extra curobo   # (pip: pip install -e "open-robot-skills[curobo]" --no-build-isolation)
```

See `tools/curobo/SKILL.md` for the full recipe and gotchas.

## When to use

- Default grasping skill whenever `curobo` is deployed.
- Cluttered scenes where the arm must thread between obstacles.

## When NOT to use

- `curobo` not deployed. Use `grasping-direct-ik` instead.
- Scene is uncluttered AND CuRobo is not desired (faster, less safe).
- The graspable region is NOT the OBB centroid — bowl rim, mug / moka-pot
  / frying-pan handle, or any off-center grasp. The OBB top-down
  candidates emitted by `geometry.top_down_grasp_candidates` are
  centered on the OBB XY, so they slip on hollow centers and miss
  handles. Use `grasping-short-axis` for elongated handles, where the
  centroid is graspable but orientation is what matters.

## Recommended subgraph state flow

8 states, in order:

```text
open → compute_grasp → approach → observe → build_world → plan → execute → close → grasped
```

(`grasped` is the success-marker `noop` from `sg.add_exit("grasped")`,
with an edge to `END`.)

State details:

1. **`open`** — `type: tool`, `tool: "robot.open_gripper"`, `inputs: { settle_steps: 40 }`.
2. **`compute_grasp`** — `type: tool`, `tool: "geometry.top_down_grasp_candidates"`,
   `inputs: { obb: Ref("in.target_obb") }`. Returns
   `candidates: {poses: list[Se3Pose]}`.
3. **`approach`** — `type: script`, file `scripts/<sg>/approach_above.py`
   from this bundle's canonical_scripts. Inputs:
   `target_position = Ref("compute_grasp.candidates.poses.0.position")`,
   `rotation = Ref("compute_grasp.candidates.poses.0.rotation")`,
   `target_obb = Ref("in.target_obb")`. Pre-rotates the gripper
   into the chosen grasp orientation at a safe height directly above
   the target. CuRobo's `plan` step then only has to descend — this is
   what justifies `use_grasp_approach=False` in `plan_grasp.py`.
   Skipping this state causes top-down grasps to descend with the wrong
   wrist rotation (slip every time).
4. **`observe`** — `type: tool`, `tool: "robot.get_observation"`,
   `inputs: {}`. Captures cameras + arm state for the collision world.
5. **`build_world`** — `type: script`, file `scripts/<sg>/build_world.py`
   from this bundle's canonical_scripts. Inputs:
   `observation = Ref("observe")`,
   `target_mask = Ref("in.target_mask")`,
   `target_obb = Ref("in.target_obb")`, `target_name = "target"`.
6. **`plan`** — `type: script`, file `scripts/<sg>/plan_grasp.py`. Inputs:
   `world_config = Ref("build_world.config")`,
   `observation = Ref("observe")`,
   `grasp_poses = Ref("compute_grasp.candidates.poses")`,
   `target_name = "target"` (must match step 5). CuRobo plans a
   collision-aware descent from the post-`approach` pose to the chosen
   grasp pose; `use_grasp_approach=False` because `approach` already
   pre-rotated the gripper above the target.
7. **`execute`** — `type: tool`, `tool: "robot.execute_trajectory"`,
   `inputs: { trajectory: Ref("plan.trajectory") }`.
8. **`close`** — `type: tool`, `tool: "robot.close_gripper"`, `inputs: { settle_steps: 60 }`.
   Edge directly from `close` to the `grasped` success marker; the
   subgraph's `on_error: "failed"` catches any raise from earlier steps.
   Whether the gripper actually closed on the object is checked by the
   `target_held` postcondition checkpoint (see `## Checkpoints`), NOT by
   a re-check-and-raise node (none such exists).

   The subgraph publishes **two** cross-subgraph outputs:

   - `ee_pose_at_grasp` — the live TCP pose captured at the `observe`
     step (which sits between `approach` and `plan`). Downstream
     `transporting-objects` uses it to compute a drop height that accounts
     for the panda hand-to-tcp offset and the held object's geometry.
   - `grasp_pose` — the **computed** grasp pose emitted by
     `compute_grasp`, i.e. what the planner is *targeting*. Distinct
     from `ee_pose_at_grasp` (which is the *current* EE pose at the
     time of observe, before plan + execute). Exposing `grasp_pose`
     lets the checkpoint author write an output-anchored verifier like
     `predicate=lambda w, o: o["grasp_pose"]["position"]["z"] > 0.01` to
     catch sub-table grasp poses *before* they cascade into a
     `target_held=False` failure.

   **Hard rule on the output binding:** the `robot.get_observation`
   response is an `Observation { cameras: list[CameraFrame];
   arms: list[ArmState] }`. There is **no** flat `ee_pose` field
   on the response — the EE pose lives at `arms[0].ee_pose`. The
   binding must therefore be exactly:

   ```python
   sg.set_outputs(
       ee_pose_at_grasp=Ref("observe.arms.0.ee_pose"),
       grasp_pose=Ref("compute_grasp.candidates.poses.0"),
   )
   ```

   Do **not** write `Ref("observe.ee_pose")` — that path does not
   exist on the response and the cross-subgraph binding will silently
   resolve to None, sending the downstream `compute_drop_pose.py` into
   its inferior no-`ee_pose_at_grasp` fallback (drop height too high,
   placement misses).

   Cross-subgraph data flow is by name, so any downstream subgraph
   that declares `ee_pose_at_grasp` as an input automatically receives
   the value bound here.

   ```json
   "edges": [ ..., ["close", "grasped"], ["grasped", "END"] ],
   "conditional_edges": {},
   "exit": { "router_field": null, "success_values": ["grasped"] },
   "on_error": "failed"
   ```

   The lift onto a safe carry height is handled by the next
   `transporting-objects` subgraph (its `waypoint_move` script lifts before
   lateral motion); do NOT add a lift step here.

## Optional candidate-reordering state

For a thin/elongated target (frypan handle, screwdriver, spoon, rod) insert
ONE `type: script` state `select_short_axis`
(`scripts/<sg>/select_short_axis.py`) between `compute_grasp` and
`approach`. Inputs: `target_obb = Ref("in.target_obb")`,
`candidate_poses = Ref("compute_grasp.candidates.poses")`. It reorders the
candidate fan so poses whose finger-opening axis aligns with the OBB's
short horizontal axis come first (count and pose values preserved — only
the ordering changes), then wire `approach` and `plan` against
`Ref("select_short_axis.poses...")` instead. For a deterministic,
geometry-locked single short-axis pose use the dedicated
`grasping-short-axis` skill instead.

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Gripper has closed on the object after the descend. Route to next subgraph (typically `transporting-objects`). |
| `failed` | Any grasp-attempt failure: collision-aware planning failure or trajectory execution error (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |


## See also

- `references/design_grasp_curobo.md` — why disable `use_grasp_approach`.
- `references/gripper_settle_constants.md` — settle-step tunings.
- `scripts/{approach_above,build_world,plan_grasp,compute_align_pose,select_short_axis}.py` — canonical scripts.
