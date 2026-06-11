---
name: grasping-short-axis
description: Deterministic short-axis-aligned grasp with CuRobo. The grasp
  pose is computed geometrically (NOT sampled/scored) — the gripper
  approaches straight down with its finger-opening axis locked to the OBB's
  SHORTER horizontal axis, so the jaws close ACROSS the narrow dimension of
  an elongated target. Use for pan/pot handles, bottles, tools, utensils —
  anywhere the OBB centroid is graspable but orientation is the thing that
  matters. An optional offset_from_base node slides the grasp outward along
  the handle, clear of the heavier object body, when a second OBB of that
  body is wired in. A scored/sampled pose often holds at close but shears
  out under the lift; the short-axis pose grips the bar squarely so it
  survives lift + transport.
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, manipulation, collision-aware, curobo, short-axis, handle]}
gap:
  allowed_tools:
    - geometry.build_world_config
    - robot.get_observation
    - curobo.plan_to_grasp_poses
    - curobo.plan_to_pose
    - curobo.plan_grasp_motion
    - curobo.plan_directed_linear
    - robot.execute_trajectory
    - robot.move_to_joints
    - robot.get_ee_pose
    - robot.go_to_pose
    - robot.open_gripper
    - robot.close_gripper
  exit_conditions:
    grasped: Object held in the gripper after `close`.
    failed: Any failure during the grasp attempt — collision-aware planning failure or trajectory execution error (a raise propagating to `on_error`). Coordinator routes to abort.
  required_inputs:
    # MANDATORY — the coordinator MUST declare both in `grasp_sg.inputs`
    # and wire them from the upstream handle-perception subgraph.
    # OPTIONAL extra input (NOT listed here so graph-validation does not
    # force it): `base_obb: OrientedBoundingBox`. See the `offset_from_base`
    # hard_rule — declare + wire it ONLY when you also authored a second
    # perception subgraph for the heavier object body (pan bowl / pot
    # belly). Omitting it is valid; the grasp degrades to a plain
    # short-axis grasp (still correct, just not slid off the body).
    target_obb: OrientedBoundingBox
    target_mask: Mask
  produces_outputs:
    ee_pose_at_grasp: Se3Pose
    grasp_pose: Se3Pose
  hard_rules:
    - >
      ALWAYS begin with `robot.open_gripper` (settle_steps=40). Skipping
      this is the #1 silent grasp-failure mode when the gripper starts
      closed from a previous task step.
    - >
      Use the canonical `compute_grasp` (`scripts/<sg>/short_axis_grasp_pose.py`)
      — it returns ONE deterministic Se3Pose aligned to the OBB's shorter
      horizontal axis. Do NOT swap in `geometry.top_down_grasp_candidates`
      or any sampled/scored generator: the whole point of this skill is
      that the orientation is geometry-locked, not discriminator-ranked.
      `plan_grasp` auto-wraps the single pose into a one-element goalset.
    - >
      `offset_from_base` (`scripts/<sg>/offset_grasp_from_base.py`) is the
      robustness lever for a handle attached to a heavier body (frying pan,
      pot). It slides the grasp outward along the handle, clear of the
      body, so the jaws grip the bar — not the rim — and the grip survives
      the lift. To enable it: (1) author a SECOND perception subgraph that
      segments the object BODY (e.g. perceiving-object-parts /
      perceiving-objects with object/subpart name like "the round base of
      the frying pan"),
      (2) add `base_obb: OrientedBoundingBox` to THIS subgraph's `inputs`
      and wire it from that body perception, (3) bind
      `base_obb = Ref("in.base_obb")` on the `offset_from_base` node.
      If you do NOT author a body perception, KEEP the `offset_from_base`
      state but bind only `handle_obb` + `grasp_pose` (omit `base_obb`);
      the script no-ops and returns the grasp unchanged. NEVER write
      `Ref("in.base_obb")` unless `base_obb` is a declared subgraph input —
      that is a graph-validation error.
    - >
      When building the collision world, ALWAYS pass
      `target_mask = Ref("in.target_mask")` AND
      `target_obb = Ref("in.target_obb")` AND a `target_name`. The mask is
      the authoritative target silhouette (pixel-accurate); the OBB is a
      safety fallback. The same `target_name` MUST be passed to `plan`
      (`plan_grasp.py`) so the planner excludes that mesh from collision.
    - >
      ALWAYS include the `approach` state between `offset_from_base` and
      `observe`. `plan_grasp` disables CuRobo's `use_grasp_approach` flag on
      the assumption the gripper is already oriented over the target —
      skipping `approach` makes the descent start from a wrong wrist
      rotation and slip. `approach_above.py` lifts to a safe height,
      translates above the target, then rotates in place to the grasp
      orientation via CuRobo `curobo.plan_to_pose` (NOT the connector's
      basic IK — it cannot flip the Franka wrist and silently no-ops the
      rotate, leaving a horizontal EE that descends sideways).
    - >
      ALWAYS insert a `finalize` state (`scripts/<sg>/finalize_trajectory.py`,
      input `trajectory = Ref("plan.trajectory")`) BETWEEN `execute` and
      `close`. Edges: `execute → finalize → close`. This is mandatory and
      is the #1 silent grasp-failure cause: `execute` runs
      `robot.execute_trajectory` with `subsample=4`, so the body PD is
      still actively driving toward the last waypoint when control
      returns. If `robot.close_gripper` fires then, the simulator's
      action-manager tick silently DROPS the binary close action — the
      fingers never close, nothing is grasped, and `target_held` fails
      with no error (the planned-pose checkpoints still pass, masking the
      cause). `finalize` issues one `robot.move_to_joints` to the
      trajectory's final waypoint with full convergence so the PD is
      quiescent before `close`. Never wire `execute → close` directly.
    - >
      Do NOT add a `verify_gripper_grasp` node — there is no such skill.
      Express "the gripper is actually holding the target after close" as a
      `validate=True` checkpoint (`target_held`). See `## Checkpoints`.
    - >
      OPTIONAL escalation lever — `plan_grasp_motion`
      (`scripts/<sg>/plan_grasp_motion.py`, cuRobo v0.8
      `curobo.plan_grasp_motion`). NOT part of the default 10-state flow
      and MUST NOT always be used. Consider composing it ONLY as an
      escalation when the standard `plan` (`plan_grasp.py` /
      `curobo.plan_to_grasp_poses`) raises "0/N feasible" / "not
      reachable" on a FAR / LOW / OCCLUDED handle (the `failed` route
      fires from `plan`, not from `compute_grasp` — the geometry-locked
      short-axis pose itself is correct). `curobo.plan_grasp_motion`
      decomposes the grasp into THREE separate trajectories — a
      free-space `approach_trajectory` to a pre-grasp offset, a
      constrained `grasp_trajectory` into the grasp pose, and a
      constrained `lift_trajectory` away from it — which can be feasible
      where the single all-in-one IK to the grasp pose is not. When you
      DO use it, replace the `plan → execute → finalize → close` tail
      with: execute `approach_trajectory`, execute `grasp_trajectory`,
      `finalize` its last waypoint, `robot.close_gripper`, THEN execute
      `lift_trajectory` — the gripper close is interleaved BETWEEN the
      grasp leg and the lift leg (that interleave is the whole reason the
      legs are returned separately). Inputs: `observation = Ref("observe")`
      and a single `grasp_pose = Ref("offset_from_base.adjusted_grasp")`
      (the script takes one pose, not a goalset). Discoverable, not
      mandatory — the standard `plan` remains the default path.
  canonical_scripts:
    - compute_grasp: scripts/short_axis_grasp_pose.py
    - offset_from_base: scripts/offset_grasp_from_base.py
    - approach_above: scripts/approach_above.py
    - build_world: scripts/build_world.py
    - plan_grasp: scripts/plan_grasp.py
    - plan_grasp_motion: scripts/plan_grasp_motion.py
    - finalize_trajectory: scripts/finalize_trajectory.py
  references:
    - title: Why disable use_grasp_approach
      path: references/design_grasp_curobo.md
    - title: Gripper settle-step constants
      path: references/gripper_settle_constants.md
  streaming: false
---

# grasping-short-axis

Deterministic, geometry-locked grasping with CuRobo. The grasp pose is
computed directly from the target OBB — the gripper descends along world
−Z with its finger-opening axis snapped to the OBB's *shorter* horizontal
axis, so the jaws close across the narrow dimension of the bar. An
optional node then slides the grasp outward along the handle, clear of a
heavier attached body. The subgraph builds a per-observation collision
world (target excluded), plans a CuRobo trajectory, executes it,
finalizes the last waypoint, closes the gripper, and outputs the EE pose
at grasp.

## Install

This skill depends on the **curobo** and **geometry** tool bundles:

```bash
export CUDA_HOME=/usr/local/cuda
uv sync --extra curobo --extra geometry   # (pip: pip install -e "open-robot-skills[curobo,geometry]" --no-build-isolation)
```

## When to use

- Elongated targets where grasp orientation matters: pan / pot handles,
  bottles, tools, utensils.
- A subpart (handle) that protrudes from a heavier body — wire `base_obb`
  for the outward slide.
- When a sampled/scored grasp pose holds at close but the object slips
  during the lift (marginal off-axis contact patch).

## When NOT to use

- Symmetric objects (boxes, cans) where any yaw works — use
  `grasping-with-planner`.
- `curobo` not deployed.
- You genuinely want multiple sampled candidates for the planner to
  choose from — use `grasping-with-planner`.

## Recommended subgraph state flow

10 states, in order:

```text
open → compute_grasp → offset_from_base → approach → observe
     → build_world → plan → execute → finalize → close → grasped
```

(`grasped` is the success-marker `noop` from `sg.add_exit("grasped")`,
with an edge to `END`.)

State details:

1. **`open`** — `type: tool`, `tool: "robot.open_gripper"`,
   `inputs: { settle_steps: 40 }`.
2. **`compute_grasp`** — `type: script`, file
   `scripts/<sg>/short_axis_grasp_pose.py` (canonical — do NOT re-emit a
   ```` ```python ```` block; the bundle materializes it). Inputs:
   `target_obb = Ref("in.target_obb")`. Optional `z_offset` (default
   −0.04: descend the fingertip 4 cm into the OBB top). Returns
   `{grasp_pose: Se3Pose}`.
3. **`offset_from_base`** — `type: script`, file
   `scripts/<sg>/offset_grasp_from_base.py` (canonical). Inputs:
   `handle_obb = Ref("in.target_obb")`,
   `grasp_pose = Ref("compute_grasp.grasp_pose")`, and — ONLY when a
   body perception was authored and `base_obb` declared as a subgraph
   input — `base_obb = Ref("in.base_obb")`. Returns
   `{adjusted_grasp: Se3Pose}`. Safe no-op when `base_obb` is absent.
4. **`approach`** — `type: script`, file
   `scripts/<sg>/approach_above.py` (canonical). Inputs:
   `target_position = Ref("offset_from_base.adjusted_grasp.position")`,
   `rotation = Ref("offset_from_base.adjusted_grasp.rotation")`,
   `target_obb = Ref("in.target_obb")`.
5. **`observe`** — `type: tool`, `tool: "robot.get_observation"`.
6. **`build_world`** — `type: script`, file
   `scripts/<sg>/build_world.py` (canonical). Inputs:
   `observation = Ref("observe")`,
   `target_mask = Ref("in.target_mask")`,
   `target_obb = Ref("in.target_obb")`, `target_name = "target"`.
7. **`plan`** — `type: script`, file `scripts/<sg>/plan_grasp.py`
   (canonical). Inputs: `world_config = Ref("build_world.config")`,
   `observation = Ref("observe")`,
   `grasp_poses = Ref("offset_from_base.adjusted_grasp")`,
   `target_name = "target"`. `plan_grasp.py` auto-wraps the single bare
   Se3Pose into a one-element list. All four inputs are required.
8. **`execute`** — `type: tool`,
   `tool: "robot.execute_trajectory"`,
   `inputs: { trajectory: Ref("plan.trajectory"), subsample: 4 }`.
9. **`finalize`** — `type: script`, file
   `scripts/<sg>/finalize_trajectory.py` (canonical). Inputs:
   `trajectory = Ref("plan.trajectory")`. MANDATORY — see the
   `execute → finalize → close` hard_rule. Edges:
   `execute → finalize`, `finalize → close`.
10. **`close`** — `type: tool`, `tool: "robot.close_gripper"`,
    `inputs: { settle_steps: 60 }`. Edge directly from `close` to the
    `grasped` success marker.

The cross-subgraph output binding:

```python
sg.set_outputs(
    ee_pose_at_grasp=Ref("observe.arms.0.ee_pose"),
    grasp_pose=Ref("offset_from_base.adjusted_grasp"),
)
```

Wire the exit:

```python
sg.add_edge("close", "grasped")
sg.add_edge("grasped", END)

sg.add_exit("grasped")
sg.set_on_error("failed")
```

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Gripper has closed on the object after the descend. Route to the next subgraph (typically `transporting-objects`). |
| `failed` | Any grasp-attempt failure: collision-aware planning failure or trajectory execution error (a raise to `on_error`). Lives only in `on_error` — never declare a `failed` node. |

## Checkpoints

Express grasp success as a `validate=True` postcondition checkpoint
`target_held` on the `close` state (the gripper is holding the target
after close). Do not add a re-check-and-raise node.

## See also

- `../grasping-with-planner/SKILL.md` — the sampled-candidate OBB
  counterpart; this skill mirrors its
  `approach`/`build_world`/`plan`/`finalize` tail (the scripts are
  bundled here so the skill is self-contained).
