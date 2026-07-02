---
name: grasping-with-planner
description: Top-down grasping via a fast axis-locked linear descend with a
  collision-aware cuRobo fallback. A single `grasp_descend_linear` node rises,
  translates over the target, and descends straight down (Z-locked,
  orientation held) onto the object; if the straight-line solve is infeasible
  (far-edge reach, no IK for the fixed wrist) it falls back to the cuRobo
  planner over the candidate fan. Use when the curobo tool bundle is installed
  and a perceived object (OBB) must be grasped — the default grasping skill
  whenever cuRobo is deployed, in clean and cluttered scenes alike.
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, manipulation, collision-aware, curobo, linear]}
gap:
  allowed_tools:
    - geometry.top_down_grasp_candidates
    - geometry.build_world_config
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - curobo.plan_directed_linear
    - curobo.plan_to_grasp_poses
    - robot.execute_trajectory
    - robot.open_gripper
    - robot.close_gripper
  exit_conditions:
    grasped: Object held in the gripper after `close`.
    failed: Any failure during the grasp attempt — the cartesian descend AND the collision-aware planner fallback both failed (a raise propagating to `on_error`). Coordinator routes to abort.
  required_inputs:
    target_obb: OrientedBoundingBox
  produces_outputs:
    ee_pose_at_grasp: Se3Pose
    grasp_pose: Se3Pose
  hard_rules:
    - >
      ALWAYS begin with `robot.open_gripper` (settle_steps=40). Skipping this
      is the #1 silent grasp-failure mode when the gripper starts closed from
      a previous task step.
    - >
      Use `geometry.top_down_grasp_candidates` (returns
      `candidates: {poses: list[Se3Pose]}`), NOT
      `geometry.top_down_grasp_from_obb` (single bare pose). The grasp node
      takes candidate 0 as the primary target AND the full fan as the
      planner-fallback goalset — both `Ref("compute_grasp.candidates.poses.0")`
      and `Ref("compute_grasp.candidates.poses")` must be wired.
    - >
      The grasp is ONE `grasp_descend_linear` script node — do NOT split it
      back into separate approach / build_world / plan / execute nodes. The
      script already does rise → translate → Z-locked linear descend, and
      internally falls back to `curobo.plan_to_grasp_poses` (building its own
      per-observation collision world) when the straight-line solve fails.
    - >
      Place `observe` (robot.get_observation) AFTER `grasp_descend_linear`,
      before `close`, and bind `ee_pose_at_grasp` from it — this captures the
      live TCP pose AT the grasp for the downstream drop-height computation.
    - >
      Do NOT add a `verify_gripper_grasp` node — there is no such skill;
      postcondition verification is a `validate=True` checkpoint. Express "the gripper is actually holding the target after close" as a
      `validate=True` checkpoint (`target_held`). See `## Checkpoints` below.
  canonical_scripts:
    - grasp_descend_linear: scripts/grasp_descend_linear.py
    - select_short_axis: scripts/select_short_axis.py
  references:
    - title: Why a Z-locked linear descend (fingertip-frame, orientation LOCK)
      path: references/design_grasp_curobo.md
    - title: Gripper settle-step constants
      path: references/gripper_settle_constants.md
  examples:
    - title: Canonical top-down linear-descend grasp (open → compute_grasp → goto_grasp → observe → close)
      path: examples/canonical_subgraph.json
  streaming: false
---

# grasping-with-planner

Top-down grasping with a fast **axis-locked linear descend** and a
**collision-aware cuRobo fallback**. The primary path rises to a hover height,
translates over the target while rotating to the grasp yaw, and descends
straight down (Z-only, orientation locked) onto the object — which, unlike a
goalset planner, happily grips a *flat* object by simply lowering onto it. If
the straight-line solve is infeasible, the same node hands off to the cuRobo
planner, which builds a per-observation collision world and searches the whole
candidate fan for a reachable, collision-free wrist. This is the
`grocery_packing` grasp motion, distilled to the general single-object case.

## Install

This skill depends on the **curobo tool bundle** (`curobo.plan_directed_linear`,
`curobo.plan_to_grasp_poses`). cuRobo JIT-compiles CUDA extensions at install
time — build isolation must be off and `CUDA_HOME` must point at a toolkit
matching your torch build:

```bash
export CUDA_HOME=/usr/local/cuda
uv sync --extra curobo   # (pip: pip install -e "open-robot-skills[curobo]" --no-build-isolation)
```

See `tools/curobo/SKILL.md` for the full recipe and gotchas.

## When to use

- Default grasping skill whenever `curobo` is deployed — clean or cluttered.
- Flat / low-profile objects (a butter box, cream cheese) where a goalset
  planner struggles but a straight lower-on-top succeeds.

## When NOT to use

- `curobo` not deployed. Use `grasping-direct-ik` instead.
- The graspable region is NOT the OBB centroid — bowl rim, mug / moka-pot /
  frying-pan handle, or any off-center grasp. The OBB top-down candidates from
  `geometry.top_down_grasp_candidates` are centered on the OBB XY, so they slip
  on hollow centers and miss handles. Use `grasping-short-axis` for elongated
  handles, where the centroid is graspable but orientation is what matters.

## Recommended subgraph state flow

6 states, in order:

```text
open → compute_grasp → goto_grasp → observe → close → grasped
```

(`grasped` is the success-marker `noop` from `sg.add_exit("grasped")`,
with an edge to `END`.)

State details:

1. **`open`** — `type: tool`, `tool: "robot.open_gripper"`, `inputs: { settle_steps: 40 }`.
2. **`compute_grasp`** — `type: tool`, `tool: "geometry.top_down_grasp_candidates"`,
   `inputs: { obb: Ref("in.target_obb") }`. Returns
   `candidates: {poses: list[Se3Pose]}` — a yaw fan of top-down grasps.
3. **`goto_grasp`** — `type: script`, file `scripts/<sg>/grasp_descend_linear.py`
   from this bundle's canonical_scripts. Inputs:
   `grasp_pose = Ref("compute_grasp.candidates.poses.0")`,
   `candidate_poses = Ref("compute_grasp.candidates.poses")`,
   `target_obb = Ref("in.target_obb")`, `hover_z = 0.2`. Rises to `hover_z`,
   translates over the target at the grasp yaw, then Z-locked linear-descends
   onto it (shallow grip near the perceived top, floored a hair above the
   object base so the fingers never ram the table). On an infeasible cartesian
   solve it falls back to `curobo.plan_to_grasp_poses` over `candidate_poses`.
4. **`observe`** — `type: tool`, `tool: "robot.get_observation"`,
   `inputs: {}`. Captures the arm state AT the grasp (post-descend, pre-close)
   so `ee_pose_at_grasp` reflects the real TCP pose the object was gripped at.
5. **`close`** — `type: tool`, `tool: "robot.close_gripper"`, `inputs: { settle_steps: 60 }`.
   Edge directly from `close` to the `grasped` success marker; the subgraph's
   `on_error: "failed"` catches any raise from `goto_grasp` (both paths failed).
   Whether the gripper actually closed on the object is checked by the
   `target_held` postcondition checkpoint (see `## Checkpoints`), NOT by a
   re-check-and-raise node (none such exists).

   The subgraph publishes **two** cross-subgraph outputs:

   - `ee_pose_at_grasp` — the live TCP pose captured at the `observe` step
     (which sits between the descend and `close`). Downstream
     `transporting-objects` uses it to compute a drop height that accounts for
     the panda hand-to-tcp offset and the held object's geometry.
   - `grasp_pose` — the **computed** grasp pose emitted by `compute_grasp`,
     i.e. what the descend is *targeting*. Distinct from `ee_pose_at_grasp`
     (the *actual* EE pose at grasp time). Exposing `grasp_pose` lets the
     checkpoint author write an output-anchored verifier like
     `predicate=lambda w, o: o["grasp_pose"]["position"]["z"] > 0.01` to catch
     sub-table grasp poses *before* they cascade into a `target_held=False`
     failure.

   **Hard rule on the output binding:** the `robot.get_observation` response
   is an `Observation { cameras: list[CameraFrame]; arms: list[ArmState] }`.
   There is **no** flat `ee_pose` field — the EE pose lives at
   `arms[0].ee_pose`. The binding must therefore be exactly:

   ```python
   sg.set_outputs(
       ee_pose_at_grasp=Ref("observe.arms.0.ee_pose"),
       grasp_pose=Ref("compute_grasp.candidates.poses.0"),
   )
   ```

   Do **not** write `Ref("observe.ee_pose")` — that path does not exist and the
   cross-subgraph binding will silently resolve to None, sending the downstream
   `compute_drop_pose.py` into its inferior no-`ee_pose_at_grasp` fallback
   (drop height too high, placement misses).

   Cross-subgraph data flow is by name, so any downstream subgraph that
   declares `ee_pose_at_grasp` as an input automatically receives it.

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

For a thin/elongated target (frypan handle, screwdriver, spoon, rod) insert ONE
`type: script` state `select_short_axis` (`scripts/<sg>/select_short_axis.py`)
between `compute_grasp` and `goto_grasp`. Inputs:
`target_obb = Ref("in.target_obb")`,
`candidate_poses = Ref("compute_grasp.candidates.poses")`. It reorders the
candidate fan so poses whose finger-opening axis aligns with the OBB's short
horizontal axis come first (count and pose values preserved — only the ordering
changes), then wire `goto_grasp`'s `grasp_pose`/`candidate_poses` against
`Ref("select_short_axis.poses...")` instead. For a deterministic,
geometry-locked single short-axis pose use the dedicated `grasping-short-axis`
skill instead.

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Gripper has closed on the object after the descend. Route to next subgraph (typically `transporting-objects`). |
| `failed` | Grasp-attempt failure: the cartesian descend AND the cuRobo planner fallback both failed (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |


## See also

- `references/design_grasp_curobo.md` — the Z-locked (fingertip-frame,
  orientation-LOCK) linear descend and why it beats a blended rotate+descend.
- `references/gripper_settle_constants.md` — settle-step tunings.
- `scripts/{grasp_descend_linear,select_short_axis}.py` — canonical scripts.
