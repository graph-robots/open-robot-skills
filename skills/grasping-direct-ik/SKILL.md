---
name: grasping-direct-ik
description: Direct IK align-then-descend grasping. The gripper pre-rotates to
  the grasp orientation at a safe height ABOVE the target before descending
  straight down, avoiding the twist-while-closing failure mode of a blended
  rotate+descend. Use when no trajectory planner (curobo) is deployed or the
  scene is uncluttered enough that a straight-line approach is safe.
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, manipulation, direct-ik]}
gap:
  allowed_tools:
    - geometry.top_down_grasp_candidates
    - robot.go_to_pose
    - robot.open_gripper
    - robot.close_gripper
  exit_conditions:
    grasped: Object held in the gripper after `close`.
    failed: Any failure during the grasp attempt — planning failure or trajectory execution error (a raise to `on_error`). Coordinator routes to abort.
  required_inputs:
    target_obb: OrientedBoundingBox
  hard_rules:
    - >
      Use `geometry.top_down_grasp_candidates` (returns
      `candidates: {poses: list[Se3Pose]}`), NOT
      `geometry.top_down_grasp_from_obb` (single bare pose). The downstream
      align-pose construction assumes `compute_grasp.candidates.poses.0`
      exists.
    - >
      The `align_pose` descends straight down with the gripper pre-rotated.
      DO NOT skip the `compute_align` + `rotate_align` states — a direct
      go_to_pose to the grasp pose blends rotation and descent and twists the
      gripper against the object.
    - >
      Do NOT add a `verify_gripper_grasp` node — there is no such skill;
      postcondition verification is a `validate=True` checkpoint. Express "the gripper is holding the target after close" as a
      `validate=True` checkpoint (`target_held`). See `## Checkpoints` below.
  canonical_scripts:
    - compute_align_pose: scripts/compute_align_pose.py
    - plan_to_pose: scripts/plan_to_pose.py
  references:
    - title: Why pre-rotate-then-descend instead of blended rotate+descend?
      path: references/design_align_then_descend.md
  streaming: false
---

# grasping-direct-ik

Direct-IK grasp: rotate the gripper to grasp orientation at a safe height
above the target, then descend straight down, then close. No trajectory
planner — works on platforms where CuRobo is not deployed, or in
uncluttered scenes where planning is overkill.

## When to use

- The `curobo` tool bundle is not deployed (no collision-aware planner
  available).
- The scene is uncluttered enough that a straight-line approach is safe.

## When NOT to use

- Cluttered scenes where the arm must thread between obstacles. Prefer
  `grasping-with-planner` if available.

## Recommended subgraph state flow

The subgraph state machine the agent generates should look like (6 states):

```text
open → compute_grasp → compute_align → rotate_align → descend → close → grasped
```

(`grasped` is the success-marker `noop` from `sg.add_exit("grasped")`,
with an edge to `END`.)

State details:

1. **`open`** — `type: tool`, `tool: "robot.open_gripper"`, `inputs: { settle_steps: 40 }`.
2. **`compute_grasp`** — `type: tool`, `tool: "geometry.top_down_grasp_candidates"`,
   `inputs: { obb: Ref("in.target_obb") }`.
3. **`compute_align`** — `type: script`, file
   `scripts/<sg>/compute_align_pose.py` (from this bundle's
   `canonical_scripts`). Inputs:
   `grasp_pose = Ref("compute_grasp.candidates.poses.0")`,
   `target_obb = Ref("in.target_obb")`. Returns `align_pose`.
4. **`rotate_align`** — `type: tool`, `tool: "robot.go_to_pose"`,
   `inputs: { pose: Ref("compute_align.align_pose") }`.
5. **`descend`** — `type: tool`, `tool: "robot.go_to_pose"`,
   `inputs: { pose: Ref("compute_grasp.candidates.poses.0") }`.
6. **`close`** — `type: tool`, `tool: "robot.close_gripper"`, `inputs: { settle_steps: 60 }`.
   Edge directly from `close` to the `grasped` success marker; the
   subgraph's `on_error: "failed"` catches any raise from earlier steps.
   Whether the gripper actually closed on the object is checked by the
   `target_held` postcondition checkpoint (see `## Checkpoints`), NOT by
   a re-check-and-raise node (none such exists).

   ```json
   "edges": [ ..., ["close", "grasped"], ["grasped", "END"] ],
   "conditional_edges": {},
   "exit": { "router_field": null, "success_values": ["grasped"] },
   "on_error": "failed"
   ```

   The lift onto a safe carry height is handled by the next
   `transporting-objects` subgraph (its `waypoint_move` script lifts before
   lateral motion); do NOT add a lift step here.

## Hard rules

1. Use `geometry.top_down_grasp_candidates` (returns
   `candidates: {poses: list[Se3Pose]}`), not
   `geometry.top_down_grasp_from_obb` (single bare pose). The align-pose
   construction in step 3 assumes `compute_grasp.candidates.poses.0` exists.
2. The `align_pose` descends straight down with the gripper pre-rotated.
   Do NOT skip the `compute_align` + `rotate_align` states — a direct
   `robot.go_to_pose` to the grasp pose blends rotation and descent and
   twists the gripper against the object.

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Gripper has closed on the object after the descend. Route to next subgraph (typically `transporting-objects`). |
| `failed` | Any grasp-attempt failure: planning failure or trajectory execution error (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |


## See also

- `references/design_align_then_descend.md` — why pre-rotate-then-descend
  beats blended rotate+descend.
- `scripts/compute_align_pose.py` — the canonical align-pose construction.
