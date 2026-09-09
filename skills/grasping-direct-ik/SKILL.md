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
    - robot.go_to_pose_cartesian
    - robot.get_ee_pose
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
    - execute_grasp_align: scripts/execute_grasp_align.py
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
   `target_obb = Ref("in.target_obb")`, the selected `grasp_profile`, and
   `arm_id`. Returns the corrected `grasp_pose` and its `align_pose`. The
   correction aligns the embodiment's calibrated closing axis to the
   perceived OBB short axis and keeps the fingertips above the support.
4. **`rotate_align`** — `type: script`, file
   `scripts/<sg>/execute_grasp_align.py` (from this bundle's
   `canonical_scripts`). Inputs:
   `pose = Ref("compute_align.align_pose")` and an optional declarative
   `grasp_profile`. The script verifies the reached translation and both
   grasp-frame axes. A pose that already passes is never disturbed; a pose
   that would fail verification receives Cartesian correction before the
   failure is reported. For a large translation residual, recovery first
   translates at the controller's current wrist orientation and then corrects
   rotation, avoiding an otherwise infeasible combined Cartesian move near a
   workspace boundary.
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

## Declarative grasp profile

The caller should provide one selected `grasp_profile` containing
`approach_clearance_m`, `position_tolerance_m`, `angular_tolerance_deg`, and
`cartesian_recovery_threshold_m`. Set `use_embodiment_calibration: true` to
align the perceived short axis through `robot.grasp_frame` and enforce the
reported finger reach/clearance; otherwise the supplied grasp candidate's
rotation and height are preserved. These values describe gripper clearance and
grasp-region tolerance; the canonical scripts never select behavior from an
object or task name. Functional-feature exclusion and preferred grasp regions
belong to upstream object-feature perception. `target_kind + grasp_profiles`
is accepted only as a compatibility adapter for already-materialized graphs.
Set `cartesian_recovery_on_verification_failure: false` only when a platform
cannot safely make the short local correction; it defaults to true and is
entered exclusively for an arrival that would otherwise fail verification.

Both canonical scripts accept `arm_id`; all gripper metadata, grasp-frame
calibration, motion, and verification calls use that arm. This makes the same
skill valid for either side of a dual-arm embodiment.

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Gripper has closed on the object after the descend. Route to next subgraph (typically `transporting-objects`). |
| `failed` | Any grasp-attempt failure: planning failure or trajectory execution error (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |


## See also

- `references/design_align_then_descend.md` — why pre-rotate-then-descend
  beats blended rotate+descend.
- `scripts/compute_align_pose.py` — the canonical align-pose construction.
