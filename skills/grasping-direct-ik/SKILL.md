---
name: grasping-direct-ik
description: Direct IK align-then-descend grasping. The gripper pre-rotates to
  the grasp orientation at a safe height ABOVE the target before descending
  straight down, avoiding the twist-while-closing failure mode of a blended
  rotate+descend. The grasp rotation is rebuilt so the hand's declared closing
  axis (robot.describe_gripper, composed by robot.grasp_frame) closes across the
  target's short horizontal axis, and the hover clearance is the hand's own
  (robot.describe_workspace) unless the workflow pins it. Use when no
  trajectory planner (curobo) is deployed or the scene is uncluttered enough
  that a straight-line approach is safe.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: grasping, tags: [grasping, manipulation, direct-ik]}
gap:
  requires: {connector: [robot.describe_workspace, robot.describe_gripper, robot.grasp_frame, robot.set_grip, robot.wait_steps, motion.plan_joint, motion.plan_linear]}
  allowed_tools:
    - geometry.top_down_grasp_candidates
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.get_ee_pose
    - robot.open_gripper
    - robot.close_gripper
    - robot.describe_workspace
    - robot.describe_gripper
    - robot.grasp_frame
    - robot.set_grip
    - robot.wait_steps
    - robot.execute_trajectory
    - motion.plan_joint
    - motion.plan_linear
    - curobo.plan_to_pose
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
      refine / align-pose construction assumes `compute_grasp.candidates.poses.0`
      exists.
    - >
      The `align_pose` descends straight down with the gripper pre-rotated.
      DO NOT skip the `compute_align` + `rotate_align` states — a direct
      go_to_pose to the grasp pose blends rotation and descent and twists the
      gripper against the object.
    - >
      `descend` targets the SAME pose `compute_align` was given
      (`refine_grasp.grasp_pose` when `refine_grasp` is present, else
      `compute_grasp.candidates.poses.0`). Feeding `compute_align` one pose and
      `descend` another re-introduces the twist the hover exists to avoid.
    - >
      Do NOT add a `verify_gripper_grasp` node — there is no such skill;
      postcondition verification is a `validate=True` checkpoint. Express "the gripper is holding the target after close" as a
      `validate=True` checkpoint (`target_held`). See `## Checkpoints` below.
  canonical_scripts:
    - compute_align_pose: scripts/compute_align_pose.py
    - refine_top_down_grasp: scripts/refine_top_down_grasp.py
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

Nothing here is written for one hand. The grasp rotation is composed by the
connector's `robot.grasp_frame` from the hand's measured approach and closing
axes, the fingertip floor and the hover clearance come from
`robot.describe_gripper` / `robot.describe_workspace`, so the same subgraph
grasps the same way on a hand that closes along tool-local x and on one that
closes along tool-local y.

## When to use

- The `curobo` tool bundle is not deployed (no collision-aware planner
  available).
- The scene is uncluttered enough that a straight-line approach is safe.

## When NOT to use

- Cluttered scenes where the arm must thread between obstacles. Prefer
  `grasping-with-planner` if available.

## Recommended subgraph state flow

The subgraph state machine the agent generates should look like (7 states):

```text
open → compute_grasp → refine_grasp → compute_align → rotate_align → descend → close → grasped
```

(`grasped` is the success-marker `noop` from `sg.add_exit("grasped")`,
with an edge to `END`.)

State details:

1. **`open`** — `type: tool`, `tool: "robot.open_gripper"`, `inputs: { settle_steps: 40 }`.
2. **`compute_grasp`** — `type: tool`, `tool: "geometry.top_down_grasp_candidates"`,
   `inputs: { obb: Ref("in.target_obb") }`.
3. **`refine_grasp`** — `type: script`, file
   `scripts/<sg>/refine_top_down_grasp.py` (from this bundle's
   `canonical_scripts`). Inputs:
   `grasp_pose = Ref("compute_grasp.candidates.poses.0")`,
   `target_obb = Ref("in.target_obb")`. Returns `grasp_pose`: the candidate
   with its rotation rebuilt so this hand's closing axis lies across the
   OBB's short horizontal axis (`robot.describe_gripper` +
   `robot.grasp_frame(approach=-z, close_heading_deg)`), and its Z raised to
   the fingertip floor (`support_z + finger.reach_m + finger.clearance_m`)
   when the hand states its finger envelope. Keep this state: the candidate
   fan is world-aligned and a thin object's centred grasp jams the jaws on
   the table without the floor.
4. **`compute_align`** — `type: script`, file
   `scripts/<sg>/compute_align_pose.py`. Inputs:
   `grasp_pose = Ref("refine_grasp.grasp_pose")`,
   `target_obb = Ref("in.target_obb")`, optionally `clearance` (a literal in
   metres). Returns `align_pose` at the grasp XY and rotation, with
   `z = max(obb_top, grasp_z) + clearance`. Omit `clearance` (or pass `0`)
   and the script asks `robot.describe_workspace` for `align_clearance_m` —
   the hand's own envelope above the fingertips. Pin it (e.g. `0.12`) when
   the *held* object is what needs the room, such as a long tool that will
   hang below the fingertips on the way up.
5. **`rotate_align`** — `type: tool`, `tool: "robot.go_to_pose"`,
   `inputs: { pose: Ref("compute_align.align_pose") }`.
6. **`descend`** — `type: tool`, `tool: "robot.go_to_pose"`,
   `inputs: { pose: Ref("refine_grasp.grasp_pose") }` — the same pose
   `compute_align` was given, so hover and grasp share one rotation.
7. **`close`** — `type: tool`, `tool: "robot.close_gripper"`, `inputs: { settle_steps: 60 }`.
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

`scripts/<sg>/plan_to_pose.py` is the optional planned variant of a single
leg: it calls `curobo.plan_to_pose` from the observation's joint state to a
target pose and returns the `trajectory` (raising `PlanningFailed` when the
planner refuses) for a platform that deploys CuRobo as a fast single-pose IK
fallback without the goalset grasp planner.

## Packed-tray options (opt-in)

Folded in from RoboSimStudio's `sharps_disposal/gap_perception_v2` (sweep s8:
16/30 cluttered syringe trays fully solved, 152/176 syringes posted), where a
thin barrel is taken between neighbours 37-49 mm away. Every option is off by
default and adds no tool call until asked for; the scripts' docstrings carry
the measurements.

`refine_top_down_grasp` (pure math, still one `describe_gripper` and one
`grasp_frame` call):

| input | default | effect |
|---|---|---|
| `support_z` | `None` (OBB bottom) | perceived support height for the fingertip floor |
| `preclose_max_width_m` | `None` (descend open) | turns on a pre-closed descent; widest jaw |
| `jaw_clearance_m` | `None` (widest) | measured room across the jaw; width = `2·(clearance − finger_band_past_jaw_m − preclose_neighbour_margin_m)`, clipped to `[object_width_m + preclose_object_margin_m (0.006), max]` |
| `finger_band_past_jaw_m` | `None` → `(open_footprint_m − span_m)/2` from `describe_gripper` | how far the fingertip band reaches past each side of the gap |
| `preclose_neighbour_margin_m` | `0.002` | margin to the neighbour |
| `fingertip_beyond_tcp_m` | `None` (no drop) | `[[gap, fingertip beyond TCP], …]` for this hand; the TCP rises by the drop at the pre-close width. Hand data: `describe_gripper` does not report it |
| `target_cloud` + `width_percentile` | `None` | `object_width_m` from the cloud's transverse radius percentile about the long axis, kept only inside `width_range_m` |
| `object_width_m` | `None` (OBB short width) | explicit width, and the fallback for the cloud estimate |
| `end_feature_center` + `station_offset_m` | `None` | grasp XY at a known offset along the long axis from a perceived end feature (`station_min_points`: 30 for the cloud-median origin) |

Additive outputs: `object_width_m`, `preclose_width_m` (`0.0` = open) and
`grasp_station` (`candidate` / `end_feature_offset`). The rotation stays
`robot.grasp_frame`'s, and the height stays the candidate's under the floor.

`execute_grasp_align`:

| input | default | effect |
|---|---|---|
| `half_turn_tolerance_rad` | `None` (`robot.go_to_pose`) | plan with `motion.plan_joint(orientation="lock")`, execute the trajectory, and retry at yaw + π when the rotation error exceeds it, keeping the better |
| `approach_check_tolerance_m` / `_rad` | `None` (profile tolerances only) | after the profile verification passes, the same TCP reading must also be within this position / full-rotation error, else raise |
| `grasp_pose` | `None` | descend to it with `robot.go_to_pose_cartesian` + `robot.wait_steps(settle_steps=30)` after the pre-grasp is verified; the options below need it |
| `descent_planner` | `false` | first fly an orientation-locked `motion.plan_linear` line to the grasp (`robot.execute_trajectory` slowed 3x, TCP within 10 mm / 0.035 rad or raise); a refused, empty or raising plan leaves the Cartesian descent alone |
| `preclose_width_m` | `None` | `robot.set_grip(width_m=…)` before the descent |
| `pad_envelope` | `None` | `{across_m, along_m, vertical_m, rotation_rad, across_axis="x", max_corrections=0, max_correction_m=0.012, max_correction_rad=0.06}` in the grasp's tool frame; bias corrections, then raise when still outside |
| `object_width_m`, `squeeze_m` | `None`, `0.002` | close with `robot.set_grip(object_width_m=…, squeeze_m=…)` |

Additive outputs: `commanded_pose`, `commanded_grasp_pose`, `grasp_final_pose`,
`half_turn_used`, `bias_corrections`, `descent_planned`. The sharps graph's
settings, for reference: `preclose_max_width_m=0.030`, `width_percentile=35`,
`width_range_m=[0.006, 0.025]`, `object_width_m=0.0107`, `station_offset_m=0.043`
from the needle cap, `half_turn_tolerance_rad=0.05`,
`approach_check_tolerance_m=0.003` / `approach_check_tolerance_rad=0.035`,
`descent_planner=true`, `pad_envelope` 4.5 / 10 / 5 mm and 0.04 rad with two
corrections.

## Hard rules

1. Use `geometry.top_down_grasp_candidates` (returns
   `candidates: {poses: list[Se3Pose]}`), not
   `geometry.top_down_grasp_from_obb` (single bare pose). The refine and
   align-pose constructions assume `compute_grasp.candidates.poses.0` exists.
2. The `align_pose` descends straight down with the gripper pre-rotated.
   Do NOT skip the `compute_align` + `rotate_align` states — a direct
   `robot.go_to_pose` to the grasp pose blends rotation and descent and
   twists the gripper against the object.
3. `descend` uses the pose `compute_align` was given (`refine_grasp.grasp_pose`).
   Never descend to the raw candidate after hovering at the refined rotation.

## Required end states

| End state | Meaning |
|---|---|
| `grasped` | Gripper has closed on the object after the descend. Route to next subgraph (typically `transporting-objects`). |
| `failed` | Any grasp-attempt failure: planning failure or trajectory execution error (a raise to `on_error`). Coordinator routes to abort. Lives only in `on_error` — never declare a `failed` node. |


## See also

- `references/design_align_then_descend.md` — why pre-rotate-then-descend
  beats blended rotate+descend.
- `scripts/refine_top_down_grasp.py` — the close-axis rotation and fingertip
  floor, read off the live hand.
- `scripts/compute_align_pose.py` — the canonical align-pose construction.
- `scripts/plan_to_pose.py` — the optional CuRobo single-pose leg.
