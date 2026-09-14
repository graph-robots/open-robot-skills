---
name: verifying-grasps
description: Three independent checks on one grasp attempt -- that the hand arrived where it was sent before the jaws close, that the jaw gap after the close is not the mechanical stop, and that the lifted object is visible above the surface and near the hand. Use when a grasp closed on a small or thin object and the graph must know, before carrying or inserting, whether anything is actually in the gripper.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: verification, tags: [grasp, verification, rgb-d, wrist-camera]}
gap:
  requires: {connector: [robot.describe_workspace, robot.describe_gripper]}
  allowed_tools:
    - robot.get_ee_pose
    - robot.go_to_pose_cartesian
    - robot.get_observation
    - robot.describe_workspace
    - sam3.segment_text
    - geometry.mask_to_world_points
    - robot.get_gripper
    - robot.describe_gripper
  required_inputs:
    object_description: string
  produces_outputs:
    verified: bool
    observed_center: Vec3
  exit_conditions:
    verified: The object was seen after the lift, above the surface and near the hand.
    not_held: Nothing matching was seen, or what was seen stayed on the surface or far from the hand.
    reached: The hand is close enough to the commanded grasp pose to have the object between the pads.
    missed: The hand stopped further from the commanded pose than the object's own smallest half-extent allows.
    held: The jaws closed on something -- the gap is above the mechanical stop.
    empty: The jaws ran to the stop, so they closed on nothing.
    unknown: The hand could not be read, which is a gap in the instrument rather than a failed grasp.
  canonical_scripts:
    - verify_grasp: scripts/verify_grasp.py
    - verify_reach: scripts/verify_reach.py
    - verify_grip: scripts/verify_grip.py
  streaming: false
---

# verifying-grasps

Three instruments on one grasp attempt, at three different moments. A graph may
use any of them alone; together they separate "the hand never got there" from
"it got there and closed on nothing" from "it closed on something and dropped
it", which a single check cannot.

| when | script | routes |
| --- | --- | --- |
| after the approach, before the close | `verify_reach` | `reached` / `missed` |
| immediately after the close | `verify_grip` | `held` / `empty` / `unknown` |
| after a short lift | `verify_grasp` | `verified` / `not_held` |

`verify_reach` compares the commanded grasp pose with the observed one and
routes `missed` when the distance exceeds the object's own smallest half-extent
(clamped to `floor_m`..`ceiling_m`). The tolerance is the object's size rather
than a constant because the question is whether the object can still be between
the pads, and that is a question about the object. Cheap -- no camera, no
motion -- and it fires before the close, when the graph can still do something
about it.

`verify_grip` reads `robot.get_gripper` and converts through
`robot.describe_gripper`'s `width_fit`, routing `empty` when the gap is within
`empty_margin_m` of the mechanical stop. Also cameraless. It reports
`expected_width_m` alongside, which is context and not a test: a planner that
grasps at a fitted line's centre may legitimately hold a wide part of the body.

`verify_grasp` raises the end effector by `lift_m` along world Z with a
Cartesian move, takes a fresh observation, and looks for `object_description`
(then `marker_description`, when given) first in the camera whose name contains
`wrist_camera_keyword` and then in `overhead_camera_name`. The first
segmentation at or above `score_min` is back-projected; the grasp is
`verified` only when the cloud has at least `min_points` valid depth points,
its median lies at least `min_above_table_m` above
`robot.describe_workspace().surface_z`, and within `max_hand_distance_m` of
the lifted hand. Every failed gate returns `route: not_held` with a `reason`.

`score_min` is deliberately low: a small held object occupies only a few
dozen wrist pixels and true positives score in the hundredths. Confidence
admits a candidate; geometry decides.

### Look-alikes in view (opt-in)

Folded in from RoboSimStudio's `sharps_disposal/gap_perception_v2` (sweep s8),
whose lifted wrist looks down on a tray still holding the held syringe's twins.
They are large, well-lit, and outscore the foreshortened one between the
fingers: on `tray_clutter_t03` the held syringe came back fourth, 23 mm from
the hand, behind three tray syringes 217-281 mm away. Each option is off by
default:

| input | default | effect |
|---|---|---|
| `max_masks_per_query` | `3` | `max_results` asked of `sam3.segment_text`; `0` = every mask |
| `nearest_to_hand` | `false` (top mask of the first query/camera decides) | every mask at or above `score_min` is back-projected and the one nearest the lifted hand, across all queries and cameras so far, is judged; it returns as soon as that one passes, in the same query order, and asks `robot.describe_workspace` only once a candidate has a cloud |
| `max_span_m` / `max_width_m` | `None` | a cloud longer or wider than one object is not a candidate (in top-1 mode: `not_held`, "larger than one object"). Keeps the arm's own mask out |

The graph ran `nearest_to_hand` with `max_masks_per_query: 0`, `0.25` m / `0.06` m
and `lift_m: 0` (it lifts in its own node). The routing is unchanged: every
failed gate is `not_held` with a `reason`.

## Boundaries

- The lift in `verify_grasp` is the only motion any of the three makes; the
  object is neither released nor moved elsewhere.
- No force or tactile reading is consulted. `verify_grip` uses the jaw gap,
  which every parallel hand reports; torque and slip belong to a connector's
  own grasp check when it has one.
- Only a rig without a wrist camera raises, and only from `verify_grasp`. A
  missing input elsewhere routes the benign way -- `reached`, `unknown` --
  because absent evidence is not evidence of a failed grasp.
