---
name: counting-remaining-items
description: Counts how many described items are still left in the workspace from calibrated RGB-D views, judging every segmenter mask, re-examining weak detections in a padded crop, excluding the robot hand, a named exclusion zone and anything above a height limit, and routing more, none or uncertain so that a negative answer needs several cameras that saw nothing even weakly. Use when a pick-and-place loop must decide whether another pass is needed and a false "done" is worse than an extra look.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, loop, counting, multi-view, rgb-d, sam3]}
gap:
  allowed_tools:
    - sam3.segment_text
    - geometry.mask_to_world_points
    - robot.get_ee_pose
  required_inputs:
    observation: Observation
    object_description: string
  produces_outputs:
    remaining: int
    status: string
  exit_conditions:
    more: At least one eligible detection scored at or above score_min in some camera; remaining is the number of distinct centres seen in that camera (a lower bound). Route to the next target.
    none: At least min_negative_views cameras were consulted and none returned an eligible detection, weak or strong. remaining is 0. Route to done.
    uncertain: Only weak eligible detections survived the re-crop, or fewer than min_negative_views cameras were available. remaining is -1. Observe again or abort; never treat as done.
  canonical_scripts:
    - count_remaining: scripts/count_remaining.py
  streaming: false
---

# counting-remaining-items

A loop terminator for "move every X" tasks. Call `count_remaining` on a fresh
observation after each item has been handled, with the robot parked where it
does not hide the workspace (the sharps graph goes home first).

## Using it as a loop terminator

Route on `route` (equal to `status`):

| Exit | Route to | Why |
| --- | --- | --- |
| `more` | the next-target perception | something is left |
| `none` | done | enough cameras saw nothing at all |
| `uncertain` | abort, or observe again | a weak mask stayed weak, or too few views |

The sharps-disposal graph wires exactly that: `more → target`,
`none → done`, `uncertain → abort`, with `on_error: uncertain`, so a raised
tool error cannot end the task as a success either.

## Parameters

- `object_description` is the segmenter prompt. Nothing defaults to an
  object class.
- `camera_names` (default `[overhead, agentview]`) sets which cameras are
  consulted and in what order. The first camera with a strong detection
  returns `more` at once.
- `score_min` (0.08) splits strong from weak. Detections below
  `detection_score_floor` (0.01) are dropped.
- `min_negative_views` (2) is how many cameras must see nothing before
  `none`.
- `recrop_weak` / `recrop_padding_px` (50). When a view has only weak
  detections, it is segmented again inside their padded pixel union. What
  becomes strong there counts; what stays weak makes the result `uncertain`.
- Eligibility:
  - `max_z`: centres must be strictly below it. Pass the container's rim
    height, so items already placed and sticking out do not count.
  - `exclude_center` + `exclude_radius_m` (0.12): an XY exclusion disc, e.g.
    around the aperture.
  - `hand_exclusion_m` (0.14): distance from the live TCP, read with
    `robot.get_ee_pose`. Set 0 to skip that call.
  - `min_length_m`, `max_length_m`, `max_width_m`, `min_aspect`: limits on
    the 5-95 % extents along the detection's principal axes. `None` (the
    default) means no gate. The sharps graph's syringe values were 0.025,
    0.22, 0.045 and 2.0.
- `distinct_m` (0.02): detections closer than this are one item.

Besides `remaining` and `status`, the output carries `centers` (a list of
`{x, y, z}`) and `evidence` (per-detection score, centre, camera, crop and
extent; for `none`, the number of negative views). Neither list is a
registry type, so neither is listed under `produces_outputs`.

## Boundaries

- `remaining` under `more` is a lower bound from one camera, not a census.
  Use it to decide whether to loop, not to plan how many passes are left.
- Detection is by text prompt and shape only. Items that the prompt also
  matches elsewhere, like placed items still visible below `max_z` outside
  the exclusion disc, count as remaining. Tighten `max_z`, the exclusion disc
  or the shape limits.
- Cameras that are not in the observation are not counted as views, so a
  single-camera setup can never reach `none` with the default
  `min_negative_views`. That is deliberate; lower it knowingly.
- No simulator query or object pose is read.
