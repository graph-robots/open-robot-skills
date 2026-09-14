---
name: verifying-placement
description: Confirms from one RGB-D view that a released object went through an aperture, accepting either that it vanished into the container or that its visible centre lies below the rim within the opening's footprint, and routing not_placed when it is still visible above or beside the aperture; an opt-in multi-view mode instead separates verified, clear, blocked and uncertain. Use when an insertion-and-release must be checked without simulator state, after the release has completed.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: verification, tags: [placement, verification, aperture, rgb-d, multi-view]}
gap:
  allowed_tools:
    - sam3.segment_text
    - geometry.mask_to_world_points
    - robot.get_ee_pose
  required_inputs:
    observation: Observation
    object_description: string
    aperture_center: Vec3
    rim_z: float
  produces_outputs:
    verified: bool
    evidence: string
  exit_conditions:
    verified: The object is no longer visible (single-camera mode), or its visible centre is below the rim within the aperture footprint (single-camera mode), or every confident detection at the aperture stays below the above-rim test (multi-view mode).
    not_placed: Single-camera mode only. The object is still visible above the rim or outside the aperture footprint, or too little depth was valid to confirm.
    clear: Multi-view mode only. No detection at the aperture in any of the required views. Nothing obstructs the hole, but delivery is not established by absence.
    blocked: Multi-view mode only. A confident detection at the aperture reaches more than above_rim_tolerance_m above the rim; the object is jammed in or on the opening.
    uncertain: Multi-view mode only. Fewer than min_views calibrated views of the aperture, or only weak detections reach above the rim; observe again rather than proceed.
  canonical_scripts:
    - verify_placement: scripts/verify_placement.py
  streaming: false
---

# verifying-placement

Use once, after `executing-feature-mating` has released the object. Take a
fresh observation and call `verify_placement` with the aperture the fixture
perception measured (`aperture_center`, `rim_z`).

## Single-camera mode (default)

The script segments `object_description` in `camera_name`. Below
`score_min` the object is taken to have disappeared into the container,
which is `verified`. Otherwise its mask is back-projected and the median
must lie more than `depth_margin_m` below `rim_z` and within
`xy_tolerance_m` of the aperture centre in XY; anything else, including a
mask with fewer than `min_points` valid depth points, routes `not_placed`
with the evidence spelled out. The script never raises on a verdict.

## Multi-view mode (`views`)

Pass `views: [overhead, agentview]` (camera names) when items are posted one
after another into the same opening, where "not seen" is not reliable. The
script reads the live TCP with `robot.get_ee_pose`, then in every named
camera:

1. crops to a world box around the aperture (`crop_half_width_m`,
   `crop_below_m`, `crop_above_m`, `crop_padding_px`). A camera that cannot
   see that box does not count as a view;
2. judges every mask the segmenter returns (`max_results=0`) above
   `detection_score_floor`, keeping those centred within `xy_tolerance_m` of
   the aperture;
3. skips flat-lid masks (more than `flat_lid_fraction` of the points within
   `flat_lid_tolerance_m` of `rim_z`) and masks within `hand_exclusion_m` of
   the TCP (counted in `ignored_robot_masks`);
4. splits the rest into strong and weak at `strong_score_min`.

A detection is "above the rim" when at least `min_points` of its points,
and at least `above_rim_min_fraction` of them, are more than
`above_rim_tolerance_m` (10 mm) above `rim_z`. The verdict, in order:

| `route` = `status` | When |
| --- | --- |
| `uncertain` | fewer than `min_views` (2) calibrated views |
| `blocked` | a strong detection is above the rim |
| `uncertain` | only a weak detection is above the rim |
| `verified` | strong detections exist, none above the rim |
| `clear` | nothing at the aperture in any view |

The output adds `status`, `observations` (score, centre, camera, crop and
extent of the detections behind the verdict) and `ignored_robot_masks`.
`verified` is true only for `verified`.

Routing as the sharps-disposal graph does it: `verified` and `clear` go on to
`counting-remaining-items`; `blocked` goes to a recovery push, then a second
check, and a second `blocked` aborts; `uncertain` aborts. `clear` is not
`verified` on purpose. Pair it with a remaining-items count, or with a
proprioceptive seat check before the release, rather than treating it as
proof of delivery.

## Boundaries

- Only meaningful for an opaque container after a completed release; an
  object still in the gripper would also "disappear" from an overhead view.
  Multi-view mode excludes the hand by distance, not by identity. An object
  still held within `hand_exclusion_m` of the TCP is ignored.
- The above-rim margin was measured on a lidded box holding a pile of
  syringes. A shallow opening, where a legitimately placed object sticks up
  10 mm, needs its own `above_rim_tolerance_m`.
- No simulator query or object pose is read.
