---
name: planning-held-object-motion
description: "Plans the carry and engagement phases for an already-held rigid object: a clearance-first lift, the smallest feasible symmetry-equivalent reorientation, an orientation-locked transit above the fixture, or a direct plan to the approach pose; then a typed engagement or a straight linear insertion, optionally re-observing the held tip from the wrist first. Use when a held object must be brought to a fixture for insertion, hanging, packing, racking, or constrained sorting and an execution skill will run the resulting plans."
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: planning, tags: [motion-planning, held-object, clearance, insertion, hanging, packing, sorting]}
gap:
  requires: {connector: [motion.plan_joint]}
  allowed_tools:
    - robot.get_ee_pose
    - motion.plan_joint
    - sam3.segment_text
    - geometry.mask_to_world_points
  required_inputs:
    held_feature_in_tcp: Se3Pose
    fixture_feature: FunctionalFeature
    relation: string
    world_config: WorldConfig
    attached_object: AttachedObject
  produces_outputs:
    reorientation_plan: PoseSequence
    placement_plan: PoseSequence
  exit_conditions:
    planned: A carry plan or an engagement plan was produced.
    blocked: No feasible symmetry-equivalent held-object orientation was found (unless accept_unchecked_symmetry), or the direct strategy was requested without an approach pose.
  canonical_scripts:
    - plan_clearance_motion: scripts/plan_clearance_motion.py
    - plan_feature_engagement: scripts/plan_feature_engagement.py
    - plan_linear_engagement: scripts/plan_linear_engagement.py
  streaming: false
---

# planning-held-object-motion

Turn a feature mate into safe phases for a held object:

```text
lift away from support
  -> rotate minimally in clear space
  -> translate above the fixture with orientation fixed
  -> approach and engage linearly
```

The skill plans; it does not move the robot or release the object. Follow it
with `executing-held-object-motion` to execute the carry plan and
`executing-feature-mating` to execute the engagement plan.

## Choosing the carry strategy

`plan_clearance_motion` takes `strategy`:

- `"clearance_first"` (default) derives a lift from the farthest attached
  sphere, selects the smallest reachable symmetry-equivalent rotation among
  eight roll candidates (each checked with `motion.plan_joint`), rotates while
  clear, and translates the aligned feature to a staging location on the free
  side of the fixture, `approach_clearance_m` along its axis. The staging
  location is derived from the fixture geometry and attached-object extent.
  `stage_outward_m` (default 0) shifts that location within the support
  plane away from the fixture toward the current hand — use it when a
  recessed opening is observed mostly at its far rim and the object should
  stage over the interior instead.
- `"direct"` escapes `escape_m` (default 6 cm) along the support normal as a
  `contact_transition` waypoint, then hands `approach_pose` to the executor as
  one `planned_joint` waypoint with `allow_start_contact` and three attempts.
  It calls no planner itself. Use it when the object is already clear of its
  support, the orientation change is small, and a single attached-object plan
  to the approach pose is appropriate; it requires `approach_pose`.

- `"carry_then_turn"` escapes like clearance-first, then carries to the same
  staging hand position still in the pick orientation (`planned_joint`), then
  turns in place there (`planned_joint` with `hold_position: true`). The turn
  waypoint carries a `turn_search` block (`fixture_center`, `axis`,
  `feature_offset`, `feature_rotation`, `transit_clearance_m`,
  `transit_target`, `insert_depths_m`) that `executing-held-object-motion`
  uses to choose the carry and wrist yaw together by planning them, insertion
  strokes and retract included; the planned smallest turn is only its
  geometry. `insert_depths_m` defaults to `[-0.020, -0.010, 0.0]` (signed feature
  positions along the fixture axis; set `turn_search_depths_m` to match the
  engagement actually flown); `turn_search=False` emits only `hold_position`.
  **This strategy needs a connector whose `motion.plan_to_pose` accepts
  `hold_position` and `seed_joints`** -- RoboSimStudio's does, the library's
  `tools/curobo` does not. Use it when a large turn at the pickup, or the
  planner's smallest-turn yaw, leaves the engagement near a joint limit.

`accept_unchecked_symmetry` (default `False`) keeps the plan when every symmetry
candidate fails the `motion.plan_joint` check, using the smallest turn
unchecked; left off, that still raises and routes to `blocked`. It is meant for
`carry_then_turn`, whose turn is re-planned by the executor.

If uncertain, choose the clearance-first strategy for a large orientation
change or a long held object. Do not add arbitrary midpoint waypoints and do
not remove collision geometry to make direct planning succeed.

## Engagement

`plan_feature_engagement` converts the typed relation and the approach,
engaged, and mate poses into execution semantics. `loop_over_shaft` crosses the
shaft tip with a tracked `cartesian_cross` leg — stopping on first contact can
leave a loop balanced against the tip without enclosing the shaft — and then
seats with the bounded `contact_seat` servo; `feature_to_fixture` uses the
contact servo for both legs; aperture insertion uses `planned_linear`
waypoints that allow goal contact. The task graph therefore does not decide
ad hoc whether a contact servo is needed.

`plan_linear_engagement` plans a straight insertion from the fixture feature
alone: a `planned_joint` pre-contact pose `precontact_clearance_m` before the
opening and a `planned_linear` engaged pose `engagement_depth_m` past it, both
holding the current orientation. Give it `observation` (the current
`Observation`), `object_description` and `direction_marker_description` to
re-observe the held tip from the wrist first: when both the object and its
direction marker are visible, the endpoint of the object's principal axis
toward the marker replaces the carried feature position.

Opt-in additions to `plan_linear_engagement` (defaults leave the legs and calls
as above):

- `stroke_depths_m` (e.g. `[-0.020, -0.010, 0.0]`) replaces the two legs with
  staged strokes to those signed depths along the fixture axis. The first uses
  `first_stroke_mode` (`"planned_linear"` by default, or `"planned_joint"`);
  the rest are `planned_linear` with start and goal contact allowed at
  `stroke_contact_margin_m` (0.008).
- `mirror_leading_end` aims whichever end of the held object leads along the
  fixture axis, mirroring the carried feature through the TCP when it trails
  (valid for a grasp at the object's middle). The wrist marker read then applies
  only when the carried feature leads.
- `leading_end_band_m: [min, max]` and `leading_end_max_lateral_m` (0.02) bound
  where an aimed end may sit in the hand frame, ahead of the TCP along the
  fixture axis; perceived ends outside are ignored. The sharps graph used
  `[0.045, 0.11]` for a 150 mm syringe gripped near its centre.
- `tip_source`: `"perception"` (default, perception replaces the carried
  feature when seen) or `"carried_unless_implausible"` (keeps the carried
  feature while it is inside the band; needs `leading_end_band_m`).
- `side_camera_names` reads the leading end from those fixed cameras first
  (segmenting `object_description` in a crop around the hand, keeping
  rod-shaped detections aligned with the fixture axis within
  `held_max_distance_m`, 0.15 m).

The plan also carries additive, call-free keys: `aim_source` (`carried`,
`carried_mirrored`, `side_view` or `wrist_tip`), `leading_end_in_hand`,
`aperture` (the fixture centre), `fixture_axis` and `rim_z`.

`carry_then_turn`, `accept_unchecked_symmetry` and the engagement additions
were folded from RoboSimStudio `sharps_disposal/gap_perception_v2`
(`plan_clearance_motion.py`, `plan_visual_insert.py`), measured in its sweep s8
(16/30 cluttered trays, 152/176 syringes). The syringe prompts, camera name and
45-110 mm band became parameters; the library's staging direction for
`loop_over_shaft` is kept. The script docstrings carry the measurements.

## Recommended skill sequence

1. Use `registering-held-objects` to obtain `held_feature_in_tcp` and
   `attached_object`.
2. Use `computing-feature-mating-poses` when explicit approach/engaged/mate
   poses are needed.
3. Select the carry strategy using the criteria above.
4. Execute the returned `reorientation_plan` with
   `executing-held-object-motion`.
5. Execute `placement_plan` with `executing-feature-mating`.

Waypoint modes are semantic: `contact_transition` leaves the initial support,
`planned_joint` is collision-aware free-space motion, `planned_linear` locks
orientation for a straight local leg, `cartesian_cross` is a tracked short
crossing, and `contact_seat` is the final intentional-contact servo.

## Boundaries

- Upstream perception supplies the held feature in TCP coordinates and the
  fixture pose/axis. Registration supplies attached-object collision geometry.
- Inputs describe functional geometry and a relation rather than an object
  class or task name; the graph chooses `strategy`, `stage_outward_m` and the
  re-observation descriptions.
- Free roll about the fixture axis is treated as symmetry. Candidates are
  checked for reachability and the smallest feasible TCP rotation is selected.
- `support_normal` may be supplied when the escape direction is known; it
  defaults to world up for a horizontal support surface.
- Relationship-specific clearances may be supplied by the graph when fixture
  depth is observable or specified. Conservative geometric defaults are used
  otherwise.
- The final crossing is linear and may allow goal contact. All earlier motion
  remains collision-aware free-space motion.
