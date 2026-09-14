---
name: perceiving-functional-features
description: Locates a language-described functional part of a rigid object or fixture from calibrated RGB-D and fits a typed loop, shaft, tip, aperture, surface, or region feature with a metric centre, axis, and radius, either inside an already-segmented parent or end to end from a described object, protruding shaft, directed tip, or lidded aperture. Use when a manipulation step depends on where a specific part is rather than on the whole object, such as a tool loop to hang, a shaft to hang it on, an insertion tip, or an opening to insert into.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, affordance, geometry, rgb-d]}
gap:
  allowed_tools:
    - sam3.segment_text
    - sam3.segment_box
    - grounding-dino.detect
    - geometry.mask_to_world_points
    - geometry.filter_and_compute_obb
    - geometry.fit_planar_feature
    - geometry.fit_linear_feature
  exit_conditions:
    found: One geometrically valid feature was fitted and its centre, axis, and radius are bound.
    not_found: No described object or feature passed the segmentation and geometry gates (raised by every script, or returned as `route` by `perceive_protruding_shaft` when `required` is false).
  canonical_scripts:
    - perceive_feature: scripts/perceive_feature.py
    - perceive_object_feature: scripts/perceive_object_feature.py
    - perceive_protruding_shaft: scripts/perceive_protruding_shaft.py
    - perceive_directed_tip: scripts/perceive_directed_tip.py
    - perceive_aperture: scripts/perceive_aperture.py
    - perceive_fixture_feature: scripts/perceive_fixture_feature.py
  streaming: false
---

# perceiving-functional-features

Use when manipulation depends on a loop, shaft, tip, aperture, surface, or
region rather than on a whole object. Every script localizes geometry from a
calibrated RGB-D camera and returns a `FunctionalFeature` (`kind`, `pose`
whose local Z is the feature axis, `axis`, `confidence`, and the radius or
length the mating skill needs). None of them computes mating poses or moves
the robot.

Four of the scripts take an `Observation` and pick one camera by name
(`camera_name`, default `overhead`). `perceive_feature` instead takes
`cameras` -- the `Observation`'s `cameras` list -- because it searches
every view; the type registry names no bare list of frames, so the shape is
stated here rather than under `required_inputs`.

## Which script

- `perceive_feature` -- the part of an **already segmented** parent. Inputs
  `parent_cloud`, `parent_mask`, `feature_description`, `feature_type`
  (loop, aperture, surface, region, shaft, tip). Candidates outside the
  parent's 3D bounds are rejected. Outputs `feature`, `feature_mask`,
  `feature_cloud`.
- `perceive_object_feature` -- **object and its planar feature end to
  end**. `candidates` is a JSON list of rows `{kind, object_description,
  feature_description, feature_type, feature_score_min}`; each object
  description is segmented, the strongest wins, its cloud is optionally
  completed down to the support surface (`complete_to_support`), and the
  row's feature (when described) is segmented, centred, and fitted as a
  planar loop with the camera position fixing the normal sign. When
  `instruction` names one of the candidate kinds only those rows are tried.
  Outputs `target_kind`, `target_obb`, `target_mask`, `target_cloud`,
  `feature_center`, `feature_pose`, `functional_feature`. A row whose
  `feature_description` is empty yields the OBB centre as a point feature
  for a downstream landmark rule to refine; object-specific landmark rules
  belong to the graph, not to this script.
- `perceive_protruding_shaft` -- a **thin shaft protruding from a mounting
  plane** (a hook, peg, or pin). Detector boxes for `fixture_description`
  are gated by `label_keyword` and `aspect_min`, refined by SAM3, then the
  full protrusion is recovered from depth inside the box with the mounting
  plane as the far-X plane of the ROI (the shaft protrudes toward
  decreasing world X). `shaft_length_min/max` and `transverse_max` reject
  look-alikes. Outputs `fixture_obb/mask/cloud`, `shaft_tip`,
  `fixture_axis` (plane to tip), and `fixture_feature` with `radius_outer`,
  `usable_length`, and `seating_margin`. With `required: false` a miss
  returns `route: not_found` and empty outputs instead of raising.
- `perceive_directed_tip` -- an **elongated object's insertion tip**. The
  long axis is directed by a visible marker at one end
  (`direction_marker_description`, optional) or by the narrower end; the tip
  is the cloud's extreme along that axis. Outputs `target_obb/mask/cloud`,
  `insertion_pose` (local Z through the tip), `marker_center`, `support_z`
  (the cloud's 5th-percentile z unless `support_ring_px` reads it from depth
  beside the mask), `jaw_clearance_m` (NaN unless `jaw_clearance_free_m` is
  set), and a `tip` `functional_feature` with `radius_outer`. Everything in
  [Directed tip in clutter](#directed-tip-in-clutter-opt-in) is opt-in.
- `perceive_aperture` -- an **opening in a lid**. The lid plane is fitted
  from the fixture cloud; the aperture centre and radius come from
  intersecting calibrated rays through the opening's 2D mask with that plane,
  since depth inside a hole is the interior, not the rim. Outputs
  `aperture_center`, `fixture_axis` (inward), `aperture_radius`, `rim_z`,
  the two masks and clouds, and an `aperture` `fixture_feature` with
  `radius_inner`.

## Directed tip in clutter (opt-in)

`perceive_directed_tip` takes SAM's top mask and the detected marker by
default. The parameters below were folded in from RoboSimStudio's
`sharps_disposal/gap_perception_v2` (`perceive_target.py`, the graph behind
sweep s8: 16/30 cluttered trays fully solved, 152/176 syringes posted), where
the object is one of 4-8 syringes lying in a tray and the marker is the red
needle cap. Every default reproduces the script as it was, down to the tool
calls, and `tests/test_sharps_v2_perception.py` checks that. The fold added
these options and nothing it did not. The graph's constants are parameters
here, not defaults. Their values are in the right-hand column as the syringe
example.

| Parameter (default) | What it does | Syringe graph |
| --- | --- | --- |
| `max_results` (3) | SAM cap for the object and marker queries; `0` returns every mask. | `0` |
| `candidate_score_floor` (None = top mask) | Keep every mask at or above the floor, drop the lower-scored of two masks overlapping by `candidate_overlap_max` (0.3) of the smaller, then rank. At least one survivor must score `object_score_min` or hold marker-coloured pixels. | `0.02` |
| `min_mask_pixels` (0), `min_span_m` (0), `max_span_m` (None), `max_width_m` (None) | Size gates. Span is the 3rd-97th percentile along the principal axis; width is twice the 90th-percentile radial distance. | `500`, `0.08`, `0.20`, `0.045` |
| `exclude_center` (None), `exclude_radius_m` (0), `max_z` (None) | Reject a candidate whose median lies within the radius (xy) or above `max_z`. A detected marker inside the radius is ignored too. | the aperture centre, `0.12`, `rim_z + 0.02` |
| `min_height_above_floor_m` (None) | Reject a candidate whose 90th-percentile z is less than this above the 10th percentile of a depth ring `depth_ring_gap_px`-`depth_ring_outer_px` (6-60) px outside its mask. | `0.005` |
| `jaw_clearance_free_m` (None) | Read the free distance across the axis to the nearest raised surface (`protrusion_m`, 0.004, above the same ring's floor). Skip points within `own_body_half_width_m` (0) of the axis. Only count points within `clearance_half_length_m` (None = the whole length) along it. Report `no_neighbour_clearance_m` (0.08) when nothing is near. Rank on it. | `0.021`, with `jaw_clearance_enough_m` `0.045`, `own_body_half_width_m` `0.013`, `clearance_half_length_m` `0.035` |
| `marker_rgb_rule` (None) | A dict `{channel, min_value, min_margin, min_pixels, max_lateral_m, min_along_m}`. A pixel counts when its `channel` (0/1/2 = R/G/B) exceeds `min_value` and both other channels by `min_margin`. `min_pixels` such pixels within `max_lateral_m` of the axis and at least `min_along_m` from the centre give the marker. They also rank the candidate: a mask holding the colour off its axis goes last. | `{channel: 0, min_value: 110, min_margin: 40, min_pixels: 8, max_lateral_m: 0.010, min_along_m: 0.025}` |
| `marker_max_lateral_m` (None), `marker_max_gap_m` (None) | Ignore a detected marker whose median is farther than this from the axis, or from the nearest object point. | `0.012`, `0.025` |
| `width_landmark_to_marker_m` (None) | Before the narrow-end fallback, find the widest along-axis bin. It counts only when at least `width_landmark_ratio_min` (1.5) x the median bin width. Place the marker this far past it, toward the other end. | `0.088` (flange to cap) |
| `axis_from_svd` (False) | Principal axis signed toward the marker, instead of `marker - obb centre`. | `true` |
| `support_ring_px` (0), `support_ring_gap_px` (4) | `support_z` from the median depth of a ring this far outside the mask, clear of every detection scoring `object_score_min`. | `14`, `4` |

Rank order with several candidates: marker not off-axis, then clearance at or
above `jaw_clearance_free_m`, then `min(clearance, jaw_clearance_enough_m)`,
then SAM's score. The marker is the first of: marker-coloured pixels on the
axis, the detected marker (gated as above), the width landmark, the narrower
end. The measurements behind each gate are in the script's docstrings.

## Boundaries

- Descriptions and score floors are inputs; no object class, CAD landmark,
  or world coordinate is assumed. Object-specific refinements (CAD offsets,
  a handle-to-tip axis, the mating relation) are graph glue downstream.
- A script raises on a failed gate; the graph routes that to `not_found`.
  Only `perceive_protruding_shaft(required=false)` returns the route.
- This skill localizes geometry. Registration, mating poses, and motion are
  other skills.
