---
name: perceiving-functional-features
description: Locates a language-described functional part of a segmented rigid object or fixture and fits a typed loop, shaft, tip, aperture, surface, or region feature. Use when manipulation requires metric geometry for a specific object part rather than only a whole-object pose.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, affordance, geometry, rgb-d]}
gap:
  allowed_tools: [sam3.segment_text, sam3.segment_box, grounding-dino.detect, geometry.mask_to_world_points, geometry.fit_planar_feature, geometry.fit_linear_feature, geometry.filter_and_compute_obb]
  required_inputs: {parent_cloud: PointCloud, parent_mask: Mask, feature_description: string, feature_type: string}
  produces_outputs: {feature_mask: Mask, feature_cloud: PointCloud}
  exit_conditions:
    found: One geometrically valid feature was fitted.
    ambiguous: Multiple materially different candidates remain.
    not_found: No valid feature was visible.
  canonical_scripts:
    - perceive_feature: scripts/perceive_feature.py
    - perceive_object_feature: scripts/perceive_object_feature.py
    - perceive_fixture_feature: scripts/perceive_fixture_feature.py
  streaming: false
---

# perceiving-functional-features

Use after whole-object perception when manipulation depends on a loop, shaft,
tip, aperture, surface, or region. The caller supplies a natural-language part
description and the expected geometry family. The skill segments candidates,
fits the requested geometry, and validates the result against the parent
object's 3D bounds so a nearby object or background region is not accepted.

## When to use

- A manipulation target is a specific functional part rather than the entire
  object, such as a tool loop, insertion tip, peg, or aperture.
- Downstream mating or motion planning requires a metric center, axis, normal,
  or radius.

## Inputs

- `feature_description` names the visible part, for example `ring end of the
  tool` or `circular disposal opening`.
- `feature_type` selects one supported geometry family: loop, shaft, tip,
  aperture, surface, or region.
- `parent_cloud` and `parent_mask` constrain the search to the known object or
  fixture.
- `method` is a domain-neutral geometry strategy: `semantic_fit`,
  `parent_landmark`, `parent_distal_endpoint`, or `parent_inferred_aperture`.
- `feature_options` holds declared geometry such as relative landmarks,
  percentile bounds, expected axes, radii, and uncertainty. These values belong
  to the graph/task profile, not conditional branches on object names.

For workflows that must jointly select a parent instance, a grasp landmark,
and its manipulation feature, use `perceive_object_feature.py`. It dispatches
only on a declared geometry strategy: `planar_cad_loop`, `landmark_offsets`,
`distal_tip`, or `distal_planar_loop`. For fixtures, use
`perceive_fixture_feature.py` with `depth_support`,
`parent_inferred_aperture`, or `thin_projection`. Semantic category names are
profile keys used to select declarations; they never select algorithms.

An object profile may set `grasp_toward_feature_m` to move its grasp landmark
toward the perceived functional feature. `grasp_toward_feature_axes` selects
the displacement axes (for example `["x", "y"]` for an object resting on a
table), while `minimum_grasp_feature_separation_m` reserves clearance around
the feature. This is a category-independent way to shorten a carried object's
lever arm without hiding or gripping its mating geometry.

For repeated fixtures, a `depth_support` profile may declare
`anchor_fallback_partition: true`, `partition_axis`, and `partition_split`.
The metric anchor remains the primary estimate. Only when that narrow ROI
would return `not_found` does the skill search the matching workspace
partition. This supports source and destination layouts with different spacing
without encoding left/right object names or calibrated fixture coordinates in
the script.

These extended scripts return compatibility fields for generated manipulation
graphs (`target_obb`, `functional_feature`, `fixture_feature`) in addition to
the typed feature geometry. All CAD paths, landmarks, workspaces, scale gates,
queries, and mating metadata remain caller-owned profile data.

## Output quality

Every feature reports `confidence`, `method`, and `uncertainty_m` in addition
to its typed pose/axis/radius fields. Downstream mating code should consume the
typed feature and reject uncertainty outside its own tolerance; it should not
reconstruct a feature from task-specific world coordinates.

## Boundaries

- Use whole-object perception before this skill.
- Return `ambiguous` when materially different valid candidates remain; do not
  choose one using task-specific coordinates.
- This skill localizes geometry. It does not compute mating poses or execute
  motion.
- CAD landmarks are allowed only as optional geometry-model inputs. The script
  must dispatch by a declared method, never by a tool or fixture class name.
