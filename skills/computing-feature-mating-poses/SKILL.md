---
name: computing-feature-mating-poses
description: Computes desired approach and mate geometry from held and fixture features, resolving free mate symmetries from the current robot pose without executing motion. Use when a typed manipulation relation must become metric approach, engagement, and seated poses.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: planning, tags: [fixture, insertion, hanging, geometry]}
gap:
  allowed_tools: [robot.get_ee_pose, geometry.compute_feature_mate]
  required_inputs: {held_feature_in_tcp: Se3Pose, fixture_feature: Se3Pose, relation: string, attached_object: AttachedObject}
  produces_outputs: {mate_pose: Se3Pose, approach_pose: Se3Pose, engaged_pose: Se3Pose, approach_axis: Vec3, seating_distance: float, minimum_clearance: float}
  exit_conditions:
    computed: A valid mate was computed.
    ambiguous: Symmetry leaves different mates.
    infeasible: Feature dimensions cannot satisfy the relation.
  canonical_scripts:
    - compute_mate: scripts/compute_mate.py
  streaming: false
---

# computing-feature-mating-poses

This skill computes geometry only. When a relation has a free rotational
degree of freedom, it selects the equivalent mate closest to the current TCP
orientation to avoid arbitrary or unreachable reorientation. Its pre-contact
standoff encloses the complete attached-object sphere projection along the
fixture axis, preventing long held bodies from sweeping into the fixture.

## When to use

- A held loop or ring must be placed over a shaft, peg, or hook.
- A held shaft or tip must enter an aperture.
- The mating pose has rotational symmetry that should be resolved from the
  current robot pose.

Use `loop_over_shaft` for a loop or ring placed over a peg or hook. Use
`shaft_into_aperture` for a shaft entering an opening;
`tip_through_aperture` and `insert_through` are accepted aliases. The skill
checks that the relevant inner and outer dimensions have sufficient clearance.

For `loop_over_shaft`, describe the fixture at its distal tip with an axis
pointing outward from the mounting surface, `radius_outer`, and the observed
`usable_length` from tip toward the mount. The skill uses these dimensions to
center the shaft inside the loop during crossing, travel to a stable interior
shaft position, and then seat the loop without requiring an object-specific
target point.

For shaft/aperture relations, the fixture axis points into the opening and an
optional `insertion_depth` advances both engaged and mate poses along that
axis. The approach pose remains on the free side. This convention applies to
containers, sockets, holes, and other apertures without object-name branches.

An optional nested `mating_profile` on the fixture feature describes measured
execution compensation without naming an object class. Supported fields are
`settling_axis`, per-pose `crossing_offsets_m`, `seated_radial_offset_m`, and
`mate_settling_offset_m`. These are graph-owned fixture/compliance parameters;
the canonical algorithm never infers a profile from a radius or object name.

## Outputs

- `approach_pose` is a collision-free standoff before fixture engagement.
- `engaged_pose` crosses the distal tip far enough for the loop to surround the
  shaft.
- `mate_pose` places the loop farther along the usable shaft and seats it.
- `seating_distance` and `minimum_clearance` describe the computed mate.

## Boundaries

- This skill computes poses but does not plan or execute robot motion.
- Inputs describe functional geometry and a typed relation, not object names.
- Execute the result with `planning-held-object-motion` and the corresponding
  motion-execution skills.
