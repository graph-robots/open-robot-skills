---
name: registering-held-objects
description: Reobserves a grasped rigid object, estimates its functional feature in the TCP frame, and constructs attached collision geometry. Use when downstream mating or collision-aware transport needs a rigid held-object transform with confidence and uncertainty.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [in-hand, registration, wrist-camera, collision]}
gap:
  allowed_tools: [robot.get_ee_pose, sam3.segment_text, geometry.mask_to_world_points, geometry.fit_planar_feature, geometry.cloud_to_attachment]
  required_inputs: {reference_cloud: PointCloud, functional_feature: Se3Pose, object_description: string}
  produces_outputs: {feature_in_tcp: Se3Pose, object_in_tcp: Se3Pose, attached_object: AttachedObject, registration_confidence: float, registration_method: string, fallback_used: bool, translation_uncertainty_m: float}
  exit_conditions:
    registered: A reliable transform was measured.
    fallback: The grasp-time transform was retained.
    lost: The object is no longer localized.
  canonical_scripts:
    - register_held: scripts/register_held.py
  streaming: false
---

# registering-held-objects

Use immediately after grasping or again at a pre-contact pose. For pre-contact
realignment, pass the prior feature-in-TCP and attachment: wrist views localize
the functional feature directly, fit a loop's geometric center and plane rather
than the centroid of its visible arc, and retain the collision model.
Grasp-time geometry remains an explicit
low-confidence fallback.

The functional feature may carry a declarative `registration_profile` with a
strategy, semantic query, optional CAD model/scale/landmarks/symmetry,
candidate-distance gates, rigid TCP jump limits, and uncertainty values.
Implementations dispatch on method names such as `semantic_feature`,
`planar_cad_landmark`, and `cad_distal_loop`; they must never select an
algorithm by matching an object/task name or by recognizing one asset's size.

`cad_distal_loop` is a general estimator for a planar rigid object whose
functional loop is a declared CAD landmark. `planar_cad_landmark` is the
corresponding estimator for any flat object with one of several semantic
landmarks. Both use the same generic policy: prefer the active wrist, use a
remote wrist only when the active view is occluded, gate candidates by their
distance to the active TCP, refine against declared CAD, and reject non-rigid
feature jumps. The implementation contains no dispatch on semantic object
names.

If a held loop is self-occluded in both wrist views, the skill may use the
declarative `overview_loop_fallback`. It segments loop candidates in the
declared overview cameras, back-projects them to metric 3D, and associates the
nearest candidate to the rigid feature prediction within
`overview_feature_jump_m`. Only the observed loop centre updates the rigid
transform; orientation remains the prior because a distant partial circle does
not reliably constrain plane normal or roll. This association also prevents a
previously placed object from being selected in multi-object scenes.

Return `registration_method`, `fallback_used`, and
`translation_uncertainty_m` together with confidence. A rejected wrist
measurement preserves the rigid grasp prior rather than silently changing the
feature-to-TCP transform.

The default `attachment_fit_type="morphit"` fits 64 collision spheres to a
watertight convex hull of the observed object and contracts their radii by 2
mm. This is the standard CuRobo attachment representation for both transport
and constrained fixture motion. `surface` and `voxel` remain available only
for explicit fitting experiments; ordinary workflows should keep the default.
