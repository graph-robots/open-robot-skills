---
name: registering-held-objects
description: Reobserves a grasped rigid object from the wrist cameras, estimates its functional feature in the TCP frame, and fits attached collision spheres, retaining the grasp-time transform at low confidence when the object is not seen. Use when a held object must be localized in the hand right after grasping or re-checked at a pre-contact pose before a fixture engagement.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [in-hand, registration, wrist-camera, collision]}
gap:
  allowed_tools: [robot.get_ee_pose, sam3.segment_text, geometry.mask_to_world_points, geometry.fit_planar_feature, curobo.cloud_to_attachment, geometry.cloud_to_attachment]
  required_inputs: {reference_cloud: PointCloud, functional_feature: FunctionalFeature, object_description: string}
  produces_outputs: {feature_in_tcp: Se3Pose, object_in_tcp: Se3Pose, attached_object: AttachedObject, registration_confidence: float, registration_method: string, fallback_used: bool, translation_uncertainty_m: float}
  exit_conditions:
    registered: A feature-in-TCP transform and attachment were produced; registration_confidence is the accepted mask score, or 0.25 when the grasp-time or prior transform was retained.
    lost: Registration raised — no cloud could be fitted into an attachment, a geometry fit failed, or require_observed was set and nothing was observed, so the object is no longer localized.
  canonical_scripts:
    - register_held: scripts/register_held.py
  streaming: false
---

# registering-held-objects

`cameras` is a list of `CameraFrame` -- an `Observation`'s `cameras` field.
It is stated here rather than under `required_inputs` because the type
registry names no bare list of frames.

Use immediately after grasping or again at a pre-contact pose. For pre-contact
realignment, pass the prior feature-in-TCP and attachment: wrist views localize
the functional feature directly, fit a loop's geometric center and plane rather
than the centroid of its visible arc, and retain the collision model.
Grasp-time geometry remains an explicit low-confidence fallback
(`registration_confidence == 0.25`); the script does not raise for it, so a
graph that must stop on a weak registration routes on the confidence.

## How the feature is measured

- A loop is segmented from `functional_feature.description` on every pass and
  its plane and circle centre are fitted. Only the minimum rotation that aligns
  the prior normal with the fitted normal is applied, so the in-plane roll of
  the prior (or grasp-time) frame is preserved: a circle has no observable
  roll and the fit's roll changes between cameras. The coordinate along the
  normal is the robust midpoint of the observed ring thickness.
- A tip is segmented from `object_description`. With
  `direction_marker_description` set (a cap, a coloured band, a head) and both
  masks visible in the same wrist view, the tip is re-derived as the endpoint
  of the object's principal axis directed toward the marker; with a prior, the
  prior roll about that axis is preserved. Without a marker the cloud median is
  compared with the predicted feature centre.
- With a prior, an observed centre more than 4 cm from the prediction is a
  mask on the gripper or background and is rejected (confidence 0.25).
- On the first pass for non-loop features, an observed cloud whose 3D extent
  is outside 0.55--1.80 of the reference cloud's extent is rejected before it
  becomes an enormous attachment or clearance waypoint (confidence 0.25).

## Grasp transform and attachment

- `grasp_pose` (optional) is the commanded grasp TCP pose. When given, the
  reference cloud and the pre-grasp feature pose are carried by the rigid
  transform current-TCP · inv(grasp_pose), so the fallback feature-in-hand is
  inv(grasp_pose) · feature_world rather than inv(current TCP) · feature_world.
  Pass it whenever registration happens after a lift or a verification move.
- `attachment_source` selects the cloud that bounds the collision model when
  no `prior_attached_object` is given: `"reference"` (default) fits the
  complete pre-grasp cloud, carried as above; `"observed"` fits the accepted
  wrist cloud, falling back to the carried reference cloud when the wrist saw
  nothing usable. `object_in_tcp` is centred on the same cloud.
- The attachment is `curobo.cloud_to_attachment(surface_radius=0.002,
  margin=0.002, max_spheres=64)`: 64 MORPHIT spheres on a watertight convex
  hull, radii contracted by 2 mm, fitted where cuRobo lives. An
  `attachment_fit_type` of `surface` or `voxel` goes to
  `geometry.cloud_to_attachment` instead, for explicit fitting experiments
  only.
- `camera_name_filter` (default `"eye_in_hand"`) is a substring the camera
  name must contain; set it to another camera name when a held object is
  better observed from an overview camera.

## Verifying the hold in clutter (opt-in)

Four parameters were folded in from RoboSimStudio's
`sharps_disposal/gap_perception_v2` (`map_syringe_to_hand.py`, sweep s8). There
a syringe lifted by a Robotiq hand has to be told apart from 3-7 look-alikes
still in the tray. Each default is the behaviour above, and
`tests/test_sharps_v2_perception.py` checks the defaults call-for-call against
the script before the fold. The graph's own in-process cuRobo sphere fit was
not ported: the attachment is still `curobo.cloud_to_attachment`. The fold
added nothing else.

- `held_candidate_cameras` (None). Set it to a list of exact camera names to
  replace the top-mask-per-camera search. Every mask from those cameras
  (`held_max_results`, 0 = all) that scores at least `held_score_min` (0.05)
  is back-projected. The cloud whose median is nearest the TCP, within
  `held_max_distance_m` (0.30), is the observation, and its score is the
  confidence. `held_extent_m` (None) first trims points farther than that from
  the cloud's median. `rod_length_range_m` (None) keeps only a cloud whose
  5th-95th percentile length along its principal axis is in range and at least
  `rod_aspect_min` (3.0) times its width. With a spare syringe 90 mm away the
  two scored within 0.01, and with 6+ in the tray the held one was not in
  SAM's top 8.
- `presence_only` (False). The accepted observation confirms the object
  moved with the hand but never shifts the carried feature: no median
  correction, no marker tip re-derivation. `registration_method` is
  `rigid_grasp_prior` and the confidence is still the mask score. A partial
  post-lift mask moved the syringe's registered tip from 46-65 mm to 9-18 mm
  along the shaft. It applies to the generic path only, so a
  `registration_profile` with it raises.
- `snap_feature_axis_deg` (0 = off). When the feature's z in TCP is within
  this many degrees of the hand's +-x or +-y axis, rotate it exactly onto
  that axis. A top-down pinch across a cylinder lays it along a jaw axis. The
  true barrel sat 0.3-1.5 deg off, while the silhouette's SVD pitched it up to
  4.3 deg and one syringe struck the lid.
- `require_observed` (False). With nothing accepted, raise (exit `lost`)
  instead of retaining the grasp-time transform at 0.25.

Syringe example (the graph's values): `held_candidate_cameras: [eye_in_hand,
agentview, overhead]`, `held_score_min: 0.005`, `held_max_distance_m: 0.12`,
`held_extent_m: 0.10`, `rod_length_range_m: [0.075, 0.18]`, `rod_aspect_min:
3.0`, `presence_only: true`, `snap_feature_axis_deg: 15`, `require_observed:
true`, `grasp_pose` bound, `attachment_source: reference`.

Object-specific refinement (CAD registration, per-object landmarks) is not
part of this skill; a graph performs it in its own node and passes the result
as `prior_feature_in_tcp` / `prior_object_in_tcp`.
