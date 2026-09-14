---
name: reconstructing-collision-worlds
description: Reconstructs a planner collision world from calibrated RGB-D views alone, excluding the grasp target, the robot, and a perceived fixture mask, dropping depth-edge slivers and invalid-depth sheets, and carving the free space an intentional-contact goal needs as an approach tube along a fixture axis or a corridor through a lid aperture. Use when a held-object motion must be planned against obstacles no simulator or CAD scene supplies, before registering the object and planning the carry.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [collision-world, rgb-d, motion-planning, reconstruction]}
gap:
  requires: {connector: [motion.get_robot_collision_spheres, motion.build_world_tsdf]}
  allowed_tools:
    - sam3.segment_text
    - motion.get_robot_collision_spheres
    - motion.build_world_tsdf
    - geometry.build_world_config
  required_inputs:
    observation: Observation
    target_mask: Mask
  produces_outputs:
    world_config: WorldConfig
    strategy: string
  exit_conditions:
    built: A scene mesh set (or, in TSDF mode, a backend voxel grid) with the goal's free space carved or cleared was produced.
    failed: No camera was selected or the reconstruction produced no scene mesh (raised).
  canonical_scripts:
    - build_collision_world: scripts/build_collision_world.py
    - filter_mesh_components: scripts/filter_mesh_components.py
  streaming: false
---

# reconstructing-collision-worlds

Run once after perception and before grasping, with the same observation
the perception skills used. `build_collision_world` merges depth from the
selected views (`camera_names`, comma-separated; empty = all) into
alpha-shape meshes through `geometry.build_world_config` and returns a
`WorldConfig` plus the mesh names.

## What is excluded

- The grasp target by its perceived `target_mask` in `mask_camera_name`
  (default `overhead`), and by segmenting `target_description` in every
  other selected view so the object does not become a static obstacle where
  it later rotates in the hand.
- The robot, by text segmentation per view and by
  `motion.get_robot_collision_spheres` when the connector provides it (a
  connector without it falls back to the visual masks).
- A perceived `fixture_mask` from the mask camera, unless it is a full-frame
  placeholder that would erase the world.

## What is carved

- An approach tube of 55 mm radius along `outward_sign * fixture_axis`
  ending 12 mm behind `fixture_tip`, when both are given. Use
  `outward_sign: -1` for an axis that points into the fixture (an aperture's
  inward axis) and `1` for one that points toward its free side (a shaft).
- A vertical corridor of `corridor_radius` through `corridor_center` between
  `corridor_rim_z - 35 mm` and `+180 mm`, when the centre and a positive
  radius are given. The radius is the caller's measurement (for example a
  fraction of the perceived aperture radius); the script adds no margin.

## Cleaning and packing

Sparse, nearly zero-thickness sheets under 15 cm are dropped; components
lying entirely below `sheet_floor_z` or spanning more than
`sheet_max_extent_m` are invalid-depth returns and are dropped too (0
disables either gate). `voxel_size` trades surface ripple for detail.
`pack_meshes: true` folds every component into one `perceived_scene` mesh
for planners whose scene cache admits few named obstacles.

Values the RoboSimStudio sharps graph ran with (sweep s8), all existing
parameters: one camera (`camera_names: overhead`), `voxel_size: 0.016` (coarser
than the 8 mm default, to drop RGB-D surface ripple and sliver triangles a
planner reads as narrow collisions), `corridor_radius: max(5 mm, 0.85 *
aperture radius)`, `pack_meshes: true` (cuRobo's scene cache admits 32 meshes).

## TSDF mode (opt-in)

`tsdf: true` (or a collision profile with `strategy: rgbd_tsdf`) replaces the
alpha-shape meshes with `motion.build_world_tsdf`: depth is integrated into an
ESDF voxel grid on the motion backend, and the script returns
`world_config: {use_tsdf: true, meshes: [<declared keep-out boxes>], tsdf:
<summary>}` with `strategy: rgbd_tsdf`. The same target, robot and fixture
masks are sent as `exclude_masks`; the backend removes the robot's own body by
its sphere model, so no `motion.get_robot_collision_spheres` call is made.

- **Connector.** `motion.build_world_tsdf` is a runtime-connector tool, not part
  of this library's `tools/curobo` bundle: RoboSimStudio's motion connector
  provides it. Downstream planners must accept a `use_tsdf` world config (the
  grid is not in the returned dict; the planner loads the one the last build
  left on its backend).
- **Corridor.** A positive `corridor_radius` at `corridor_center` becomes one
  cleared vertical cylinder of radius `max(tsdf_clear_radius_min_m (0.05),
  tsdf_clear_radius_factor (1.1) * corridor_radius)` from `corridor_rim_z -
  tsdf_clear_below_rim_m (0.20)` to `+ tsdf_clear_above_rim_m (0.25)`: it spans
  the hand's descent and reaches down through the cavity, because a TSDF reads
  unseen interior voxels as near-occupied where a mesh world leaves them free.
  The graph's factor applied to the raw aperture radius, not to the 0.85
  fraction it carved meshes with. `tsdf_voxel_size` defaults to 0.01: at 20 mm
  the ESDF's one-voxel fattening sealed the aperture.
- **Fallback.** When the tool is absent or raises, when more than one camera is
  selected (the tool applies every mask to every same-shaped depth image), or
  when an approach tube is requested (not a vertical cylinder), the script
  prints why and builds the mesh world exactly as without `tsdf`.
- A profile's `tsdf` block overrides the tunables by `voxel_size_m`,
  `clear_radius_factor`, `clear_radius_min_m`, `clear_below_rim_m`,
  `clear_above_rim_m`.

What this fold added, from RoboSimStudio `sharps_disposal/gap_perception_v2`
(sweep s8, 16/30 trials, 152/176 syringes): the TSDF mode and its fallback
above. The graph measured 42-44 s per episode for the CPU mesh path on the
overhead camera. Mesh-path defaults are unchanged.

## Boundaries

- No simulator state, CAD model, or fixed world coordinate is read.
- The script raises when no mesh survives; the graph routes that to
  `failed`. A TSDF build never raises for that reason; its failures fall back
  to the mesh path.
