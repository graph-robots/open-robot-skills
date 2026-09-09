---
name: constructing-collision-worlds
description: Construct a collision-checking world from calibrated RGB-D views, exclude robot and manipulated-object pixels, remove reconstruction artifacts, and optionally carve a small intentional-contact corridor. Use before collision-aware grasp, carry, insertion, hanging, or placement planning. Do not use for semantic object localization or for attached-object geometry.
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [collision, rgb-d, reconstruction, planning]}
gap:
  allowed_tools: [sam3.segment_text, geometry.build_world_config]
  required_inputs: {target_mask: Mask, target_kind: string}
  produces_outputs: {world_config: WorldConfig, strategy: string}
  exit_conditions:
    built: A configured collision world was constructed, or the declared strategy explicitly disabled it.
    failed: Reconstruction or filtering could not produce the requested world.
  canonical_scripts:
    - build_collision_world: scripts/build_collision_world.py
    - filter_mesh_components: scripts/filter_mesh_components.py
  streaming: false
---

# Constructing Collision Worlds

Build conservative scene collision geometry without turning RGB-D artifacts or
the intended contact target into impossible obstacles.

## Inputs and outputs

Require calibrated RGB-D `cameras` when the selected strategy is `rgbd_mesh`.
Accept manipulated/fixture masks, a semantic target description, fixture
feature geometry, and a declarative `collision_profile`. A `disabled` strategy
returns an empty world only when downstream execution has explicitly declared
that it will not consume collision geometry.

Return `world_config` and retained `mesh_names`. The world must contain at least
one valid mesh for `rgbd_mesh`.

## Collision profile

Profiles select behavior by capability, never by matching an object/task name:

- `excluded_masks`: whether to exclude the target, its cross-view semantic
  masks, robot pixels, and an explicitly allowed fixture-contact mask;
- `fixture_keep_out`: optional box geometry, robot-model spheres, and margins;
- `approach_corridor`: an axis sign, axial interval, and radius for a narrow
  intentional-contact tube terminating at the perceived fixture feature;
- `allowed_contact_surfaces`: names of fixture masks or declared keep-out
  components that should not be treated as obstacles;
- `obstacle_filter`: metric artifact thresholds and valid workspace bounds;
- `reconstruction`: voxel, clustering, and meshing parameters.

Task graphs may hold a list keyed by a semantic `source_kind`, but all entries
use the same schema and canonical algorithm. The key selects data; it must not
select an object-specific code branch.

## Procedure

1. Mask the manipulated object in every view where it is visible. Masking only
   one camera leaves a duplicate static obstacle in other views.
2. Exclude robot geometry using model collision spheres when available; visual
   robot masks are only a fallback for visible links.
3. Call `geometry.build_world_config` with metric voxel/noise parameters.
4. Run `scripts/filter_mesh_components.py`. Remove only components that satisfy
   explicit artifact evidence; never remove all small or all thin geometry.
5. If the goal intentionally intersects a fixture, carve a narrow tube ending
   at the perceived feature. Preserve the rest of the supporting surface.

The canonical artifact rules remove:

- sparse depth-edge sheets: fewer than 32 vertices, thinner than 4 mm, and
  smaller than 150 mm on their longest axis;
- compact sparse islands: fewer than 32 vertices and smaller than 60 mm on
  every axis;
- unmistakable invalid-depth components: entirely below the configured valid
  workspace floor, or larger than the configured maximum scene span.

Keep real thin fixtures, boards, racks, tools, and compact clutter when they do
not meet the complete artifact predicate. See
[`canonical_subgraph.json`](examples/canonical_subgraph.json).

Task graphs may tune thresholds from known workspace bounds, but must expose
those values as parameters rather than embedding object names or image pixels.
