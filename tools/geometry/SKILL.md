---
name: geometry
description: Pure-math 3D geometry toolbox — back-project masks and depth to
  point clouds, DBSCAN-filter noise, fit oriented bounding boxes, derive
  top-down/front grasp poses, and reconstruct collision worlds from RGB-D
  frames. Use when a workflow needs perception geometry or planner inputs
  computed on CPU with no model weights.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, planning, geometry]}
gap:
  requires: {}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "geometry"]
    protocol: stdio-msgpack
  tools:
    - geometry.depth_to_point_cloud: Convert a metric depth image to a camera-frame point cloud.
    - geometry.mask_to_world_points: Back-project a 2D mask to 3D world points via depth + calibration.
    - geometry.pixel_to_world_point: Back-project a single pixel to a 3D world point.
    - geometry.transform_points: Apply a rigid SE(3) transform to 3D points.
    - geometry.exclude_robot_points: FK-sphere removal of robot-body points from a cloud (7-DOF Franka; other arms pass through).
    - geometry.filter_noise: DBSCAN noise filtering (keeps all non-noise points).
    - geometry.compute_obb: Fit an upright oriented bounding box to 3D points (HyRL fit).
    - geometry.filter_and_compute_obb: DBSCAN filter then OBB fit in one call.
    - geometry.top_down_grasp_from_obb: One world-aligned top-down grasp pose from an OBB.
    - geometry.top_down_grasp_candidates: Fan of top-down grasp candidates (canonical primary+alt first).
    - geometry.select_top_down_grasp: Pick the most top-down grasp from candidates.
    - geometry.front_grasp_from_obb: Front-approach grasp/pre-grasp for handles (drawers, doors).
    - geometry.build_world_config: Reconstruct a collision world (alpha-shape scene mesh) from RGB-D.
    - geometry.rotate_quat_z90: Rotate a wxyz quaternion 90 degrees around world Z.
    - geometry.compute_drop_position: Drop position above a container OBB.
    - geometry.compute_xy_distance: XY-plane distance between two 3D points.
    - geometry.iou: IoU of two axis-aligned 2D boxes.
    - geometry.pose_distance: Euclidean distance between two 3D positions.
---

# geometry

Pure-math perception/planning geometry as in-process typed tools, from
mask back-projection through OBB fitting to grasp-candidate generation,
plus the two scalar helpers (`geometry.iou`, `geometry.pose_distance`).
Fully CPU — no model weights, no GPU.

## When to use

- Turning a segmentation mask + depth + camera calibration into world-frame
  points (`mask_to_world_points`) and an object OBB
  (`filter_and_compute_obb`).
- Deriving grasp poses from an OBB: `top_down_grasp_candidates` for tabletop
  pick (feed the full list to `curobo.plan_to_grasp_poses` as a goalset),
  `front_grasp_from_obb` for horizontal interactions (drawer/door handles).
- Building the collision world for the planner: `build_world_config` with the
  target's mask in `object_masks` so the planner can `ignore_obstacle_names`
  it.

## Install

```bash
uv sync --extra geometry   # open3d + scikit-learn (cv2/scipy come with gap core)
# (pip: pip install -e ".[geometry]")
```

The module imports lazily — the bundle loads (and the light tools work)
without the extra; only OBB fitting, DBSCAN filtering and world
reconstruction need open3d/sklearn/cv2.

## Gotchas (carried over from the service)

- **OBB `extent` is HALF-extents** (gap.types convention, same as the proto).
  `compute_obb` is upright-only: rotation is around world Z (no 3D tilt), and
  extents use the 2nd/98th percentile of points, not strict min/max.
- Single-camera clouds are 2.5D: only camera-facing surfaces are observed, so
  OBB centers carry a few cm of depth bias on opaque objects. (The service's
  rehearsal-sandbox ground-truth snap that compensated for this in-container
  was deliberately NOT ported — it depended on a `/app` sandbox file.)
- `top_down_grasp_candidates` default `z_offset=-0.04`: fingertip 4 cm below
  the OBB top. With `z_offset=0.0` the fingers close above the object
  (silent empty grip). Grasp Z is clamped to -0.05 m (table-clearance floor;
  LIBERO table top is at world z=0).
- `mask_to_world_points` keeps only depths in [0.015, 20.0] m (HyRL bounds);
  invalid/zero-depth pixels are dropped.
- `filter_noise` returns the ORIGINAL cloud unchanged when DBSCAN labels
  everything noise (defensive fallback, mirrors HyRL).
- `build_world_config`: table removal only runs when `table_z_threshold != 0`
  (typical -0.01); robot-point exclusion is Franka-only (simplified DH FK)
  and skips non-7-DOF joint states; prefer explicit `object_masks` over the
  `target_obb` projection fallback — masks are pixel-accurate, the OBB
  projection is a corner-AABB approximation inflated by 2 cm.
- `top_down_grasp_from_obb` yaw is NOT derived from the OBB — fingers may
  close across the wide axis; use the candidate fan when orientation matters.
