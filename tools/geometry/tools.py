"""geometry tool bundle — pure-math perception/planning geometry ops.

Each operation is exposed as a ``@tool`` function, including the two
scalar helpers ``geometry.iou`` / ``geometry.pose_distance``. The math
lives in ``_impl.py``; this module is the typed boundary: numpy arrays +
:mod:`gap.types` TypedDicts in and out.

No model, no GPU — everything here is CPU numpy/scipy/Open3D/sklearn/cv2.
Heavy optional imports (open3d, sklearn, cv2) happen inside the functions
that need them, so importing this module is always cheap.
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from gap.tools import tool
from gap.types import (
    CameraFrame,
    GraspCandidates,
    JointState,
    Mask,
    OrientedBoundingBox,
    PointCloud,
    Quaternion,
    Se3Pose,
    Vec3,
    WorldConfig,
    pose_to_matrix,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class PointCloudResult(TypedDict):
    points: PointCloud


class ObbResult(TypedDict):
    obb: OrientedBoundingBox


class PoseResult(TypedDict):
    pose: Se3Pose


class PointResult(TypedDict):
    point: Vec3


class PositionResult(TypedDict):
    position: Vec3


class QuatResult(TypedDict):
    quat: Quaternion


class DistanceResult(TypedDict):
    distance: float


class IouResult(TypedDict):
    iou: float


class GraspCandidatesResult(TypedDict):
    candidates: GraspCandidates


class FrontGraspResult(TypedDict):
    grasp_pose: Se3Pose
    pre_grasp_pose: Se3Pose
    approach_direction: Vec3
    slide_axis: Vec3


class ObjectMaskEntry(TypedDict):
    """Named segmentation mask for build_world_config."""

    name: str
    mask: Mask
    camera_index: int


class WorldConfigResult(TypedDict):
    config: WorldConfig
    mesh_names: list[str]


def _pc(points: np.ndarray) -> PointCloud:
    return {"points": np.asarray(points, dtype=np.float32).reshape(-1, 3)}


# ---------------------------------------------------------------------------
# Back-projection / transforms
# ---------------------------------------------------------------------------


@tool(
    name="geometry.depth_to_point_cloud",
    summary="Convert a metric depth image to a 3D point cloud in the camera frame.",
    tags=("perception",),
)
def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray) -> PointCloudResult:
    """Back-project ``depth`` (float32 [H, W], meters) through the pinhole
    ``intrinsics`` (float64 [3, 3]). Pixels with depth <= 0 are dropped."""
    from gap_skills.tools.geometry import _impl

    points = _impl._depth_to_points(
        np.asarray(depth, dtype=np.float32), np.asarray(intrinsics, dtype=np.float64)
    )
    return {"points": _pc(points)}


@tool(
    name="geometry.mask_to_world_points",
    summary="Back-project a 2D segmentation mask to 3D world points using depth + camera calibration.",
    tags=("perception",),
)
def mask_to_world_points(
    mask: Mask,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose,
) -> PointCloudResult:
    """Foreground pixels of ``mask`` (uint8 0/255 [H, W]) with valid depth in
    [0.015, 20.0] m (HyRL bounds) become world-frame points via the
    camera-to-world ``camera_pose``."""
    from gap_skills.tools.geometry import _impl

    points = _impl.mask_to_world_points(
        _impl.as_mask_bool(mask),
        np.asarray(depth, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float64),
        pose_to_matrix(camera_pose),
    )
    return {"points": _pc(points)}


@tool(
    name="geometry.pixel_to_world_point",
    summary="Back-project a single pixel to a 3D world point using depth + camera calibration.",
    tags=("perception",),
)
def pixel_to_world_point(
    pixel_x: float,
    pixel_y: float,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: Se3Pose,
) -> PointResult:
    """Raises ToolError when the pixel is out of bounds or has invalid depth."""
    from gap_skills.tools.geometry import _impl

    pt = _impl.pixel_to_world_point(
        pixel_x,
        pixel_y,
        np.asarray(depth, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float64),
        pose_to_matrix(camera_pose),
    )
    return {"point": _impl.vec3(pt)}


@tool(
    name="geometry.transform_points",
    summary="Apply a rigid SE(3) transform to a set of 3D points.",
    tags=("perception",),
)
def transform_points(points: PointCloud, transform: Se3Pose) -> PointCloudResult:
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    if len(pts) == 0:
        return {"points": _pc(pts)}
    out = _impl._transform_points(pts, pose_to_matrix(transform))
    return {"points": _pc(out)}


# ---------------------------------------------------------------------------
# Filtering + OBB fitting
# ---------------------------------------------------------------------------


@tool(
    name="geometry.exclude_robot_points",
    summary="Remove points near the robot body via FK-based sphere exclusion "
            "(7-DOF Franka; other arms pass through unchanged).",
    tags=("perception",),
)
def exclude_robot_points(
    points: PointCloud,
    joint_positions: JointState,
    distance_threshold: float = 0.05,
) -> PointCloudResult:
    """Strip robot-body points from a perception cloud (HyRL RobotSegmenter
    concept, simplified FK + link spheres). Essential when the perceived
    object sits against the robot base — the segmentation mask bleeds onto
    robot pixels and the merged cloud yields a wildly oversized OBB."""
    import numpy as np

    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    joints = np.asarray(joint_positions["positions"], dtype=np.float64).reshape(-1)
    if joints.shape[0] != 7:
        return {"points": _pc(pts)}
    return {
        "points": _pc(
            _impl._exclude_robot_points(pts, joints, distance_threshold)
        )
    }


@tool(
    name="geometry.filter_noise",
    summary="Filter point-cloud noise with DBSCAN clustering (keeps all non-noise points).",
    tags=("perception",),
)
def filter_noise(
    points: PointCloud,
    eps: float = 0.005,
    min_samples: int = 10,
) -> PointCloudResult:
    """Mirrors HyRL filter_noise: keeps ALL non-noise points (labels != -1),
    not just the largest cluster. If everything is classified as noise the
    original cloud is returned unchanged."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    return {"points": _pc(_impl.filter_noise(pts, eps, min_samples))}


@tool(
    name="geometry.compute_obb",
    summary="Fit an oriented bounding box to 3D points (HyRL contour-based min-width fit, upright in Z).",
    tags=("perception",),
)
def compute_obb(points: PointCloud) -> ObbResult:
    """Statistical outlier removal → XY rasterization → contour polygon →
    min-width rectangle search → 2nd/98th percentile extents. The returned
    OBB is upright (rotation only around world Z); ``extent`` holds
    HALF-extents per gap.types. Raises PerceptionFailed on < 4 points."""
    from gap_skills.tools.geometry import _impl

    return {"obb": _impl.compute_obb(_impl.as_points(points))}


@tool(
    name="geometry.filter_and_compute_obb",
    summary="DBSCAN-filter a point cloud then fit its oriented bounding box in one call.",
    tags=("perception",),
)
def filter_and_compute_obb(
    points: PointCloud,
    eps: float = 0.005,
    min_samples: int = 10,
) -> ObbResult:
    """Sequences geometry.filter_noise + geometry.compute_obb (the servicer
    offered this fusion to avoid two round trips; kept for workflow parity)."""
    from gap_skills.tools.geometry import _impl

    pts = _impl.as_points(points)
    filtered = _impl.filter_noise(pts, eps, min_samples)
    return {"obb": _impl.compute_obb(filtered)}


# ---------------------------------------------------------------------------
# Grasp-pose derivation
# ---------------------------------------------------------------------------


@tool(
    name="geometry.top_down_grasp_from_obb",
    summary="Compute a single world-aligned top-down grasp pose from an oriented bounding box.",
    tags=("planning",),
)
def top_down_grasp_from_obb(obb: OrientedBoundingBox, z_offset: float = 0.0) -> PoseResult:
    """Gripper points straight down world -Z above the OBB centre; Z lands on
    the world-frame top surface plus ``z_offset`` (negative = lower, typical
    -0.06 for bottles), clamped to 5 cm below the table plane."""
    from gap_skills.tools.geometry import _impl

    return {"pose": _impl.compute_top_down_grasp_world_aligned(obb, z_offset)}


@tool(
    name="geometry.top_down_grasp_candidates",
    summary="Fan out top-down grasp candidates (canonical primary+alt first, then 8 yaws x 3 depths, plus pitched side-grasps for flat boxes only).",
    tags=("planning",),
)
def top_down_grasp_candidates(
    obb: OrientedBoundingBox,
    z_offset: float = -0.04,
) -> GraspCandidatesResult:
    """poses[0]/poses[1] reproduce the legacy 2-pose RPC exactly; the rest are
    enriched candidates for a planner goalset. Default ``z_offset=-0.04``
    puts the fingertip 4 cm below the OBB top — with 0.0 the fingers close
    above the object (silent empty grip)."""
    from gap_skills.tools.geometry import _impl

    poses = _impl.top_down_grasp_candidates(obb, z_offset)
    return {"candidates": {"poses": poses}}


@tool(
    name="geometry.select_top_down_grasp",
    summary="Select the most top-down oriented grasp from candidates (gripper distance as tiebreaker).",
    tags=("planning",),
)
def select_top_down_grasp(
    grasp_poses: list[Se3Pose],
    gripper_position: Vec3 | None = None,
) -> PoseResult:
    from gap_skills.tools.geometry import _impl

    return {"pose": _impl.select_top_down_grasp(grasp_poses, gripper_position)}


@tool(
    name="geometry.front_grasp_from_obb",
    summary="Compute front-approach grasp + pre-grasp poses for a handle from its OBB (drawers, doors).",
    tags=("planning",),
)
def front_grasp_from_obb(
    obb: OrientedBoundingBox,
    approach_offset: float = 0.08,
    approach_hint: Vec3 | None = None,
    z_offset: float = 0.0,
) -> FrontGraspResult:
    """Derives approach direction and slide axis from the OBB orientation.
    ``approach_hint`` points from the handle toward the robot (default:
    OBB centre → origin, XY only). Raises PlanningFailed when the approach
    direction is near-vertical — use top_down_grasp_from_obb instead."""
    from gap_skills.tools.geometry import _impl

    out = _impl.front_grasp_from_obb(obb, approach_offset, approach_hint, z_offset)
    return {
        "grasp_pose": out["grasp_pose"],
        "pre_grasp_pose": out["pre_grasp_pose"],
        "approach_direction": out["approach_direction"],
        "slide_axis": out["slide_axis"],
    }


# ---------------------------------------------------------------------------
# World reconstruction
# ---------------------------------------------------------------------------


@tool(
    name="geometry.build_world_config",
    summary="Build a planner-agnostic collision world (alpha-shape scene mesh) from RGB-D camera frames.",
    tags=("planning",),
)
def build_world_config(
    cameras: list[CameraFrame],
    object_masks: list[ObjectMaskEntry] | None = None,
    voxel_size: float = 0.005,
    noise_eps: float = 0.01,
    noise_min_samples: int = 5,
    table_z_threshold: float = 0.0,
    mesh_alpha: float = 0.03,
    robot_joint_state: JointState | None = None,
    robot_distance_threshold: float = 0.15,
    robot_file: str = "franka.yml",
    target_obb: OrientedBoundingBox | None = None,
    target_obb_name: str = "target",
) -> WorldConfigResult:
    """Pipeline: depth → merged world cloud → voxel downsample → optional
    FK-based robot-point exclusion → DBSCAN largest-cluster filter → table
    removal (when ``table_z_threshold`` != 0; typical -0.01) → alpha-shape
    ``scene`` mesh, with ``object_masks`` (or a projected ``target_obb``)
    points excluded so planners can ignore the grasp target by name.

    ``robot_file`` is accepted for parity with the service request but the
    FK exclusion is Franka-only (simplified DH model); non-7-DOF joint
    states skip exclusion. Returns an empty WorldConfig if no geometry can
    be reconstructed."""
    from gap_skills.tools.geometry import _impl

    config, mesh_names = _impl.build_world_config(
        cameras,
        list(object_masks or []),
        voxel_size=voxel_size if voxel_size > 0 else 0.005,
        noise_eps=noise_eps if noise_eps > 0 else 0.01,
        noise_min_samples=noise_min_samples if noise_min_samples > 0 else 5,
        table_z_threshold=table_z_threshold,
        mesh_alpha=mesh_alpha if mesh_alpha > 0 else 0.03,
        robot_joint_state=robot_joint_state,
        robot_distance_threshold=(
            robot_distance_threshold if robot_distance_threshold > 0 else 0.15
        ),
        target_obb=target_obb,
        target_obb_name=target_obb_name,
    )
    return {"config": config, "mesh_names": mesh_names}


# ---------------------------------------------------------------------------
# Utility ops (merged from the geometry_utils skill, same as the servicer)
# ---------------------------------------------------------------------------


@tool(
    name="geometry.rotate_quat_z90",
    summary="Rotate a wxyz quaternion by 90 degrees around the world Z axis.",
    tags=("planning",),
)
def rotate_quat_z90(quat: Quaternion) -> QuatResult:
    s = math.sqrt(2.0) / 2.0
    zw, zx, zy, zz = s, 0.0, 0.0, s
    q = quat
    rw = q["w"] * zw - q["x"] * zx - q["y"] * zy - q["z"] * zz
    rx = q["w"] * zx + q["x"] * zw + q["y"] * zz - q["z"] * zy
    ry = q["w"] * zy - q["x"] * zz + q["y"] * zw + q["z"] * zx
    rz = q["w"] * zz + q["x"] * zy - q["y"] * zx + q["z"] * zw
    return {"quat": {"w": rw, "x": rx, "y": ry, "z": rz}}


@tool(
    name="geometry.compute_drop_position",
    summary="Compute a drop position above a container from its oriented bounding box.",
    tags=("planning",),
)
def compute_drop_position(
    container_obb: OrientedBoundingBox,
    clearance: float = 0.05,
    object_z_extent: float = 0.0,
) -> PositionResult:
    obb = container_obb
    clearance = clearance or 0.05
    obj_z = object_z_extent or 0.0
    c = obb["center"]
    e = obb["extent"]
    drop_z = c["z"] + e["z"] / 2.0 + obj_z + clearance
    return {"position": {"x": c["x"], "y": c["y"], "z": drop_z}}


@tool(
    name="geometry.compute_xy_distance",
    summary="Euclidean distance between two 3D points projected onto the XY plane.",
    tags=("perception",),
)
def compute_xy_distance(point_a: Vec3, point_b: Vec3) -> DistanceResult:
    a, b = point_a, point_b
    dist = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)
    return {"distance": dist}


# ---------------------------------------------------------------------------
# Legacy canary tools (ported from the dev tree's geometry_iou / pose_distance tools)
# ---------------------------------------------------------------------------


@tool(
    name="geometry.iou",
    summary="Compute IoU of two axis-aligned 2D boxes [x1, y1, x2, y2]. Returns 0 if boxes don't overlap.",
    tags=("perception",),
)
def iou(box_a: list[float], box_b: list[float]) -> IouResult:
    """Pure-Python intersection-over-union for axis-aligned 2D boxes.

    Args:
        box_a: ``[x1, y1, x2, y2]`` corners of box A.
        box_b: ``[x1, y1, x2, y2]`` corners of box B.

    Returns:
        ``{"iou": float}`` in ``[0, 1]``. Zero when the boxes don't overlap
        or either has zero area.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return {"iou": inter / union if union > 0 else 0.0}


@tool(
    name="geometry.pose_distance",
    summary="Euclidean distance between two 3D positions [x, y, z].",
    tags=("perception",),
)
def pose_distance(a: list[float], b: list[float]) -> DistanceResult:
    """Returns the Euclidean distance between two ``[x, y, z]`` points."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError(f"expected 3-vectors, got len(a)={len(a)}, len(b)={len(b)}")
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return {"distance": math.sqrt(dx * dx + dy * dy + dz * dz)}
