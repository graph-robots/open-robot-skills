"""RGB-D helpers for ``plan_linear_engagement``'s side-view leading-end read.

Minimal, generic versions of the helpers the sharps disposal graph
(``gap_perception_v2``) keeps in ``perception_checks.py``:
crop an image to a world box, segment inside the crop and restore each mask to
the calibrated full image before back-projection, and a rod-shape test. Inputs
are observations only.
"""

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    q, p = pose["rotation"], pose["position"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def crop_for_world_box(camera: dict[str, Any], lower, upper, padding: int = 16) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` covering a world-axis box, padded.

    Cropping enlarges a small held object in the image before segmentation;
    ``detect`` restores the masks to the full image, so the camera calibration
    stays valid for RGB-D back-projection.
    """
    corners = np.array([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])], dtype=np.float64)
    local = (np.linalg.inv(_matrix(camera["pose"])) @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    local = local[local[:, 2] > 0.01]
    if not len(local):
        raise ValueError("inspection region is behind the camera")
    uv = local @ np.asarray(camera["intrinsics"], dtype=np.float64).T
    uv = uv[:, :2] / uv[:, 2:]
    h, w = np.asarray(camera["rgb"]).shape[:2]
    x0, y0 = np.maximum(0, np.floor(uv.min(axis=0) - padding)).astype(int)
    x1, y1 = np.minimum([w, h], np.ceil(uv.max(axis=0) + padding)).astype(int)
    if x1 - x0 < 12 or y1 - y0 < 12:
        raise ValueError("inspection region is outside the camera image")
    return int(x0), int(y0), int(x1), int(y1)


def detect(ctx: Any, camera: dict[str, Any], *, prompt: str, crop=None, floor: float = 0.01) -> list[dict[str, Any]]:
    """Every segmentation above ``floor`` as world points with a principal axis and extents."""
    rgb = np.asarray(camera["rgb"])
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if crop is None else crop
    result = ctx.tool("sam3.segment_text", image=rgb[y0:y1, x0:x1], query=prompt, max_results=0)
    detections = []
    for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
        if float(score) < floor:
            continue
        local = np.asarray(mask, dtype=np.uint8)
        if local.shape != (y1 - y0, x1 - x0):
            raise ValueError("segmentation mask dimensions do not match image crop")
        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:y1, x0:x1] = local
        cloud = ctx.tool("geometry.mask_to_world_points", mask=full, depth=camera["depth"],
                         intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 12:
            continue
        center = np.median(points, axis=0)
        points = points[np.linalg.norm(points - center, axis=1) < 0.22]
        if len(points) < 12:
            continue
        center = np.median(points, axis=0)
        _, _, basis = np.linalg.svd(points - center, full_matrices=False)
        projection = (points - center) @ basis.T
        extent = np.percentile(projection, 95, axis=0) - np.percentile(projection, 5, axis=0)
        detections.append({"score": float(score), "center": center, "points": points,
                           "axis": basis[0], "extent": extent})
    return detections


def rod_like(detection: dict[str, Any], length_m=(0.025, 0.22), max_width_m: float = 0.045) -> bool:
    """Elongated: long axis within ``length_m``, at most ``max_width_m`` across, 2:1 or slimmer."""
    extent = detection["extent"]
    return (length_m[0] <= extent[0] <= length_m[1] and extent[1] <= max_width_m
            and extent[0] >= 2.0 * max(extent[1], 0.003))
