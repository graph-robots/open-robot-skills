"""Multi-view RGB-D evidence helpers for ``verify_placement``'s ``views`` mode.

Vendored from the sharps_disposal benchmark's perception graph's shared helper module,
where the post-release check that these serve was measured; a bundle cannot
import a graph's files. Named ``placementcommon`` (the ``tipcommon`` pattern)
so it never collides in ``sys.modules`` with another bundle's helpers.

Every input is an observation camera or robot proprioception; nothing here
reads simulator state.
"""

from typing import Any

import numpy as np


def xyz(p: dict[str, float]) -> np.ndarray:
    return np.array([p[k] for k in ("x", "y", "z")], dtype=np.float64)


def _pose_matrix(pose: Any) -> np.ndarray:
    """4x4 camera-to-world from a ``{position, rotation}`` pose or a 4x4 array."""
    if not isinstance(pose, dict):
        return np.asarray(pose, dtype=np.float64).reshape(4, 4)
    q = pose["rotation"]
    w, x, y, z = (float(q[k]) for k in ("w", "x", "y", "z"))
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.eye(4)
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    m[:3, 3] = xyz(pose["position"])
    return m


def crop_for_world_box(camera, lower, upper, padding: int = 16) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)`` covering a world-frame box's projection.

    Raises ``ValueError`` when the box is behind the camera or projects to a
    sliver outside the image -- that camera is not a calibrated view of it.
    """
    corners = np.array(
        [[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
    )
    c = (np.linalg.inv(_pose_matrix(camera["pose"])) @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    c = c[c[:, 2] > 0.01]
    if not len(c):
        raise ValueError("inspection region is behind the camera")
    uv = c @ np.asarray(camera["intrinsics"], dtype=np.float64).T
    uv = uv[:, :2] / uv[:, 2:]
    h, w = np.asarray(camera["rgb"]).shape[:2]
    x0, y0 = np.maximum(0, np.floor(uv.min(axis=0) - padding)).astype(int)
    x1, y1 = np.minimum([w, h], np.ceil(uv.max(axis=0) + padding)).astype(int)
    if x1 - x0 < 12 or y1 - y0 < 12:
        raise ValueError("inspection region is outside the camera image")
    return (int(x0), int(y0), int(x1), int(y1))


def detect(
    ctx,
    camera,
    prompt: str,
    *,
    crop=None,
    floor: float = 0.01,
    min_points: int = 12,
    outlier_radius_m: float = 0.22,
) -> list[dict[str, Any]]:
    """Every mask the segmenter returns, back-projected in the full calibrated image.

    ``max_results=0`` asks for all masks: in a crowded scene the object that
    matters is often not among the top few. Cropped masks are restored into
    the original image before back-projection, so the camera intrinsics stay
    valid.
    """
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
        cloud = ctx.tool(
            "geometry.mask_to_world_points",
            mask=full,
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < min_points:
            continue
        center = np.median(points, axis=0)
        points = points[np.linalg.norm(points - center, axis=1) < outlier_radius_m]
        if len(points) < min_points:
            continue
        center = np.median(points, axis=0)
        _, _, basis = np.linalg.svd(points - center, full_matrices=False)
        projection = (points - center) @ basis.T
        extent = np.percentile(projection, 95, axis=0) - np.percentile(projection, 5, axis=0)
        yy, xx = np.nonzero(full)
        detections.append(
            {
                "score": float(score),
                "center": center,
                "points": points,
                "axis": basis[0],
                "extent": extent,
                "camera": camera["name"],
                "crop": crop,
                "bounds": (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)),
            }
        )
    return detections


def summary(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": d["score"],
        "center": dict(zip("xyz", map(float, d["center"]), strict=True)),
        "camera": d["camera"],
        "crop": d["crop"],
        "extent": np.asarray(d["extent"]).tolist(),
    }
