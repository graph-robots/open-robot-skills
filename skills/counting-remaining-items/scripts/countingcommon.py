"""RGB-D evidence helpers for ``count_remaining``.

Vendored from the sharps_disposal benchmark's perception graph's shared helper module; a
bundle cannot import a graph's files. Named ``countingcommon`` (the
``tipcommon`` pattern) so it never collides in ``sys.modules`` with another
bundle's helpers. The graph's syringe-specific constants (rod length and width
limits, the 20 mm duplicate radius, the 50 px re-crop pad) are parameters here,
and ``count_remaining`` exposes them.

Every input is an observation camera or robot proprioception; nothing here
reads simulator state.
"""

from typing import Any

import numpy as np


def xyz(p: dict[str, float]) -> np.ndarray:
    return np.array([p[k] for k in ("x", "y", "z")], dtype=np.float64)


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

    ``max_results=0`` asks for all masks: with six or more items in view, a
    real one is often not among the top few. Cropped masks are restored into
    the original image before back-projection, so the intrinsics stay valid.
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


def rod_like(
    d: dict[str, Any],
    *,
    min_length_m: float | None = None,
    max_length_m: float | None = None,
    max_width_m: float | None = None,
    min_aspect: float | None = None,
    width_floor_m: float = 0.003,
) -> bool:
    """Shape gate on the 90 % extents of a detection's principal axes.

    Every limit left ``None`` passes. ``min_aspect`` compares the long extent
    with the second one, floored at ``width_floor_m`` so a depth-flat mask
    cannot make a speck look elongated.
    """
    e = np.asarray(d["extent"], dtype=np.float64)
    if min_length_m is not None and e[0] < float(min_length_m):
        return False
    if max_length_m is not None and e[0] > float(max_length_m):
        return False
    if max_width_m is not None and e[1] > float(max_width_m):
        return False
    if min_aspect is not None and e[0] < float(min_aspect) * max(e[1], float(width_floor_m)):
        return False
    return True


def distinct(items: list[dict[str, Any]], distance: float = 0.02) -> list[dict[str, Any]]:
    """Highest-score-first non-maximum suppression on 3-D centres."""
    result: list[dict[str, Any]] = []
    for d in sorted(items, key=lambda x: x["score"], reverse=True):
        if not any(np.linalg.norm(d["center"] - x["center"]) < distance for x in result):
            result.append(d)
    return result


def expanded_bounds(items: list[dict[str, Any]], camera, padding: int = 50) -> tuple[int, int, int, int]:
    """The union of the items' pixel boxes, padded and clipped to the image."""
    h, w = np.asarray(camera["rgb"]).shape[:2]
    bounds = np.array([d["bounds"] for d in items])
    return (
        max(0, int(bounds[:, 0].min()) - padding),
        max(0, int(bounds[:, 1].min()) - padding),
        min(w, int(bounds[:, 2].max()) + padding),
        min(h, int(bounds[:, 3].max()) + padding),
    )


def summary(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": d["score"],
        "center": dict(zip("xyz", map(float, d["center"]), strict=True)),
        "camera": d["camera"],
        "crop": d["crop"],
        "extent": np.asarray(d["extent"]).tolist(),
    }
