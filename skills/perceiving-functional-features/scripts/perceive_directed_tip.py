"""Localize an elongated object and its directed insertion tip from one RGB-D view.

The object is segmented by text and back-projected. Its long axis gets a
direction from a visible marker at one end (``direction_marker_description``,
for example a coloured cap) or, when no marker is described or found, from the
narrower of the two ends. The tip is the extreme of the cloud along that
directed axis; the returned pose has local Z pointing from the object body
through the tip, which is the insertion direction a mating skill needs.

Everything below the defaults is opt-in and was folded in from the sharps
disposal graph (``gap_perception_v2``, ``perceive_target.py``, sweep s8), where
the object is a syringe lying among others in a tray and the marker its red
needle cap. With every new parameter at its default the script makes the same
tool calls with the same arguments and returns the same values it did before;
``support_z`` and ``jaw_clearance_m`` are additive outputs.

- ``candidate_score_floor`` keeps every mask above a rescue floor, drops the
  lower-scored of two overlapping masks, and ranks the rest instead of taking
  SAM's top mask.
- ``min_mask_pixels`` / ``min_span_m`` / ``max_span_m`` / ``max_width_m`` reject
  fragments, lines and masks that merge neighbours.
- ``exclude_center`` / ``exclude_radius_m`` / ``max_z`` reject objects already
  delivered (and a marker detected there).
- ``min_height_above_floor_m`` rejects flat look-alikes lying at floor height.
- ``jaw_clearance_free_m`` reads the room across a parallel jaw from depth and
  ranks by it.
- ``marker_rgb_rule`` / ``marker_max_lateral_m`` / ``width_landmark_to_marker_m``
  extend the direction cascade: marker-coloured pixels on the axis, then the
  detected marker on the axis, then a width landmark, then the narrower end.
- ``axis_from_svd`` takes the principal axis signed toward the marker rather than
  the marker-minus-centre direction.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy import ndimage
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, np.ndarray]
    insertion_pose: dict[str, Any]
    marker_center: dict[str, float]
    functional_feature: dict[str, Any]
    support_z: float
    jaw_clearance_m: float


def _camera(observation: dict[str, Any], name: str) -> dict[str, Any]:
    cameras = observation.get("cameras") or []
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    for camera in cameras:
        if camera.get("name") == name:
            return camera
    raise ValueError(f"observation has no camera named {name!r}")


def _mask(
    ctx: NodeContext,
    image: Any,
    query: str,
    threshold: float,
    optional: bool = False,
    max_results: int = 3,
) -> np.ndarray | None:
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=max_results)
    if not result.get("masks") or float(result["scores"][0]) < threshold:
        if optional:
            return None
        raise ValueError(f"could not perceive {query!r}")
    return np.asarray(result["masks"][0], dtype=np.uint8)


def _candidate_masks(
    ctx: NodeContext, image: Any, query: str, floor: float, max_results: int
) -> list[tuple[np.ndarray, float]]:
    """Every mask above the rescue floor, in SAM's order, with its score.

    The caller applies ``object_score_min`` only after the marker-colour
    evidence has had its chance. In the source graph SAM's score for a thin
    syringe on the textured mat collapsed to ~0.04 when the parked wrist
    encroached on the tray: 7 of 20 episodes of a 20-layout sweep aborted at a
    0.08 gate with the syringe plainly in frame. A mask that contains the red
    cap's pixels is the target whatever SAM scored it, so low-score candidates
    are kept alive (floor 0.02) long enough to take that test.
    """
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=max_results)
    pairs = [
        (np.asarray(m, dtype=np.uint8), float(s))
        for m, s in zip(result.get("masks") or [], result.get("scores") or [], strict=False)
        if float(s) >= floor
    ]
    if not pairs:
        raise ValueError(f"could not perceive {query!r}")
    return pairs


def _distinct(entries: list[dict[str, Any]], overlap_max: float) -> list[dict[str, Any]]:
    """SAM returns overlapping masks for one object; keep the best-scored of each."""
    kept: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda e: e["score"], reverse=True):
        mask = entry["mask"] > 0
        if all(
            np.logical_and(mask, k["mask"] > 0).sum()
            < overlap_max * min(mask.sum(), (k["mask"] > 0).sum())
            for k in kept
        ):
            kept.append(entry)
    return kept


def _points(ctx: NodeContext, camera: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=mask,
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _span_and_width(points: np.ndarray) -> tuple[float, float]:
    """Length (3rd-97th percentile along the principal axis) and full width (2x p90 radial).

    The source graph's syringe-sized gate: the 150 mm syringe is seen as
    113-117 mm and its flange is 23.6 mm wide. Sweep s2 (t02, t03, t04, t15)
    kept SAM's 28-43 mm sliver around a cap, centring the grasp on the needle
    end; ``tray_clutter_t16`` kept a 0.84-score mask spanning several
    neighbours whose 250 x 155 mm OBB put the grasp between syringes. It
    admitted 80-200 mm long and at most 45 mm wide.
    """
    centered = points - np.median(points, axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    along = centered @ basis[0]
    across = centered - np.outer(along, basis[0])
    span = float(np.percentile(along, 97) - np.percentile(along, 3))
    width = 2.0 * float(np.percentile(np.linalg.norm(across, axis=1), 90))
    return span, width


def _horizontal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    axis = basis[0][:2] / max(float(np.linalg.norm(basis[0][:2])), 1.0e-9)
    return center, axis


_RGB_RULE_KEYS = ("channel", "min_value", "min_margin", "min_pixels", "max_lateral_m", "min_along_m")


def _check_rgb_rule(rule: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in _RGB_RULE_KEYS if key not in rule]
    if missing:
        raise ValueError(f"marker_rgb_rule is missing {missing}; it needs all of {list(_RGB_RULE_KEYS)}")
    if int(rule["channel"]) not in (0, 1, 2):
        raise ValueError("marker_rgb_rule channel must be 0, 1 or 2 (R, G, B)")
    return rule


def _marker_colour(rgb: Any, rule: dict[str, Any]) -> np.ndarray:
    """Pixels whose ``channel`` exceeds ``min_value`` and both other channels by ``min_margin``.

    Plain channel arithmetic rather than another SAM call: in the source graph
    the cap query itself was what failed in the runs this exists for, and the
    red cap was the scene's one deliberately red thing (R > 110, R > G + 40,
    R > B + 40).
    """
    pixels = np.asarray(rgb, dtype=np.int16)
    channel = int(rule["channel"])
    value = pixels[..., channel]
    selected = value > int(rule["min_value"])
    for other in (0, 1, 2):
        if other != channel:
            selected &= value > pixels[..., other] + int(rule["min_margin"])
    return selected


def _marker_pixel_count(rgb: Any, mask: np.ndarray, rule: dict[str, Any]) -> int:
    """How many of a mask's pixels are marker-coloured.

    A count, not a fraction: ranking by the fraction preferred SAM's low-score
    sliver around the cap alone over the whole syringe that also contains it.
    ``tray_clutter_t03``: a 165-px, score-0.06 cap-end fragment (18 red px,
    10.9 %) beat the 1764-px, score-0.79 syringe (36 red px, 2.0 %).
    """
    return int((_marker_colour(rgb, rule) & (np.asarray(mask) > 0)).sum())


def _colour_marker_point(
    ctx: NodeContext, camera: dict[str, Any], mask: np.ndarray, points: np.ndarray, rule: dict[str, Any]
) -> np.ndarray | None:
    """The marker's world point from marker-coloured pixels on the target's own axis, or None.

    The pixels must lie within ``max_lateral_m`` of the axis and at least
    ``min_along_m`` toward one end. Sweep s3 ``t12``: a syringe-sized mask
    along the tray's end wall, 22.7 mm beside ``syringe_0``, held that
    syringe's cap pixels, ranked first on clearance, and the grasp landed on
    the wall (graph: 10 mm lateral, 25 mm along).
    """
    colour = _marker_colour(camera["rgb"], rule) & (np.asarray(mask) > 0)
    if int(colour.sum()) < int(rule["min_pixels"]):
        return None
    marker = _points(ctx, camera, colour.astype(np.uint8))
    if len(marker) < 4:
        return None
    center, axis = _horizontal_axis(points)
    point = np.median(marker, axis=0)
    delta = point[:2] - center[:2]
    if (
        abs(float(delta @ [-axis[1], axis[0]])) > float(rule["max_lateral_m"])
        or abs(float(delta @ axis)) < float(rule["min_along_m"])
    ):
        return None
    return point


def _marker_from_width_landmark(
    points: np.ndarray, landmark_to_marker_m: float, ratio_min: float
) -> np.ndarray | None:
    """Where the marker is, from the widest stretch of the cloud along its axis.

    The source graph's syringe flange is 23.6 mm wide and its cap centre sits
    88 mm from it on the needle side (mesh survey); a flange narrower than 1.5x
    the barrel's median width is not trusted. Which end is the needle decides
    which end goes into the box first: a syringe posted plunger-first stands on
    its thumb press under the hole and the next one jams on it (sweep s5
    t05/t06), and the narrow-tail guess cannot tell the needle hub from the
    plunger rod in overhead depth.
    """
    center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    axis = basis[0]
    along = (points - center) @ axis
    lateral = np.linalg.norm((points - center) - np.outer(along, axis), axis=1)
    edges = np.linspace(np.percentile(along, 3), np.percentile(along, 97), 11)
    widths, mids = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (along >= lo) & (along < hi)
        if inside.sum() >= 8:
            widths.append(2.0 * float(np.percentile(lateral[inside], 90)))
            mids.append(0.5 * (lo + hi))
    if len(widths) < 5:
        return None
    widths = np.asarray(widths)
    landmark = int(np.argmax(widths))
    if widths[landmark] < ratio_min * float(np.median(widths)):
        return None
    toward = axis if mids[landmark] > 0 else -axis
    return center + toward * mids[landmark] * np.sign(mids[landmark]) - toward * landmark_to_marker_m


def _depth_ring(
    ctx: NodeContext, camera: dict[str, Any], mask: np.ndarray, gap_px: int, outer_px: int
) -> np.ndarray:
    """World points of a ring ``gap_px``-``outer_px`` pixels outside the mask."""
    distance = ndimage.distance_transform_edt(~(np.asarray(mask) > 0))
    ring = (distance > gap_px) & (distance <= outer_px)
    return _points(ctx, camera, ring.astype(np.uint8))


def _jaw_clearance(
    around: np.ndarray,
    points: np.ndarray,
    protrusion_m: float,
    half_length_m: float | None,
    own_body_half_width_m: float,
    no_neighbour_m: float,
) -> float:
    """Free distance [m] across the axis to the nearest surface standing proud of the floor.

    Read from depth, so undetected neighbours and walls count too. Raised
    points closer than ``own_body_half_width_m`` to the axis are the object's
    own body: SAM's mask stops short of a rounded barrel edge and the flange.
    An offline replay of sweep s4's ``t08`` read 8.5 mm of clearance for a
    syringe whose nearest neighbour lay 57.6 mm away (graph: 13 mm).
    """
    if len(around) < 50:
        return no_neighbour_m
    floor = float(np.percentile(around[:, 2], 10))
    raised = around[around[:, 2] > floor + protrusion_m]
    center, axis = _horizontal_axis(points)
    delta = raised[:, :2] - center[:2]
    across = np.abs(delta @ [-axis[1], axis[0]])
    near = across > own_body_half_width_m
    if half_length_m is not None:
        near &= np.abs(delta @ axis) <= half_length_m
    if not near.any():
        return no_neighbour_m
    return float(np.min(across[near]))


def _support_z(
    ctx: NodeContext,
    camera: dict[str, Any],
    target_mask: np.ndarray,
    occupied: np.ndarray,
    points: np.ndarray,
    outer_px: int,
    gap_px: int,
) -> float:
    """The height of the surface the target rests on, from depth beside it.

    The target's own cloud is a poor floor estimate: transparent-looking bodies
    and mask edges mix in floor pixels, which drags an OBB bottom (and a grasp
    height derived from it) onto or below the floor. A thin ring just outside
    the mask, clear of every confident detection, sees the support itself. Falls
    back to the cloud's 5th percentile when the ring is too small to trust.
    """
    if outer_px > 0:
        target = np.asarray(target_mask) > 0
        ring = ndimage.binary_dilation(target, iterations=outer_px) & ~ndimage.binary_dilation(
            occupied | target, iterations=gap_px
        )
        if int(ring.sum()) >= 50:
            z = _points(ctx, camera, ring.astype(np.uint8))[:, 2]
            if len(z) >= 50:
                return float(np.median(z))
    return float(np.percentile(points[:, 2], 5))


def _narrow_endpoint(points: np.ndarray) -> np.ndarray:
    """The end of the principal axis whose cross-section is narrower."""
    origin = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - origin, full_matrices=False)
    axis = basis[0]
    along = (points - origin) @ axis
    tails = [
        points[along <= np.percentile(along, 15)],
        points[along >= np.percentile(along, 85)],
    ]
    widths = []
    for tail in tails:
        centered = tail - np.median(tail, axis=0)
        perpendicular = centered - np.outer(centered @ axis, axis)
        widths.append(float(np.percentile(np.linalg.norm(perpendicular, axis=1), 90)))
    return np.median(tails[int(np.argmin(widths))], axis=0)


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    object_description: str,
    direction_marker_description: str = "",
    feature_type: str = "tip",
    object_score_min: float = 0.08,
    marker_score_min: float = 0.06,
    camera_name: str = "overhead",
    max_results: int = 3,
    candidate_score_floor: float | None = None,
    candidate_overlap_max: float = 0.3,
    min_mask_pixels: int = 0,
    min_span_m: float = 0.0,
    max_span_m: float | None = None,
    max_width_m: float | None = None,
    exclude_center: dict[str, float] | None = None,
    exclude_radius_m: float = 0.0,
    max_z: float | None = None,
    min_height_above_floor_m: float | None = None,
    depth_ring_gap_px: int = 6,
    depth_ring_outer_px: int = 60,
    jaw_clearance_free_m: float | None = None,
    jaw_clearance_enough_m: float | None = None,
    clearance_half_length_m: float | None = None,
    own_body_half_width_m: float = 0.0,
    protrusion_m: float = 0.004,
    no_neighbour_clearance_m: float = 0.08,
    marker_rgb_rule: dict[str, Any] | None = None,
    marker_max_lateral_m: float | None = None,
    marker_max_gap_m: float | None = None,
    width_landmark_to_marker_m: float | None = None,
    width_landmark_ratio_min: float = 1.5,
    axis_from_svd: bool = False,
    support_ring_px: int = 0,
    support_ring_gap_px: int = 4,
) -> Output:
    camera = _camera(observation, camera_name)
    rule = _check_rgb_rule(marker_rgb_rule) if marker_rgb_rule is not None else None
    ranked_mode = candidate_score_floor is not None
    if ranked_mode:
        candidates = _candidate_masks(
            ctx, camera["rgb"], object_description, float(candidate_score_floor), max_results
        )
    else:
        top = _mask(ctx, camera["rgb"], object_description, object_score_min, max_results=max_results)
        candidates = [(top, float("inf"))]
    exclude_xy = (
        np.array([float(exclude_center[k]) for k in ("x", "y")]) if exclude_center is not None else None
    )
    sized = min_span_m > 0.0 or max_span_m is not None or max_width_m is not None

    entries: list[dict[str, Any]] = []
    occupied = None
    for mask, score in candidates:
        if score >= object_score_min:
            # Every confident detection -- merged ones included -- is somewhere
            # the support ring must not sample.
            occupied = (mask > 0) if occupied is None else (occupied | (mask > 0))
        if int((mask > 0).sum()) < min_mask_pixels:
            continue
        cloud = _points(ctx, camera, mask)
        if len(cloud) < 30:
            if not ranked_mode:
                raise ValueError(f"{object_description!r} mask contains too few depth points")
            continue
        if sized:
            span, width = _span_and_width(cloud)
            if span < min_span_m or (max_span_m is not None and span > max_span_m):
                continue
            if max_width_m is not None and width > max_width_m:
                continue
        center = np.median(cloud, axis=0)
        if exclude_xy is not None and np.linalg.norm(center[:2] - exclude_xy) <= exclude_radius_m:
            continue
        if max_z is not None and center[2] > float(max_z):
            continue
        marker_px = _marker_pixel_count(camera["rgb"], mask, rule) if rule is not None else 0
        entries.append({"mask": mask, "score": score, "marker_px": marker_px, "cloud": cloud})
    if not entries:
        raise ValueError(f"no {object_description!r} candidate passed the size and exclusion gates")
    entries = _distinct(entries, candidate_overlap_max) if ranked_mode else entries
    min_marker_px = int(rule["min_pixels"]) if rule is not None else 0
    if ranked_mode and not any(
        (rule is not None and e["marker_px"] >= min_marker_px) or e["score"] >= object_score_min
        for e in entries
    ):
        raise ValueError(f"could not perceive {object_description!r}")

    rings: dict[int, np.ndarray] = {}

    def ring_of(index: int) -> np.ndarray:
        if index not in rings:
            rings[index] = _depth_ring(
                ctx, camera, entries[index]["mask"], depth_ring_gap_px, depth_ring_outer_px
            )
        return rings[index]

    if min_height_above_floor_m is not None:
        # A lying syringe's top stands 11 mm proud of what it rests on. Sweep
        # s7's t09 (and s6's t23) grasped a 170 x 40 mm "syringe" in the box's
        # shadow on the bench, 4 mm thick at bench height (graph: 5 mm).
        raised = []
        for index, entry in enumerate(entries):
            around = ring_of(index)
            floor = float(np.percentile(around[:, 2], 10)) if len(around) >= 50 else None
            if floor is None or float(np.percentile(entry["cloud"][:, 2], 90)) - floor >= min_height_above_floor_m:
                raised.append(index)
        if not raised:
            raise ValueError(f"no {object_description!r} candidate stands above its support")
    else:
        raised = list(range(len(entries)))

    # Rank: a candidate holding marker-coloured pixels off its own axis is not
    # the object and goes last (one holding none is neutral: SAM's masks often
    # stop short of a thin cap). Then room across the jaw -- a neighbour closer
    # than the fingertip band is landed on (tray_clutter_t16/t17: the hand
    # stopped 7-9 mm high on the next barrel). Then SAM's score.
    ranked = []
    for index in raised:
        entry = entries[index]
        clearance = float("nan")
        if jaw_clearance_free_m is not None:
            clearance = _jaw_clearance(
                ring_of(index),
                entry["cloud"],
                protrusion_m,
                clearance_half_length_m,
                own_body_half_width_m,
                no_neighbour_clearance_m,
            )
        not_vetoed = (
            rule is None
            or entry["marker_px"] < min_marker_px
            or _colour_marker_point(ctx, camera, entry["mask"], entry["cloud"], rule) is not None
        )
        if jaw_clearance_free_m is not None:
            enough = jaw_clearance_enough_m if jaw_clearance_enough_m is not None else float("inf")
            room = (clearance >= jaw_clearance_free_m, min(clearance, enough))
        else:
            room = (True, 0.0)
        ranked.append((not_vetoed, *room, entry["score"], index, clearance))
    best_rank = max(ranked)
    chosen = entries[best_rank[4]]
    target_mask, points, jaw_clearance = chosen["mask"], chosen["cloud"], float(best_rank[5])

    obb = ctx.tool("geometry.filter_and_compute_obb", points={"points": points.astype(np.float32)})[
        "obb"
    ]
    marker_mask = None
    if direction_marker_description:
        marker_mask = _mask(
            ctx,
            camera["rgb"],
            direction_marker_description,
            marker_score_min,
            optional=True,
            max_results=max_results,
        )
    marker_points = (
        _points(ctx, camera, marker_mask) if marker_mask is not None else np.empty((0, 3))
    )
    # The direction cascade. Each opt-in step goes ahead of the ones it
    # outranks; with none enabled it is the detected marker, then the narrow end.
    marker = None
    if rule is not None:
        marker = _colour_marker_point(ctx, camera, target_mask, points, rule)
    if marker is None and len(marker_points) >= 8:
        hint = np.median(marker_points, axis=0)
        if exclude_xy is not None and np.linalg.norm(hint[:2] - exclude_xy) <= exclude_radius_m:
            # A marker on an object already delivered stays highly visible.
            hint = None
        if hint is not None and (marker_max_lateral_m is not None or marker_max_gap_m is not None):
            shaft_center = np.median(points, axis=0)
            _, _, basis = np.linalg.svd(points - shaft_center, full_matrices=False)
            delta = hint - shaft_center
            lateral = float(np.linalg.norm(delta - basis[0] * float(delta @ basis[0])))
            nearest = float(np.min(np.linalg.norm(points - hint, axis=1)))
            if (marker_max_lateral_m is not None and lateral > marker_max_lateral_m) or (
                marker_max_gap_m is not None and nearest > marker_max_gap_m
            ):
                hint = None
        marker = hint
    if marker is None and width_landmark_to_marker_m is not None:
        marker = _marker_from_width_landmark(
            points, float(width_landmark_to_marker_m), float(width_landmark_ratio_min)
        )
    if marker is None:
        marker = _narrow_endpoint(points)
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=np.float64)
    if axis_from_svd:
        # The principal axis, signed toward the marker. ``marker - centre``
        # tilts the axis by however far the marker point sits off the
        # centreline (a cap's median, a landmark estimate).
        shaft_center = np.median(points, axis=0)
        _, _, basis = np.linalg.svd(points - shaft_center, full_matrices=False)
        axis = basis[0] if float(basis[0] @ (marker - center)) > 0 else -basis[0]
    else:
        axis = marker - center
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    along = (points - center) @ axis
    tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
    reference = (
        np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    x_axis = np.cross(reference, axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, axis))).as_quat()
    insertion_pose = {
        "position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
        "rotation": {
            "w": float(quat[3]),
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
        },
    }
    centered = points - tip
    radial = centered - np.outer(centered @ axis, axis)
    radius = float(np.percentile(np.linalg.norm(radial, axis=1), 35))
    if occupied is None:
        occupied = np.zeros(np.asarray(target_mask).shape, dtype=bool)
    support_z = _support_z(
        ctx, camera, target_mask, occupied, points, int(support_ring_px), int(support_ring_gap_px)
    )
    return {
        "target_obb": obb,
        "target_mask": target_mask,
        "target_cloud": {"points": points.astype(np.float32)},
        "marker_center": {"x": float(marker[0]), "y": float(marker[1]), "z": float(marker[2])},
        "insertion_pose": insertion_pose,
        "support_z": support_z,
        "jaw_clearance_m": jaw_clearance,
        "functional_feature": {
            "kind": feature_type,
            "pose": insertion_pose,
            "axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
            "radius_outer": max(radius, 1.0e-4),
            "confidence": 1.0,
            "description": direction_marker_description or f"tip of {object_description}",
        },
    }
