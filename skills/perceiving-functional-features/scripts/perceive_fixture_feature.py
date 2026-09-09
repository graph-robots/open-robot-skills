"""Locate a profile-declared fixture feature with semantic RGB-D geometry."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


def _overhead(observation):
    return next(c for c in observation["cameras"] if c.get("name") == "overhead")


def _backproject(ctx, camera, mask):
    return ctx.tool("geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8), depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]


def _box_mask(shape, box):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1 = max(0, int(np.floor(float(box["x1"]))))
    y1 = max(0, int(np.floor(float(box["y1"]))))
    x2 = min(shape[1], int(np.ceil(float(box["x2"]))) + 1)
    y2 = min(shape[0], int(np.ceil(float(box["y2"]))) + 1)
    mask[y1:y2, x1:x2] = 255
    return mask


class Output(TypedDict):
    fixture_kind: str
    fixture_obb: dict[str, Any]
    fixture_mask: np.ndarray
    fixture_cloud: dict[str, Any]
    hook_tip: dict[str, float]
    fixture_axis: dict[str, float]
    fixture_feature: dict[str, Any]


def _feature(kind, tip, axis, radius_outer=0.0, seating_margin=0.0):
    feature = {"kind": kind, "pose": {"position": dict(tip),
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
            "axis": dict(axis), "radius_outer": float(radius_outer), "confidence": 1.0}
    if seating_margin > 0.0:
        feature["seating_margin"] = float(seating_margin)
    return feature


def _depth_support_points(workcell: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Extract foreground support points or raise the existing quality error."""
    if len(workcell) < 100:
        raise ValueError("too few calibrated board-region depth points")
    board_x = float(np.percentile(workcell[:, 0], float(profile.get("board_percentile", 85))))
    points = workcell[workcell[:, 0] < board_x - float(profile.get("foreground_margin", 0.004))]
    if len(points) < 30:
        raise ValueError("support foreground was not recovered")
    return points


def run(ctx: NodeContext, observation: dict[str, Any], target_kind: str = "",
        fixture_description: str = "fixture",
        feature_type: str = "shaft", destination_anchor_y: float | None = None,
        fixture_profiles: list[dict[str, Any]] | None = None) -> Output:
    camera = _overhead(observation)
    profiles = {str(profile["source_kind"]): profile for profile in (fixture_profiles or [])}
    if target_kind not in profiles:
        raise ValueError(f"no fixture profile declared for source kind {target_kind!r}")
    profile = profiles[target_kind]
    strategy = str(profile["strategy"])
    fixture_description = str(profile.get("fixture_description", fixture_description))
    feature_type = str(profile.get("feature_type", feature_type))
    if strategy == "depth_support":
        full_mask = np.full(camera["depth"].shape, 255, dtype=np.uint8)
        all_cloud = _backproject(ctx, camera, full_mask)
        all_points = np.asarray(all_cloud["points"], dtype=np.float64).reshape(-1, 3)
        bounds = profile["workspace"]
        unanchored_workcell = all_points[
            (all_points[:, 0] > float(bounds["x_min"]))
            & (all_points[:, 0] < float(bounds["x_max"]))
            & (all_points[:, 1] > float(bounds["y_min"]))
            & (all_points[:, 1] < float(bounds["y_max"]))
            & (all_points[:, 2] > float(bounds["z_min"]))
            & (all_points[:, 2] < float(bounds["z_max"]))
        ]
        workcell = unanchored_workcell
        if destination_anchor_y is not None:
            workcell = workcell[
                np.abs(workcell[:, 1] - float(destination_anchor_y))
                < float(profile.get("anchor_tolerance", 0.065))
            ]
        try:
            points = _depth_support_points(workcell, profile)
        except ValueError:
            # A source object may be farther from the centreline than its
            # corresponding destination. If the narrow metric anchor would
            # have failed, optionally fall back to the same declared workspace
            # partition. This never changes a successful anchored estimate.
            if destination_anchor_y is None or not bool(profile.get("anchor_fallback_partition", False)):
                raise
            axis_name = str(profile.get("partition_axis", "y"))
            axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_name)
            if axis_index is None:
                raise ValueError(f"unsupported partition_axis {axis_name!r}")
            split = float(profile.get("partition_split", 0.0))
            margin = max(0.0, float(profile.get("partition_margin", 0.0)))
            anchor_value = float(destination_anchor_y)
            if anchor_value >= split:
                fallback = unanchored_workcell[unanchored_workcell[:, axis_index] >= split + margin]
            else:
                fallback = unanchored_workcell[unanchored_workcell[:, axis_index] <= split - margin]
            points = _depth_support_points(fallback, profile)
        cutoff = np.percentile(points[:, 0], float(profile.get("tip_percentile", 3)))
        tip_points = points[points[:, 0] <= cutoff]
        tip = np.median(tip_points, axis=0)
        # Cross the support at a height clear of its top edge, then let the
        # feature-mating skill lower the handles onto it after the crossing.
        support_top = float(np.percentile(points[:, 2], 99))
        tip[0] += float(profile.get("tip_axis_offset", 0.003))
        tip[2] = support_top + float(profile.get("top_clearance", 0.018))
        cloud = {"points": points.astype(np.float32)}
        obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
        tip_dict = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
        axis_values = profile.get("axis", [-1.0, 0.0, 0.0])
        axis_dict = dict(zip(("x", "y", "z"), map(float, axis_values)))
        fixture_feature = _feature(
            feature_type, tip_dict, axis_dict,
            seating_margin=float(profile.get("seating_margin", 0.015))
        )
        if profile.get("mating_profile"):
            fixture_feature["mating_profile"] = dict(profile["mating_profile"])
        return {
            "fixture_kind": "support",
            "fixture_obb": obb,
            "fixture_mask": full_mask,
            "fixture_cloud": cloud,
            "hook_tip": tip_dict, "fixture_axis": axis_dict,
            "fixture_feature": fixture_feature,
        }
    if strategy == "parent_inferred_aperture":
        # Canonical perceiving-functional-features pipeline: language-localize
        # the fixture and its opening, back-project both, then fit the opening
        # plane. No calibrated rack coordinates or hole histogram are used.
        parent = ctx.tool(
            "sam3.segment_text", image=camera["rgb"],
            query=fixture_description, max_results=int(profile.get("max_results", 3)),
        )
        if not parent.get("masks"):
            raise ValueError("declared parent fixture was not visible")
        # Text segmentation occasionally ranks the entire pegboard above the
        # small box, especially after the first placed tool changes the scene.
        # Select a compact 3-D candidate before fitting the fixture. These are
        # scale gates, not calibrated world-position gates.
        parent_candidates = []
        scores = list(parent.get("scores") or [])
        for index, candidate_mask in enumerate(parent["masks"]):
            candidate_cloud = _backproject(ctx, camera, candidate_mask)
            points = np.asarray(candidate_cloud["points"], dtype=np.float64).reshape(-1, 3)
            if len(points) < 40:
                continue
            lower, upper = np.percentile(points, [1, 99], axis=0)
            span = upper - lower
            span_max = np.asarray(profile.get("span_max", [0.20, 0.22, 0.30]), dtype=float)
            if np.any(span > span_max):
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            parent_candidates.append((score, candidate_mask, candidate_cloud))
        if not parent_candidates:
            raise ValueError("parent fixture candidates were not compact in 3-D")
        _, fixture_mask, cloud = max(parent_candidates, key=lambda item: item[0])
        obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]

        parent_points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        # Infer the opening from the box as a whole. Once one long tool is in
        # the cup, a second text segmentation of "open interior" frequently
        # locks onto the front wall or the inserted tool and moves the target
        # several centimetres. The parent box remains stable under that
        # occlusion: its horizontal centre locates the opening and its dense
        # upper depth band locates the rim (sparse still-higher points are the
        # mounting hooks).
        top_lo, top_hi = np.percentile(
            parent_points[:, 2], profile.get("rim_percentiles", [85, 90])
        )
        top_band = parent_points[
            (parent_points[:, 2] >= top_lo) & (parent_points[:, 2] <= top_hi)
        ]
        if len(top_band) < 20:
            raise ValueError("container parent cloud did not expose a stable top rim")

        center_y = float(obb["center"]["y"])
        target_y = center_y
        if destination_anchor_y is not None:
            # Preserve the source object's left/right ordering at a much
            # smaller scale inside a shared container. This gives repeated
            # placements distinct landing points without encoding object
            # names, arm indices, fixture count, or calibrated coordinates.
            y_lo, y_hi = np.percentile(
                parent_points[:, 1], profile.get("lateral_percentiles", [2, 98])
            )
            half_span = 0.5 * float(y_hi - y_lo)
            lateral_offset = min(float(profile.get("lateral_offset_max", 0.012)),
                                 float(profile.get("lateral_offset_fraction", 0.25)) * half_span)
            if abs(float(destination_anchor_y) - center_y) > 1e-4:
                target_y += float(np.sign(float(destination_anchor_y) - center_y)) * lateral_offset
        tip_dict = {
            "x": float(obb["center"]["x"]),
            "y": target_y,
            "z": float(np.median(top_band[:, 2])),
        }
        axis_dict = {"x": 0.0, "y": 0.0, "z": -1.0}
        fixture_feature = _feature(feature_type, tip_dict, axis_dict)
        fixture_feature["radius_inner"] = float(profile.get("radius_inner", 0.025))
        # A container drop needs the tool tip just inside the rim before
        # release; dropping from the rim plane can leave a long object leaning
        # outside. Derive this shallow guide depth from opening size rather
        # than a world-coordinate waypoint.
        fixture_feature["insertion_depth"] = (
            float(profile.get("insertion_depth_radius_scale", 0.8))
            * fixture_feature["radius_inner"]
        )
        fixture_feature["description"] = str(profile.get("feature_description", "open interior"))
        if profile.get("mating_profile"):
            fixture_feature["mating_profile"] = dict(profile["mating_profile"])
        return {
            "fixture_kind": "container",
            "fixture_obb": obb,
            "fixture_mask": fixture_mask,
            "fixture_cloud": cloud,
            "hook_tip": tip_dict, "fixture_axis": axis_dict,
            "fixture_feature": fixture_feature,
        }
    if strategy != "thin_projection":
        raise ValueError(f"unsupported fixture feature strategy {strategy!r}")
    detections = ctx.tool(
        "grounding-dino.detect", image=camera["rgb"], query=fixture_description,
        box_threshold=float(profile.get("box_threshold", 0.18)),
        text_threshold=float(profile.get("text_threshold", 0.18)),
    )["detections"]

    candidates = []
    for detection in detections[:8]:
        label = str(detection.get("label", "")).strip().lower()
        # Grounding DINO can return the whole pegboard with a higher score than
        # the requested small hook.  A large scene/board box must never enter
        # the geometric hook fitter merely because it is elongated.
        # Accept phrases such as "small yellow hook on the pegboard".  Reject
        # only detections that do not identify a hook (for example a bare
        # "pegboard" label); the geometric gates below handle hook-like false
        # positives.
        label_token = str(profile.get("required_label_token", "")).strip().lower()
        if label_token and label_token not in label:
            continue
        box = detection["box"]
        width = max(1.0, float(box["x2"] - box["x1"]))
        height = max(1.0, float(box["y2"] - box["y1"]))
        if max(width / height, height / width) < float(profile.get("minimum_aspect_ratio", 2.5)):
            continue
        segmented = ctx.tool("sam3.segment_box", image=camera["rgb"], box=box)
        if not segmented["masks"] or float(segmented["scores"][0]) < float(profile.get("minimum_mask_score", 0.50)):
            continue
        sam_mask = segmented["masks"][0]

        # SAM often captures the thick mount and misses the hook's thin front
        # half.  Recover the complete protrusion from calibrated depth inside
        # DINO's tight box: the pegboard is the far-X plane and the hook is the
        # connected foreground in front of it.
        roi_mask = _box_mask(camera["depth"].shape, box)
        roi_cloud = _backproject(ctx, camera, roi_mask)
        roi_points = np.asarray(roi_cloud["points"], dtype=np.float64).reshape(-1, 3)
        if len(roi_points) < 40:
            continue
        board_x = float(np.percentile(roi_points[:, 0], 80))
        points = roi_points[roi_points[:, 0] < board_x - float(profile.get("foreground_margin", 0.003))]
        if len(points) < 20 or float(np.ptp(points[:, 0])) < 0.025:
            continue
        shaft_length = float(board_x - np.percentile(points[:, 0], 5))
        transverse_extent = min(float(np.ptp(points[:, 1])), float(np.ptp(points[:, 2])))
        # Physical plausibility gates for a small pegboard hook.  These reject
        # another elongated object that DINO occasionally assigns the fixture
        # label, while remaining independent of randomized world pose.
        length_bounds = profile.get("length_bounds", [0.025, 0.15])
        if not (float(length_bounds[0]) <= shaft_length <= float(length_bounds[1])
                and transverse_extent <= float(profile.get("transverse_extent_max", 0.04))):
            continue
        cloud = {"points": points.astype(np.float32)}
        # The task description supplies a vertical pegboard workcell. Select
        # the thin candidate nearest its dominant far-X plane, estimated in
        # calibrated world coordinates—not a fixed image pixel.
        candidates.append((float(detection.get("score", 0.0)), board_x, roi_mask, cloud, points))

    if not candidates:
        raise ValueError("no thin SAM3-refined hook candidate was found")
    if destination_anchor_y is None:
        _, board_x, mask, cloud, points = max(candidates, key=lambda item: item[0])
    else:
        _, board_x, mask, cloud, points = min(
            candidates,
            key=lambda item: abs(float(np.median(item[4][:, 1])) - float(destination_anchor_y)),
        )
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]

    # Hook projects from the board toward the robot, i.e. decreasing world X.
    # The very foremost points belong to the rounded/upturned nose, so their Z
    # is systematically above the horizontal shank centreline.  Recover distal
    # X from those points, but recover transverse Y/Z from the straight 10--30%
    # section immediately behind the nose.
    cutoff = np.percentile(points[:, 0], 5)
    tip_points = points[points[:, 0] <= cutoff]
    shank_lo, shank_hi = np.percentile(points[:, 0], [10, 30])
    shank_points = points[
        (points[:, 0] >= shank_lo) & (points[:, 0] <= shank_hi)
    ]
    if len(shank_points) < 8:
        shank_points = tip_points
    radius_outer = max(
        0.001, min(float(obb["extent"]["y"]), float(obb["extent"]["z"]))
    )
    shank_z_low, shank_z_high = np.percentile(shank_points[:, 2], [2, 98])
    if float(shank_z_high - shank_z_low) > 0.008:
        # Oblique RGB-D sometimes exposes both the upper and lower faces of
        # the rectangular hook arm. In that case subtracting a radius from the
        # lower envelope biases the shaft centre downward by about 12 mm.
        visible_midpoint = 0.5 * float(shank_z_low + shank_z_high)
        # In this view the broad lower return and curved nose dominate the
        # two envelopes; the horizontal arm occupies the upper half of that
        # visible span. Move from the silhouette midpoint to the arm centre by
        # half of the excess vertical span. This is scale-derived rather than
        # a fixed world-coordinate correction.
        shank_center_z = visible_midpoint + max(
            0.5 * float((shank_z_high - shank_z_low) - 2.0 * radius_outer),
            3.0 * radius_outer,
        )
    else:
        # At this oblique overhead pose the isolated straight-arm pixels are
        # the lower return face, not the upper silhouette assumed by the old
        # code. The arm centre lies two transverse half-thicknesses above the
        # depth median (the additional radius accounts for the rectangular
        # arm's vertical/thickness aspect ratio).
        shank_center_z = float(np.median(shank_points[:, 2]) + 2.0 * radius_outer)
    tip = np.array(
        [
            float(np.median(tip_points[:, 0])),
            float(np.median(shank_points[:, 1])),
            shank_center_z,
        ],
        dtype=np.float64,
    )
    # Direction is observed from the local board plane to the distal hook tip.
    # This removes the old hard-coded "world -X" mating assumption. The local
    # DINO ROI supplies the plane coordinate; RGB-D supplies the tip.
    board_point = np.array([board_x, tip[1], tip[2]], dtype=np.float64)
    axis = tip - board_point
    axis /= np.linalg.norm(axis)
    tip_dict = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
    axis_dict = {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])}
    # Perception reports the physical distal tip and the observed usable shaft
    # length. The mating skill, rather than an object-specific perception
    # branch, chooses how far a loop must cross and travel along the shaft.
    shaft_length = float(np.linalg.norm(board_point - tip))
    fixture_feature = _feature(
        feature_type,
        tip_dict,
        axis_dict,
        radius_outer=radius_outer,
        seating_margin=float(profile.get("seating_margin", 0.015)),
    )
    fixture_feature["usable_length"] = shaft_length
    if profile.get("mating_profile"):
        fixture_feature["mating_profile"] = dict(profile["mating_profile"])
    return {
        "fixture_kind": "hook", "fixture_obb": obb, "fixture_mask": mask, "fixture_cloud": cloud,
        "hook_tip": tip_dict, "fixture_axis": axis_dict,
        "fixture_feature": fixture_feature,
    }
