"""Localize an object and a profile-declared manipulation feature from RGB-D."""

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _overhead(observation):
    return next(c for c in observation["cameras"] if c.get("name") == "overhead")


def _backproject(ctx, camera, mask):
    return ctx.tool("geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8), depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]


def _complete_to_support(ctx, camera, mask, cloud):
    fg = np.asarray(mask) > 0
    ring = binary_dilation(fg, iterations=8) & ~binary_dilation(fg, iterations=2)
    support = _backproject(ctx, camera, ring.astype(np.uint8) * 255)
    pts = np.asarray(cloud["points"], dtype=np.float32).reshape(-1, 3)
    support_pts = np.asarray(support["points"], dtype=np.float32).reshape(-1, 3)
    if len(pts) < 10 or len(support_pts) < 20:
        return cloud
    support_z, top_z = float(np.median(support_pts[:, 2])), float(np.percentile(pts[:, 2], 75))
    if not 0.001 < top_z - support_z < 0.08:
        return cloud
    bottom = pts.copy(); bottom[:, 2] = support_z + 0.001
    return {"points": np.concatenate([pts, bottom], axis=0).astype(np.float32)}


def _mask_center_world(ctx, camera, mask):
    pts = np.asarray(_backproject(ctx, camera, mask)["points"], dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        raise ValueError("ring mask contains too few depth points")
    center = np.median(pts, axis=0)
    return {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}


def _feature_pose(ctx, camera, mask, normal_hint=None):
    """Recover a complete 6D planar-feature frame, not only its centre."""
    points = _backproject(ctx, camera, mask)
    return ctx.tool(
        "geometry.fit_planar_feature", points=points, normal_hint=normal_hint,
        fit_circle_center=True,
    )["pose"]


def _feature_fit(ctx, camera, mask, normal_hint=None):
    points = _backproject(ctx, camera, mask)
    return ctx.tool(
        "geometry.fit_planar_feature", points=points, normal_hint=normal_hint,
        fit_circle_center=True,
    )


@lru_cache(maxsize=8)
def _planar_model(model_path: str) -> np.ndarray:
    """Return deterministic surface samples for planar CAD registration."""
    import trimesh

    mesh = trimesh.load_mesh(Path(model_path), process=False)
    points, _ = trimesh.sample.sample_surface(mesh, 30000, seed=0)
    return np.asarray(points[:, :2], dtype=np.float64)


def _register_planar_model(points: np.ndarray, model_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Register an observed silhouette to a declared flat-object CAD model.

    A semantic mask of an empty finger hole frequently covers only a coloured
    rim and can displace its reported centre by several centimetres. The full
    silhouette is far better constrained. Search yaw globally, then refine a
    planar rigid transform; SAM is subsequently used only to choose which of
    the two CAD bows the instruction refers to.
    """
    observed = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    observed = observed[np.all(np.isfinite(observed), axis=1)]
    if len(observed) < 80:
        raise ValueError("planar CAD registration needs at least 80 points")
    observed_xy = observed[:, :2]
    model_xy = _planar_model(model_path)
    model_center = model_xy.mean(axis=0)
    observed_center = observed_xy.mean(axis=0)
    model_tree = cKDTree(model_xy)

    best = None
    for yaw in np.deg2rad(np.arange(0.0, 360.0, 4.0)):
        rotation = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float64,
        )
        translation = observed_center - rotation @ model_center
        observed_local = (observed_xy - translation) @ rotation
        distances, _ = model_tree.query(observed_local, k=1)
        score = float(np.sqrt(np.mean(np.square(np.minimum(distances, 0.015)))))
        if best is None or score < best[0]:
            best = (score, rotation, translation)

    _, rotation, translation = best
    # Point-to-point refinement removes the coarse yaw quantization while a
    # trimmed set prevents table pixels inside the openings from biasing it.
    for _ in range(30):
        world_model = model_xy @ rotation.T + translation
        distances, indices = cKDTree(world_model).query(observed_xy, k=1)
        keep = distances <= min(0.008, float(np.quantile(distances, 0.85)))
        if int(np.count_nonzero(keep)) < 60:
            break
        source, target = world_model[indices[keep]], observed_xy[keep]
        source_center, target_center = source.mean(axis=0), target.mean(axis=0)
        u, _, vt = np.linalg.svd(
            (source - source_center).T @ (target - target_center)
        )
        correction = vt.T @ u.T
        if np.linalg.det(correction) < 0.0:
            vt[-1] *= -1.0
            correction = vt.T @ u.T
        delta = target_center - correction @ source_center
        rotation = correction @ rotation
        translation = correction @ translation + delta
        if (float(np.linalg.norm(delta)) < 2.0e-6
                and abs(float(np.arctan2(correction[1, 0], correction[0, 0]))) < 2.0e-5):
            break
    rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
    return rotation, translation, rmse


def _point_pose(center):
    """A point feature has no observed orientation; mark it with identity."""
    return {
        "position": dict(center),
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }


def _axis_pose(center, axis):
    """Pose whose local Z is a measured directed mating axis."""
    z_axis = np.asarray(axis, dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)
    # Choose a stable perpendicular without assuming a world direction for the
    # tool.  The sign of Z remains the observed handle-to-tip direction.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ z_axis)) > 0.90:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()
    return {
        "position": dict(center),
        "rotation": {
            "w": float(quat[3]), "x": float(quat[0]),
            "y": float(quat[1]), "z": float(quat[2]),
        },
    }


def _feature_center_from_mask(ctx, camera, query, threshold, z, reference_center=None):
    """Locate an empty opening in XY while keeping the tool's observed Z."""
    mask = _best_mask(ctx, camera["rgb"], query, threshold, camera, reference_center)
    center = _mask_center_world(ctx, camera, mask)
    center["z"] = float(z)
    return center


class Output(TypedDict):
    target_kind: str
    relation: str
    target_obb: dict[str, Any]
    target_mask: np.ndarray
    target_cloud: dict[str, Any]
    feature_center: dict[str, float]
    feature_pose: dict[str, Any]
    functional_feature: dict[str, Any]


def _best_mask(ctx: NodeContext, image: np.ndarray, query: str, threshold: float,
               camera=None, reference_center=None) -> np.ndarray:
    result = ctx.tool("sam3.segment_text", image=image, query=query, max_results=6)
    candidates = [(mask, float(score)) for mask, score in zip(
        result.get("masks") or [], result.get("scores") or []) if float(score) >= threshold]
    if not candidates:
        raise ValueError(f"SAM3 could not localize {query!r}")
    if camera is None or reference_center is None:
        return max(candidates, key=lambda item: item[1])[0]
    reference = np.array([reference_center[k] for k in ("x", "y", "z")], dtype=float)
    localized = []
    for mask, score in candidates:
        try:
            center = _mask_center_world(ctx, camera, mask)
        except ValueError:
            continue
        xyz = np.array([center[k] for k in ("x", "y", "z")], dtype=float)
        localized.append((float(np.linalg.norm(xyz - reference)), -score, mask))
    if not localized:
        raise ValueError(f"SAM3 localized {query!r} but no mask had usable depth")
    # Bind the part to the selected parent instance. This is essential when
    # multiple visually identical objects are present: the globally highest
    # scoring ring/handle may belong to another source object.
    return min(localized, key=lambda item: item[:2])[2]


def _best_distal_loop_mask(ctx: NodeContext, camera: dict[str, Any],
                           obb: dict[str, Any], profile: dict[str, Any]) -> np.ndarray:
    """Bind a semantic loop detection to the selected elongated parent.

    In multi-object scenes, another object's opening can score as the requested
    loop. A valid distal loop must lie near the selected parent's horizontal
    long-axis corridor and away from its centroid. This is geometric instance
    binding, independent of world left/right or semantic object identity.
    """
    q = obb["orientation"]
    rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([obb["extent"][k] for k in ("x", "y", "z")], dtype=float)
    horizontal = [i for i in range(3) if abs(float(rotation[2, i])) < 0.75]
    long_index = max(horizontal, key=lambda i: extents[i])
    long_axis = rotation[:, long_index].copy(); long_axis[2] = 0.0
    long_axis /= max(float(np.linalg.norm(long_axis)), 1.0e-12)
    center = np.array([obb["center"][k] for k in ("x", "y", "z")], dtype=float)
    candidates = []
    queries = profile.get("feature_queries") or [profile["feature_query"]]
    for query in queries:
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=6)
        for mask, score in zip(result.get("masks") or [], result.get("scores") or []):
            if float(score) < 0.05:
                continue
            try:
                feature = _mask_center_world(ctx, camera, mask)
            except ValueError:
                continue
            try:
                fit = _feature_fit(ctx, camera, mask, {"x": 0.0, "y": 0.0, "z": 1.0})
                radius = float(fit.get("radius_inner", fit.get("radius", 0.0)))
            except (ValueError, KeyError):
                continue
            delta = np.array([feature[k] for k in ("x", "y", "z")]) - center
            axial = abs(float(delta @ long_axis))
            lateral = float(np.linalg.norm(delta - (delta @ long_axis) * long_axis))
            # Use scale-relative bounds so this estimator transfers across
            # elongated objects of different physical sizes.
            min_axial = max(float(profile.get("minimum_axial", 0.035)),
                            float(profile.get("minimum_axial_scale", 0.40)) * float(extents[long_index]))
            max_lateral = max(float(profile.get("maximum_lateral", 0.025)),
                              float(profile.get("maximum_lateral_scale", 0.35)) * float(extents[long_index]))
            # An open jaw is frequently labelled as a ring, but its accidental
            # circle fit is much smaller than the closed box end. Scale the
            # minimum with the parent tool rather than hard-coding this asset.
            min_radius = max(float(profile.get("minimum_radius", 0.004)),
                             float(profile.get("minimum_radius_scale", 0.060)) * float(extents[long_index]))
            if axial >= min_axial and lateral <= max_lateral and radius >= min_radius:
                candidates.append((-radius, lateral, -axial, -float(score), mask))
    if not candidates:
        raise ValueError("SAM3 could not localize a distal loop on the selected parent")
    return min(candidates, key=lambda item: item[:4])[4]


def _shift_grasp_toward_feature(
    obb: dict[str, Any], feature_center: dict[str, float], profile: dict[str, Any]
) -> None:
    """Apply a caller-declared grasp displacement toward a perceived feature."""
    requested = max(0.0, float(profile.get("grasp_toward_feature_m", 0.0)))
    if requested <= 0.0:
        return
    axes = tuple(str(axis) for axis in profile.get(
        "grasp_toward_feature_axes", ("x", "y", "z")
    ))
    invalid = set(axes) - {"x", "y", "z"}
    if invalid:
        raise ValueError(f"invalid grasp displacement axes: {sorted(invalid)}")
    origin = np.array([obb["center"][axis] for axis in ("x", "y", "z")], dtype=np.float64)
    feature = np.array([feature_center[axis] for axis in ("x", "y", "z")], dtype=np.float64)
    direction = feature - origin
    for index, axis in enumerate(("x", "y", "z")):
        if axis not in axes:
            direction[index] = 0.0
    distance = float(np.linalg.norm(direction))
    if distance <= 1.0e-9:
        return
    minimum_separation = max(
        0.0, float(profile.get("minimum_grasp_feature_separation_m", 0.0))
    )
    displacement = min(requested, max(0.0, distance - minimum_separation))
    shifted = origin + displacement * direction / distance
    for index, axis in enumerate(("x", "y", "z")):
        if axis in axes:
            obb["center"][axis] = float(shifted[index])


def run(ctx: NodeContext, observation: dict[str, Any],
        instruction: str = "",
        selected_mask: np.ndarray | None = None,
        object_description: str | None = None,
        feature_description: str | None = None,
        feature_type: str | None = None,
        object_kind: str | None = None,
        feature_profiles: list[dict[str, Any]] | None = None) -> Output:
    camera = _overhead(observation)
    if not feature_profiles:
        raise ValueError("feature_profiles must declare object semantics and geometry strategies")
    profiles = {str(profile["kind"]): profile for profile in feature_profiles}
    instruction_lower = str(instruction).lower()
    mentioned = [profile for profile in feature_profiles if any(
        str(alias).lower() in instruction_lower
        for alias in (profile.get("aliases") or [profile["kind"]])
    )]
    candidates = mentioned or list(feature_profiles)
    if object_description is not None:
        if object_kind is None or feature_description is None:
            raise ValueError("explicit object perception requires object_kind and feature_description")
        base = dict(profiles.get(object_kind, {}))
        base.update({"kind": object_kind, "object_query": object_description,
                     "feature_query": feature_description})
        candidates = [base]
    detections = []
    if selected_mask is not None and np.asarray(selected_mask).size > 1:
        profile = candidates[0]
        detections.append((1.0, profile, selected_mask))
    else:
        for profile in candidates:
            result = ctx.tool(
                "sam3.segment_text", image=camera["rgb"],
                query=str(profile["object_query"]), max_results=1
            )
            if result["masks"] and result["scores"]:
                detections.append((float(result["scores"][0]), profile, result["masks"][0]))
    if not detections:
        raise ValueError("SAM3 could not localize a declared parent object")
    score, profile, tool_mask = max(detections, key=lambda item: item[0])
    kind = str(profile["kind"])
    feature_query = str(profile["feature_query"])
    feature_threshold = float(profile.get("feature_threshold", 0.05))
    if score < float(profile.get("object_score_min", 0.12)):
        raise ValueError(f"best parent detection was too weak ({kind}: {score:.3f})")
    cloud = ctx.tool(
        "geometry.mask_to_world_points", mask=tool_mask, depth=camera["depth"],
        intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
    )["points"]
    cloud = _complete_to_support(ctx, camera, tool_mask, cloud)
    obb = ctx.tool("geometry.filter_and_compute_obb", points=cloud)["obb"]
    selected_center = dict(obb["center"])

    feature_radius = 0.0
    strategy = str(profile["strategy"])
    if strategy == "planar_cad_loop":
        # Empty openings show the support rather than the parent in depth. Use
        # the semantic mask for XY and the segmented tool surface for Z.
        # Retain the old silhouette estimate only as a degraded fallback.
        try:
            feature_center = _feature_center_from_mask(
                ctx, camera, feature_query, feature_threshold, obb["center"]["z"],
                selected_center,
            )
        except (ValueError, KeyError):
            feature_center = {
                "x": float(obb["center"]["x"] + 0.43 * float(obb["extent"]["x"])),
                "y": float(obb["center"]["y"] + 0.85 * float(obb["extent"]["y"])),
                "z": float(obb["center"]["z"]),
            }
        # Recover the parent's rigid planar frame from its observed landmark and
        # silhouette.  The old interpolation between those two points moved
        # the grasp 8--12 mm sideways depending on which pixels SAM assigned
        # to the empty hole.  Both CAD bow centres and the designed grasp point
        # are known geometric landmarks, so use them after determining which
        # of the two symmetric bows was observed.
        tool_points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        model_path = str(profile["model_path"])
        planar_rotation, object_origin_xy, registration_rmse = _register_planar_model(
            tool_points, model_path
        )
        bow_candidates = tuple(
            np.asarray(landmark, dtype=np.float64)
            for landmark in profile["feature_landmarks_xy"]
        )
        semantic_xy = np.array(
            [feature_center["x"], feature_center["y"]], dtype=np.float64
        )
        bow_local = min(
            bow_candidates,
            key=lambda local: float(np.linalg.norm(
                planar_rotation @ local + object_origin_xy - semantic_xy
            )),
        )
        feature_xy = planar_rotation @ bow_local + object_origin_xy
        feature_center["x"], feature_center["y"] = map(float, feature_xy)
        feature_center["z"] = float(np.mean(tool_points[:, 2]))
        print(
            f"[perceive_object_feature] planar CAD registration: "
            f"rmse={1000.0 * registration_rmse:.2f}mm"
        )
        # Use the complete segmented tool for the grasp point. A finger-hole
        # mask has no object depth and SAM often covers one coloured rim rather
        # than the empty opening; extrapolating a CAD origin from that mask
        # displaced the grasp by 21--33 mm and put the jaws in open air. The
        # surface-cloud centroid lies on the solid bridge between the handles
        # and blades (4--6 mm from the measured grasp landmark in the recorded
        # randomized scenes), while remaining independent of world side, arm,
        # and object identity. The hole detection is still used below for the
        # functional mating feature, not for grasping.
        if len(tool_points) < 20:
            raise ValueError("parent cloud contains too few points for a stable grasp")
        grasp_xy = np.mean(tool_points[:, :2], axis=0)
        obb["center"]["x"], obb["center"]["y"] = map(float, grasp_xy)
        rotation = np.eye(3)
        rotation[:2, :2] = planar_rotation
        quat = Rotation.from_matrix(rotation).as_quat()
        feature_pose = {
            "position": dict(feature_center),
            "rotation": {
                "w": float(quat[3]), "x": float(quat[0]),
                "y": float(quat[1]), "z": float(quat[2]),
            },
        }
        feature_radius = float(profile.get("radius_inner", 0.0))
        feature_local_center = np.array([bow_local[0], bow_local[1], 0.0], dtype=np.float64)
    elif strategy == "landmark_offsets":
        # The empty handle gap has no tool depth and its semantic mask is
        # unstable: in seed 3 it landed 50 mm down one handle. The complete
        # segmented cloud, however, locates the CAD origin to 0.2--1.4 mm.
        # Transform caller-declared model landmarks into the observed support
        # frame; the estimator contains no semantic object-class branch.
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        origin = np.median(points, axis=0)
        hang = origin + np.asarray(profile["feature_offset"], dtype=np.float64)
        # Grip farther down the two handles instead of near their narrow
        # junction.  The broader section is much less likely to escape a
        # mirrored arm's jaws during the upright carry.  This shared landmark
        # intentionally serves both Level 1 and each Level 2 object.
        grasp = origin + np.asarray(profile["grasp_offset"], dtype=np.float64)
        feature_center = {
            "x": float(hang[0]), "y": float(hang[1]), "z": float(hang[2])
        }
        feature_pose = _point_pose(feature_center)
        obb["center"] = {
            "x": float(grasp[0]), "y": float(grasp[1]), "z": float(grasp[2])
        }
        for axis, maximum in profile.get("grasp_extent_max", {}).items():
            obb["extent"][axis] = min(float(obb["extent"][axis]), float(maximum))
    elif strategy == "distal_tip":
        handle_mask = _best_mask(
            ctx, camera["rgb"], str(profile["reference_query"]),
            float(profile.get("reference_threshold", 0.05)), camera, selected_center
        )
        handle_center = _mask_center_world(ctx, camera, handle_mask)
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        handle = np.array([handle_center[k] for k in ("x", "y", "z")], dtype=np.float64)
        distances = np.linalg.norm(points - handle, axis=1)
        distal = points[distances >= np.percentile(
            distances, float(profile.get("distal_percentile", 97.0))
        )]
        if len(distal) < 4:
            raise ValueError("parent cloud does not expose a distal endpoint")
        tip = np.median(distal, axis=0)
        feature_center = {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])}
        obb["center"] = handle_center
        grasp_extent = profile.get(
            "grasp_extent", {"x": 0.025, "y": 0.012, "z": 0.012}
        )
        obb["extent"] = {
            axis: float(grasp_extent[axis]) for axis in ("x", "y", "z")
        }
        tool_axis = np.array(
            [feature_center[k] - handle_center[k] for k in ("x", "y", "z")],
            dtype=np.float64,
        )
        if float(np.linalg.norm(tool_axis)) < float(profile.get("minimum_axis_length", 0.03)):
            raise ValueError("distal feature and reference part do not define a usable axis")
        feature_pose = _axis_pose(feature_center, tool_axis)
    elif strategy == "distal_planar_loop":
        feature_mask = _best_distal_loop_mask(ctx, camera, obb, profile)
        feature_center = _mask_center_world(ctx, camera, feature_mask)
        # The overhead camera sees the broad face of the planar loop. Its
        # optical direction resolves the otherwise unavoidable +/- plane-normal
        # ambiguity without consulting simulator object state.
        camera_position = camera["pose"]["position"]
        hint = {
            "x": float(camera_position["x"] - feature_center["x"]),
            "y": float(camera_position["y"] - feature_center["y"]),
            "z": float(camera_position["z"] - feature_center["z"]),
        }
        fit = _feature_fit(ctx, camera, feature_mask, hint)
        feature_pose = fit["pose"]
        feature_radius = float(fit.get("radius_inner", fit.get("radius", 0.0)))
    else:
        raise ValueError(f"unsupported manipulation feature strategy {strategy!r}")
    _shift_grasp_toward_feature(obb, feature_center, profile)
    feature_kind = feature_type or str(profile.get("feature_type", "loop"))
    fq = feature_pose["rotation"]
    feature_axis = Rotation.from_quat([fq["x"], fq["y"], fq["z"], fq["w"]]).as_matrix()[:, 2]
    return {
        "target_kind": kind,
        "relation": str(profile["relation"]),
        "target_obb": obb,
        "target_mask": tool_mask,
        "target_cloud": cloud,
        "feature_center": feature_center,
        "feature_pose": feature_pose,
        "functional_feature": {
            "kind": feature_kind,
            "description": feature_query,
            "pose": feature_pose,
            "axis": {
                "x": float(feature_axis[0]), "y": float(feature_axis[1]), "z": float(feature_axis[2])
            },
            "confidence": float(score),
            "radius_inner": feature_radius,
            "registration_profile": dict(profile.get(
                "registration_profile", {"strategy": "semantic_feature"}
            )),
            **({"local_center": [float(v) for v in feature_local_center]}
               if strategy == "planar_cad_loop" else {}),
        },
    }
