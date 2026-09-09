"""Canonical registering-held-objects implementation from Open Robot Skills."""

from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

class Output(TypedDict):
    feature_in_tcp: dict[str, Any]
    object_in_tcp: dict[str, Any]
    attached_object: dict[str, Any]
    registration_confidence: float
    registration_method: str
    fallback_used: bool
    translation_uncertainty_m: float


def _matrix(pose):
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def _pose(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "position": {"x": float(matrix[0, 3]), "y": float(matrix[1, 3]), "z": float(matrix[2, 3])},
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _align_normal_preserving_roll(base: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Use an observed loop normal without inventing rotation within its plane.

    A circle determines its plane normal but has no observable in-plane roll.
    ``fit_planar_feature`` may therefore return an arbitrary roll that changes
    between cameras.  Apply only the minimum rotation needed to align normals
    and preserve the prior rigid feature frame around that normal.
    """
    old_normal = np.asarray(base[:3, 2], dtype=np.float64)
    new_normal = np.asarray(observed[:3, 2], dtype=np.float64)
    old_normal /= max(float(np.linalg.norm(old_normal)), 1.0e-12)
    new_normal /= max(float(np.linalg.norm(new_normal)), 1.0e-12)
    if float(old_normal @ new_normal) < 0.0:
        new_normal = -new_normal
    correction, _ = Rotation.align_vectors([new_normal], [old_normal])
    result = base.copy()
    result[:3, :3] = correction.as_matrix() @ base[:3, :3]
    return result


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


@lru_cache(maxsize=8)
def _model_points(model_path: str, model_scale: float) -> np.ndarray:
    """Load deterministic surface samples from a caller-declared CAD model."""
    import trimesh

    mesh = trimesh.load_mesh(Path(model_path), process=False)
    # Dense sampling matters here: using only vertices and triangle centroids
    # left centimetre-scale holes on broad faces and biased nearest neighbours.
    points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    return float(model_scale) * points


def _refine_scissors_from_wrist(observed: np.ndarray, initial: np.ndarray,
                                profile: dict[str, Any]):
    """Recover in-plane settling without letting one-sided depth move the plane."""
    from scipy.spatial import cKDTree

    observed = observed[np.all(np.isfinite(observed), axis=1)]
    if len(observed) < 80:
        return None, float("inf"), 0
    pose = initial.copy()
    model = _model_points(str(profile["model_path"]), float(profile.get("model_scale", 1.0)))
    normal = pose[:3, :3] @ np.asarray(profile["normal_local"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    total = np.zeros(3)
    for _ in range(15):
        world_model = (pose[:3, :3] @ model.T).T + pose[:3, 3]
        distances, indices = cKDTree(world_model).query(observed, k=1)
        cutoff = min(0.010, float(np.quantile(distances, 0.70)))
        keep = distances <= max(cutoff, 0.0015)
        if int(np.count_nonzero(keep)) < 60:
            return None, float("inf"), int(np.count_nonzero(keep))
        delta = np.median(observed[keep] - world_model[indices[keep]], axis=0)
        delta -= normal * float(delta @ normal)
        pose[:3, 3] += delta
        total += delta
        if float(np.linalg.norm(delta)) < 2e-5:
            break
    rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
    inliers = int(np.count_nonzero(keep))
    if float(np.linalg.norm(total)) > 0.020 or rmse > 0.006:
        return None, rmse, inliers
    return pose, rmse, inliers


def _scissors_pose_from_feature(functional_feature: dict[str, Any]) -> np.ndarray:
    feature = _matrix(functional_feature["pose"])
    local = np.asarray(functional_feature["local_center"], dtype=np.float64)
    result = feature.copy()
    result[:3, 3] = feature[:3, 3] - feature[:3, :3] @ local
    return result


def _feature_from_scissors_pose(scissors: np.ndarray, local_center) -> np.ndarray:
    result = scissors.copy()
    result[:3, 3] = scissors[:3, :3] @ np.asarray(local_center, dtype=np.float64) + scissors[:3, 3]
    return result


def _initial_wrench_pose(reference_cloud: dict[str, Any], feature: np.ndarray,
                         profile: dict[str, Any]) -> np.ndarray:
    points = np.asarray(reference_cloud["points"], dtype=np.float64).reshape(-1, 3)
    center, ring = np.median(points, axis=0), feature[:3, 3]
    normal = feature[:3, 2]
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    long_axis = ring - center
    long_axis -= normal * float(long_axis @ normal)
    long_axis /= max(float(np.linalg.norm(long_axis)), 1.0e-12)
    side_axis = np.cross(long_axis, normal)
    side_axis /= max(float(np.linalg.norm(side_axis)), 1.0e-12)
    rotation = np.column_stack((long_axis, normal, side_axis))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0
    result = np.eye(4)
    result[:3, :3] = rotation
    feature_local = np.asarray(profile["feature_local"], dtype=np.float64)
    result[:3, 3] = ring - rotation @ feature_local
    return result


def _register_wrench_cad(observed: np.ndarray, initial: np.ndarray,
                         profile: dict[str, Any]):
    """Dense trimmed rigid ICP, used on the complete pre-grasp observation."""
    from scipy.spatial import cKDTree

    observed = observed[np.all(np.isfinite(observed), axis=1)]
    if len(observed) < 80:
        return None, float("inf"), 0
    pose = initial.copy()
    model = _model_points(str(profile["model_path"]), float(profile.get("model_scale", 1.0)))
    inliers, rmse = 0, float("inf")
    for _ in range(12):
        world_model = (pose[:3, :3] @ model.T).T + pose[:3, 3]
        distances, indices = cKDTree(world_model).query(observed, k=1)
        cutoff = min(0.012, float(np.quantile(distances, 0.75)))
        keep = distances <= max(cutoff, 0.0015)
        if int(np.count_nonzero(keep)) < 60:
            return None, float("inf"), int(np.count_nonzero(keep))
        delta = _rigid_fit(world_model[indices[keep]], observed[keep])
        pose = delta @ pose
        inliers = int(np.count_nonzero(keep))
        rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
        if np.linalg.norm(delta[:3, 3]) < 2e-5 and Rotation.from_matrix(delta[:3, :3]).magnitude() < np.deg2rad(0.05):
            break
    translation_jump = float(np.linalg.norm(pose[:3, 3] - initial[:3, 3]))
    rotation_jump = float(Rotation.from_matrix(pose[:3, :3] @ initial[:3, :3].T).magnitude())
    if rmse > 0.005 or translation_jump > 0.025 or rotation_jump > np.deg2rad(18.0):
        return None, rmse, inliers
    return pose, rmse, inliers


def _refine_wrench_from_wrist(observed: np.ndarray, initial: np.ndarray,
                              profile: dict[str, Any]):
    """Refine only observable translation from the partial wrist views.

    The gripper hides the ring and each depth image sees only one wrench face.
    A free 6-DoF ICP can consequently slide the CAD by its half-thickness or
    rotate its distant ring by centimetres. The complete pre-grasp cloud gives
    orientation; wrist depth safely corrects translation in the ring plane.
    """
    from scipy.spatial import cKDTree

    observed = observed[np.all(np.isfinite(observed), axis=1)]
    if len(observed) < 80:
        return None, float("inf"), 0
    pose = initial.copy()
    model = _model_points(str(profile["model_path"]), float(profile.get("model_scale", 1.0)))
    normal = pose[:3, :3] @ np.asarray(profile["normal_local"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    total = np.zeros(3)
    for _ in range(15):
        world_model = (pose[:3, :3] @ model.T).T + pose[:3, 3]
        distances, indices = cKDTree(world_model).query(observed, k=1)
        cutoff = min(0.012, float(np.quantile(distances, 0.75)))
        keep = distances <= max(cutoff, 0.0015)
        if int(np.count_nonzero(keep)) < 60:
            return None, float("inf"), int(np.count_nonzero(keep))
        delta = np.mean(observed[keep] - world_model[indices[keep]], axis=0)
        delta -= normal * float(delta @ normal)
        pose[:3, 3] += delta
        total += delta
        if float(np.linalg.norm(delta)) < 2e-5:
            break
    rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
    inliers = int(np.count_nonzero(keep))
    if float(np.linalg.norm(total)) > 0.015 or rmse > 0.006:
        return None, rmse, inliers
    return pose, rmse, inliers


def _anchor_wrench_plane(
    pose: np.ndarray, coarse: np.ndarray, reference_points: np.ndarray,
    profile: dict[str, Any],
) -> np.ndarray:
    """Keep ICP's in-plane fit but recover plane normal/thickness from RGB-D.

    A flat wrench gives weak point-to-point constraints along its thickness,
    which allowed a few degrees of pitch to move the distant ring by 5--10 mm.
    The initial fitted loop plane supplies the normal, while the clearly
    separated top-depth mode supplies the CAD mid-plane coordinate.
    """
    old_normal = pose[:3, 1].copy()
    # Level-1 wrench construction places the tool flat on the horizontal
    # table. A 2--3 degree normal error from the thin ring fit becomes a
    # 7--9 mm height error at the ring's 170 mm lever arm, so use the observed
    # support-plane normal exactly for this pre-grasp anchor.
    new_normal = np.array([0.0, 0.0, 1.0 if coarse[2, 1] >= 0.0 else -1.0])
    old_normal /= max(float(np.linalg.norm(old_normal)), 1e-12)
    new_normal /= max(float(np.linalg.norm(new_normal)), 1e-12)
    if float(old_normal @ new_normal) < 0.0:
        new_normal = -new_normal
    correction, _ = Rotation.align_vectors([new_normal], [old_normal])
    pose = pose.copy()
    pose[:3, :3] = correction.as_matrix() @ pose[:3, :3]

    points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    # Ring holes contribute table pixels. The upper world-Z mode is the metal
    # top face and is cleanly separated by roughly the wrench thickness.
    tool_face = points[points[:, 2] > float(np.quantile(points[:, 2], 0.60))]
    normal = pose[:3, 1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    points_along_normal = tool_face @ normal
    upward = float(normal[2]) >= 0.0
    surface = float(np.median(points_along_normal))
    model = _model_points(str(profile["model_path"]), float(profile.get("model_scale", 1.0)))
    normal_index = int(profile.get("normal_axis_index", 1))
    support = float(np.max(model[:, normal_index]) if upward else np.min(model[:, normal_index]))
    desired_origin = surface - support
    pose[:3, 3] += normal * (desired_origin - float(pose[:3, 3] @ normal))
    return pose


def _feature_from_wrench_pose(wrench: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    x_axis = wrench[:3, 0]
    normal_local = np.asarray(profile["normal_local"], dtype=np.float64)
    feature_local = np.asarray(profile["feature_local"], dtype=np.float64)
    z_axis = wrench[:3, :3] @ normal_local
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)
    x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
    y_axis = np.cross(z_axis, x_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = wrench[:3, :3] @ feature_local + wrench[:3, 3]
    return result


def _recover_loop_from_overview(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    predicted: np.ndarray,
    description: str,
    profile: dict[str, Any],
):
    """Recover a held loop when the close wrist view is self-occluded.

    Overview detections are associated geometrically with the rigid prediction,
    rather than by detection index. This remains valid when another object has
    already been placed elsewhere in the scene.
    """
    maximum_jump = float(profile.get("overview_feature_jump_m", 0.035))
    camera_names = set(profile.get("overview_camera_names", ["overhead", "side"]))
    normal = predicted[:3, 2].copy()
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    candidates = []
    for camera in cameras:
        if str(camera.get("name", "")) not in camera_names:
            continue
        result = ctx.tool(
            "sam3.segment_text", image=camera["rgb"], query=description,
            max_results=int(profile.get("overview_max_results", 3)),
        )
        for mask, raw_score in zip(
            result.get("masks") or [], result.get("scores") or []
        ):
            score = float(raw_score)
            if score < float(profile.get("overview_minimum_score", 0.05)):
                continue
            cloud = ctx.tool(
                "geometry.mask_to_world_points",
                mask=np.asarray(mask, dtype=np.uint8), depth=camera["depth"],
                intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
            )["points"]
            points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
            if len(points) < 20:
                continue
            try:
                fit = ctx.tool(
                    "geometry.fit_planar_feature", points=cloud,
                    normal_hint={"x": float(normal[0]), "y": float(normal[1]),
                                 "z": float(normal[2])},
                    fit_circle_center=True,
                )
            except Exception:
                continue
            # A distant planar loop localizes its centre well, but its normal
            # and in-plane roll are ambiguous under partial occlusion. Keep the
            # rigid prior orientation and use only the associated metric centre.
            observed = predicted.copy()
            center = np.array(
                [fit["pose"]["position"][key] for key in ("x", "y", "z")],
                dtype=np.float64,
            )
            jump = float(np.linalg.norm(center - predicted[:3, 3]))
            if jump <= maximum_jump:
                observed[:3, 3] = center
                candidates.append((jump, -score, observed, score))
    if not candidates:
        return None, 0.0
    _, _, observed, score = min(candidates, key=lambda item: item[:2])
    return observed, score


def _fit_attachment(cloud: dict[str, Any], world_tcp: np.ndarray) -> dict[str, Any]:
    """Fit the observed held object using CuRobo's selectable sphere fitter."""
    import trimesh
    from curobo._src.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh

    points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    if len(points) < 8:
        raise ValueError("attachment sphere fitting needs at least eight object points")
    tcp_world = np.linalg.inv(world_tcp)
    local = (tcp_world @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    # MORPHIT consumes a mesh. The observed wrench cloud supplies its external
    # occupied volume; its convex hull is deterministic, watertight, and avoids
    # feeding the fitter a non-manifold RGB-D alpha surface.
    mesh = trimesh.points.PointCloud(local).convex_hull
    result = fit_spheres_to_mesh(
        mesh,
        num_spheres=64,
        surface_radius=0.002,
        fit_type=SphereFitType.MORPHIT,
        iterations=200,
        compute_metrics=True,
    )
    centers = result.centers.detach().cpu().numpy().reshape(-1, 3)
    radii = result.radii.detach().cpu().numpy().reshape(-1)
    shrink = 0.002
    radii = np.maximum(radii - shrink, 0.001)
    valid = np.isfinite(radii) & (radii > 0.0)
    if not np.any(valid):
        raise RuntimeError("CuRobo MORPHIT fitting returned no positive object spheres")
    return {
        "frame": "tcp",
        "sphere_fit_type": SphereFitType.MORPHIT.value,
        "sphere_radius_shrink_m": shrink,
        "spheres": [
            {
                "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "radius": float(radius),
            }
            for center, radius in zip(centers[valid], radii[valid])
        ],
    }


def run(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    reference_cloud: dict[str, Any],
    functional_feature: dict[str, Any],
    object_description: str,
    prior_feature_in_tcp: dict[str, Any] | None = None,
    prior_object_in_tcp: dict[str, Any] | None = None,
    prior_attached_object: dict[str, Any] | None = None,
    arm_id: int | None = None,
) -> Output:
    profile = dict(functional_feature.get("registration_profile") or {
        "strategy": "semantic_feature"
    })
    strategy = str(profile.get("strategy", "semantic_feature"))
    if arm_id is None:
        if prior_attached_object is not None and "arm_id" in prior_attached_object:
            arm_id = int(prior_attached_object["arm_id"])
        else:
            feature_y = float(functional_feature["pose"]["position"]["y"])
            arm_id = 0 if feature_y >= 0.0 else 1
    wrist_name = "eye_in_hand_left" if int(arm_id) == 0 else "eye_in_hand_right"
    ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
    world_tcp = _matrix(ee)
    # A small visible ring arc was not enough to locate the held wrench
    # reliably. Register all wrench pixels from both wrist cameras to the CAD,
    # then derive the exact CAD ring point from that rigid pose.
    wrench_registration = None
    wrench_registration_confidence = 0.0
    preserve_wrench_prior = False
    if strategy == "cad_distal_loop":
        full_clouds, full_scores = [], []
        # Prefer the active wrist. If it is occluded by the gripper/handle,
        # use the opposite wrist as a remote observer and bind its detections
        # to the held instance by proximity to the active TCP.
        wrist_cameras = [
            camera for camera in cameras
            if str(camera.get("name", "")).startswith("eye_in_hand_")
        ]
        wrist_cameras.sort(key=lambda camera: camera.get("name") != wrist_name)
        active_tcp_position = world_tcp[:3, 3]
        for camera in wrist_cameras:
            # Only consult the remote wrist when the active wrist yielded no
            # usable cloud. This avoids merging two different viewpoints or
            # accidentally including the wrench already hanging nearby.
            if camera.get("name") != wrist_name and full_clouds:
                break
            result = ctx.tool(
                "sam3.segment_text", image=camera["rgb"],
                query=str(profile["query"]), max_results=int(profile.get("max_results", 3)),
            )
            localized = []
            for mask, raw_score in zip(
                result.get("masks") or [], result.get("scores") or []
            ):
                score = float(raw_score)
                if score < 0.08:
                    continue
                points = ctx.tool(
                    "geometry.mask_to_world_points",
                    mask=np.asarray(mask, dtype=np.uint8),
                    depth=camera["depth"], intrinsics=camera["intrinsics"],
                    camera_pose=camera["pose"],
                )["points"]
                array = np.asarray(points["points"], dtype=np.float64).reshape(-1, 3)
                if len(array) < 40:
                    continue
                distance_to_tcp = float(
                    np.linalg.norm(np.median(array, axis=0) - active_tcp_position)
                )
                if distance_to_tcp <= 0.30:
                    localized.append((distance_to_tcp, -score, array, score))
            if localized:
                _, _, array, score = min(localized, key=lambda item: item[:2])
                full_clouds.append(array)
                full_scores.append(score)
                if camera.get("name") != wrist_name:
                    print(
                        f"[register_held] active wrist occluded; using "
                        f"{camera.get('name')} remote observation"
                    )
        if full_clouds:
            if prior_object_in_tcp is not None:
                initial = world_tcp @ _matrix(prior_object_in_tcp)
            else:
                coarse = _initial_wrench_pose(
                    reference_cloud, _matrix(functional_feature["pose"]), profile
                )
                initial, reference_rmse, reference_inliers = _register_wrench_cad(
                    np.asarray(reference_cloud["points"], dtype=np.float64).reshape(-1, 3),
                    coarse, profile,
                )
                if initial is None:
                    initial = coarse
                    print(f"[register_held] pre-grasp wrench CAD ICP rejected: "
                          f"rmse={1000*reference_rmse:.2f}mm inliers={reference_inliers}")
                else:
                    initial = _anchor_wrench_plane(
                        initial,
                        coarse,
                        np.asarray(reference_cloud["points"], dtype=np.float64),
                        profile,
                    )
                    print(f"[register_held] pre-grasp wrench CAD ICP accepted: "
                          f"rmse={1000*reference_rmse:.2f}mm inliers={reference_inliers}")
            wrench_registration, wrench_rmse, wrench_inliers = _refine_wrench_from_wrist(
                np.vstack(full_clouds), initial, profile
            )
            if wrench_registration is not None:
                wrench_registration_confidence = max(full_scores) * float(np.exp(-wrench_rmse / 0.006))
                print(f"[register_held] constrained wrist CAD refinement accepted: rmse={1000*wrench_rmse:.2f}mm "
                      f"inliers={wrench_inliers} confidence={wrench_registration_confidence:.3f}")
            else:
                # On the first pass the pre-grasp CAD estimate is still the
                # safest prior. Near the hook, however, occlusion can leave too
                # little whole-wrench support while the ring-specific mask is
                # usable. Let the generic loop fitter below provide that final
                # translational correction instead of freezing a stale pose.
                if prior_object_in_tcp is None:
                    wrench_registration = initial
                    wrench_registration_confidence = 0.25
                    fallback = "preserving pre-grasp CAD prior"
                else:
                    # Once the jaws are closed, feature_in_tcp is a rigid
                    # transform. Near a hook the wrist usually sees a handle,
                    # finger, or tiny ring arc; feeding that partial mask to a
                    # circle fit moved the distant ring by centimetres in
                    # Level-2 seed 4. Preserve the last validated transform
                    # instead of inventing a non-rigid correction.
                    wrench_registration = None
                    preserve_wrench_prior = True
                    fallback = "preserving rigid TCP prior"
                print(f"[register_held] constrained wrist CAD refinement rejected; {fallback}: "
                      f"rmse={1000*wrench_rmse:.2f}mm inliers={wrench_inliers}")
    scissors_registration = None
    scissors_registration_confidence = 0.0
    if (strategy == "planar_cad_landmark"
            and functional_feature.get("local_center") is not None):
        full_clouds, full_scores = [], []
        for camera in cameras:
            if camera.get("name") != wrist_name:
                continue
            result = ctx.tool(
                "sam3.segment_text", image=camera["rgb"],
                query=str(profile["query"]), max_results=int(profile.get("max_results", 2)),
            )
            if not result.get("masks") or not result.get("scores"):
                continue
            score = float(result["scores"][0])
            if score < 0.08:
                continue
            points = ctx.tool(
                "geometry.mask_to_world_points",
                mask=np.asarray(result["masks"][0], dtype=np.uint8),
                depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"],
            )["points"]
            array = np.asarray(points["points"], dtype=np.float64).reshape(-1, 3)
            if len(array) >= 40:
                full_clouds.append(array)
                full_scores.append(score)
        if full_clouds:
            initial = (world_tcp @ _matrix(prior_object_in_tcp)
                       if prior_object_in_tcp is not None
                       else _scissors_pose_from_feature(functional_feature))
            scissors_registration, scissors_rmse, scissors_inliers = _refine_scissors_from_wrist(
                np.vstack(full_clouds), initial, profile
            )
            if scissors_registration is not None:
                scissors_registration_confidence = max(full_scores) * float(
                    np.exp(-scissors_rmse / 0.006)
                )
                print(f"[register_held] scissors wrist CAD refinement accepted: "
                      f"rmse={1000*scissors_rmse:.2f}mm inliers={scissors_inliers} "
                      f"confidence={scissors_registration_confidence:.3f}")
            else:
                print(f"[register_held] scissors wrist CAD refinement rejected: "
                      f"rmse={1000*scissors_rmse:.2f}mm inliers={scissors_inliers}")
    best = None
    for camera in cameras:
        name = camera.get("name", "")
        # A held loop is measured only from the original eye-in-hand cameras.
        # Their close view resolves the ring plane much better than the distant
        # overhead view. Other feature types retain the overhead path.
        if functional_feature.get("kind") == "loop":
            if name != wrist_name:
                continue
        elif name != "overhead":
            continue
        # A thin tip is often only a few pixels wide in an overview camera.
        # Segment the complete held object and recover its distal endpoint in
        # metric 3D instead of asking the segmenter for the tip alone.
        query = (functional_feature.get("description", object_description)
                 if functional_feature.get("kind") == "loop"
                 else object_description)
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=2)
        if result.get("masks") and result.get("scores"):
            score = float(result["scores"][0])
            if best is None or score > best[0]:
                best = (score, camera, result["masks"][0])
    cloud = reference_cloud
    confidence = 0.25
    if best is not None and best[0] >= 0.05:
        confidence = best[0]
        camera, mask = best[1], best[2]
        observed = ctx.tool(
            "geometry.mask_to_world_points",
            mask=np.asarray(mask, dtype=np.uint8),
            depth=camera["depth"],
            intrinsics=camera["intrinsics"],
            camera_pose=camera["pose"],
        )["points"]
        observed_points = np.asarray(observed["points"]).reshape(-1, 3)
        if len(observed_points) >= 8:
            accept_observed = True
            if prior_feature_in_tcp is None and functional_feature.get("kind") != "loop":
                reference_points = np.asarray(reference_cloud["points"]).reshape(-1, 3)
                reference_size = float(np.linalg.norm(np.ptp(reference_points, axis=0)))
                observed_size = float(np.linalg.norm(np.ptp(observed_points, axis=0)))
                ratio = observed_size / max(reference_size, 1.0e-6)
                # Overview segmentation can merge a held tool with the arm.
                # Reject a grossly inconsistent 3D extent before it becomes an
                # enormous attached collision model and clearance waypoint.
                if not 0.55 <= ratio <= 1.80:
                    accept_observed = False
                    confidence = 0.25
            if accept_observed:
                cloud = observed
    overview_loop_registration = None
    overview_loop_confidence = 0.0
    if (
        prior_feature_in_tcp is not None
        and functional_feature.get("kind") == "loop"
        and (
            preserve_wrench_prior
            or best is None
            or best[0] < 0.20
            or (
                wrench_registration is not None
                and wrench_registration_confidence
                < float(profile.get("overview_trigger_confidence", 0.20))
            )
        )
        and bool(profile.get("overview_loop_fallback", True))
    ):
        predicted = world_tcp @ _matrix(prior_feature_in_tcp)
        overview_loop_registration, overview_loop_confidence = (
            _recover_loop_from_overview(
                ctx, cameras, predicted,
                str(functional_feature.get("description", object_description)),
                profile,
            )
        )
        if overview_loop_registration is not None:
            print(
                "[register_held] wrist feature occluded; accepted associated "
                f"overview loop observation confidence={overview_loop_confidence:.3f}"
            )
    reliable_registration = (
        best is not None and best[0] >= 0.20 and not preserve_wrench_prior
    )
    registration_correction = np.zeros(3, dtype=np.float64)
    if overview_loop_registration is not None:
        predicted = world_tcp @ _matrix(prior_feature_in_tcp)
        world_feature = overview_loop_registration
        registration_correction = world_feature[:3, 3] - predicted[:3, 3]
        confidence = overview_loop_confidence
        preserve_wrench_prior = False
    elif scissors_registration is not None:
        world_feature = _feature_from_scissors_pose(
            scissors_registration, functional_feature["local_center"]
        )
        confidence = scissors_registration_confidence
    elif wrench_registration is not None:
        world_feature = _feature_from_wrench_pose(wrench_registration, profile)
        confidence = wrench_registration_confidence
    elif prior_feature_in_tcp is not None and reliable_registration:
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
        points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
        if functional_feature.get("kind") == "loop":
            normal = world_feature[:3, 2]
            fit = ctx.tool(
                "geometry.fit_planar_feature",
                points=cloud,
                normal_hint={
                    "x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])
                },
                fit_circle_center=True,
            )
            fitted_feature = _matrix(fit["pose"])
            world_feature = _align_normal_preserving_roll(world_feature, fitted_feature)
            observed_center = np.array(
                [fit["pose"]["position"][key] for key in ("x", "y", "z")],
                dtype=np.float64,
            )
            # An oblique view may expose primarily one face of a thick ring. The
            # circle fit recovers its in-plane centre; use the robust midpoint
            # of observed thickness for the coordinate along the plane normal.
            normal = world_feature[:3, 2]
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            projected = points @ normal
            axial_center = 0.5 * (
                float(np.quantile(projected, 0.02))
                + float(np.quantile(projected, 0.98))
            )
            observed_center += (axial_center - float(observed_center @ normal)) * normal
        else:
            observed_center = np.median(points, axis=0)
        # This pass segments the functional feature, so compare its observed
        # centre with the predicted feature centre—not the whole-object centre.
        predicted_center = world_feature[:3, 3]
        # A rigidly held feature cannot jump far from the pose predicted by
        # its prior TCP transform. Reject masks on the gripper or background.
        if float(np.linalg.norm(observed_center - predicted_center)) <= 0.04:
            registration_correction = observed_center - predicted_center
            world_feature[:3, 3] += registration_correction
        else:
            confidence = 0.25
    elif prior_feature_in_tcp is not None:
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
    else:
        world_feature = _matrix(functional_feature["pose"])
        if reliable_registration and functional_feature.get("kind") == "loop":
            normal = world_feature[:3, 2]
            fit = ctx.tool(
                "geometry.fit_planar_feature",
                points=cloud,
                normal_hint={
                    "x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])
                },
                fit_circle_center=True,
            )
            fitted_feature = _matrix(fit["pose"])
            world_feature = _align_normal_preserving_roll(world_feature, fitted_feature)
            points = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
            observed_center = np.array(
                [fit["pose"]["position"][key] for key in ("x", "y", "z")],
                dtype=np.float64,
            )
            normal = world_feature[:3, 2]
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            projected = points @ normal
            axial_center = 0.5 * (
                float(np.quantile(projected, 0.02))
                + float(np.quantile(projected, 0.98))
            )
            observed_center += (
                axial_center - float(observed_center @ normal)
            ) * normal
            world_feature[:3, 3] = observed_center
        elif reliable_registration:
            observed_center = np.median(
                np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3), axis=0
            )
            reference_center = np.median(
                np.asarray(reference_cloud["points"], dtype=np.float64).reshape(-1, 3), axis=0
            )
            correction = observed_center - reference_center
            if float(np.linalg.norm(correction)) <= 0.02:
                world_feature[:3, 3] += correction
            else:
                confidence = 0.25
    feature_tcp = _pose(np.linalg.inv(world_tcp) @ world_feature)
    preserve_prior_registration = False
    if prior_feature_in_tcp is not None:
        prior_matrix = _matrix(prior_feature_in_tcp)
        proposed_matrix = _matrix(feature_tcp)
        translation_jump = float(np.linalg.norm(
            proposed_matrix[:3, 3] - prior_matrix[:3, 3]
        ))
        rotation_jump = float(Rotation.from_matrix(
            proposed_matrix[:3, :3] @ prior_matrix[:3, :3].T
        ).magnitude())
        # A rigidly grasped feature is fixed in TCP coordinates. Wrist
        # re-observation may refine it by millimetres, but a centimetre-scale
        # or large angular jump means the mask/ICP latched onto another
        # symmetric surface. Reject it before mate planning turns the error
        # into a distant, apparently unreachable approach pose.
        maximum_translation_jump = float(profile.get("maximum_translation_jump_m", 0.025))
        maximum_rotation_jump = np.deg2rad(float(profile.get("maximum_rotation_jump_deg", 20.0)))
        if translation_jump > maximum_translation_jump or rotation_jump > maximum_rotation_jump:
            print(
                "[register_held] rejected non-rigid TCP feature jump: "
                f"translation={1000*translation_jump:.1f}mm "
                f"rotation={np.rad2deg(rotation_jump):.1f}deg; preserving prior"
            )
            feature_tcp = dict(prior_feature_in_tcp)
            preserve_prior_registration = True
            confidence = 0.25
    for key in ("kind", "radius_inner", "radius_outer"):
        if key in functional_feature:
            feature_tcp[key] = functional_feature[key]
    attachment = prior_attached_object
    if attachment is None:
        # The wrist mask above isolates the functional ring. Collision fitting
        # still needs the complete pre-grasp object cloud.
        reference_points = np.asarray(reference_cloud["points"]).reshape(-1, 3)
        if len(reference_points) < 8:
            attachment = ctx.tool(
                "geometry.cloud_to_attachment", points=reference_cloud,
                tcp_pose=ee, fit_type=str(profile.get("attachment_fit_type", "morphit")),
            )["attached_object"]
        else:
            attachment = _fit_attachment(reference_cloud, world_tcp)
    attachment = dict(attachment)
    attachment["arm_id"] = int(arm_id)
    if preserve_prior_registration and prior_object_in_tcp is not None:
        obj = world_tcp @ _matrix(prior_object_in_tcp)
    elif scissors_registration is not None:
        obj = scissors_registration
    elif wrench_registration is not None:
        obj = wrench_registration
    elif prior_object_in_tcp is not None:
        obj = world_tcp @ _matrix(prior_object_in_tcp)
        obj[:3, 3] += registration_correction
    else:
        center = np.median(np.asarray(reference_cloud["points"]).reshape(-1, 3), axis=0)
        obj = np.eye(4)
        obj[:3, 3] = center
    if overview_loop_registration is not None:
        registration_method = "overview_loop_feature"
    elif wrench_registration is not None:
        registration_method = "cad_distal_loop"
    elif scissors_registration is not None:
        registration_method = "planar_cad_landmark"
    elif reliable_registration:
        registration_method = "semantic_feature"
    else:
        registration_method = "rigid_grasp_prior"
    uncertainty = float(profile.get(
        "fallback_uncertainty_m" if confidence <= 0.25 else "measurement_uncertainty_m",
        0.010 if confidence <= 0.25 else 0.004,
    ))
    attachment["translation_uncertainty_m"] = uncertainty
    attachment["registration_method"] = registration_method
    return {
        "feature_in_tcp": feature_tcp,
        "object_in_tcp": _pose(np.linalg.inv(world_tcp) @ obj),
        "attached_object": attachment,
        "registration_confidence": float(confidence),
        "registration_method": registration_method,
        "fallback_used": bool(preserve_prior_registration or confidence <= 0.25),
        "translation_uncertainty_m": uncertainty,
    }
