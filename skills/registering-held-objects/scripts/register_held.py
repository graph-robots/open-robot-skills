"""Express a held functional feature and collision cloud in the TCP frame.

The default path -- no ``registration_profile`` on the feature -- is the
generic body: segment the held feature (or the whole object) from the
filtered cameras, correct the feature's TCP pose from what was observed, and
fit the attachment through the geometry/cuRobo tools. A graph that binds only
the original inputs gets exactly those calls.

``functional_feature["registration_profile"]`` lets a graph that carries one
profile per object kind ask for two CAD-backed strategies ahead of that path:

- ``cad_distal_loop``: register the WHOLE held object to a declared CAD model
  (``model_path``, ``model_scale``, ``feature_local``, ``normal_local``) with
  trimmed rigid ICP on the pre-grasp cloud, then refine translation only from
  the wrist views, and derive the ring point from the rigid pose. For a loop
  the wrist barely sees.
- ``planar_cad_landmark``: refine in-plane settling of a planar object
  against its CAD from the wrist, and derive the landmark from
  ``local_center``.

Both feed the same selection ladder the generic path uses and fall through to
it when they reject their own fit. The profile also carries the overview-loop
fallback, the rigid-jump rejection thresholds and the uncertainty figures;
none of those apply without a profile.

``arm_id`` names the hand on a bimanual cell and is threaded as an ABSENT
keyword when unset. It is never inferred from where the feature sits.
"""

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


def _matrix(pose: dict[str, Any]) -> np.ndarray:
    p, q = pose["position"], pose["rotation"]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    out[:3, 3] = [p["x"], p["y"], p["z"]]
    return out


def _pose(matrix: np.ndarray) -> dict[str, Any]:
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "position": {"x": float(matrix[0, 3]), "y": float(matrix[1, 3]), "z": float(matrix[2, 3])},
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _points(cloud: dict[str, Any]) -> np.ndarray:
    return np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)


def _transform_cloud(cloud: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    points = _points(cloud)
    moved = dict(cloud)
    moved["points"] = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    return moved


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


def _fit_loop_center(ctx: NodeContext, world_feature: np.ndarray, cloud: dict[str, Any]):
    """Fit the loop plane and circle centre; return (aligned frame, centre).

    The circle fit gives the in-plane centre and the plane normal.  An oblique
    view may expose primarily one face of a thick ring, so the coordinate along
    the normal is the robust midpoint of the observed thickness instead.
    """
    normal = world_feature[:3, 2]
    fit = ctx.tool(
        "geometry.fit_planar_feature",
        points=cloud,
        normal_hint={"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])},
        fit_circle_center=True,
    )
    aligned = _align_normal_preserving_roll(world_feature, _matrix(fit["pose"]))
    observed_center = np.array(
        [fit["pose"]["position"][key] for key in ("x", "y", "z")], dtype=np.float64
    )
    normal = aligned[:3, 2].copy()
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    projected = _points(cloud) @ normal
    axial_center = 0.5 * (float(np.quantile(projected, 0.02)) + float(np.quantile(projected, 0.98)))
    observed_center += (axial_center - float(observed_center @ normal)) * normal
    return aligned, observed_center


def _directed_tip_frame(points: np.ndarray, marker_points: np.ndarray) -> np.ndarray:
    """Frame at the distal endpoint of an elongated object, z toward its marker."""
    center = np.median(points, axis=0)
    centered = points - center
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    axis = basis[0]
    if float(axis @ (np.median(marker_points, axis=0) - center)) < 0.0:
        axis = -axis
    along = centered @ axis
    tip = np.median(points[along >= np.percentile(along, 98)], axis=0)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(reference @ axis)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(reference, axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
    y_axis = np.cross(axis, x_axis)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, axis))
    frame[:3, 3] = tip
    return frame


def _observed_tip_frame(
    ctx: NodeContext, camera: dict[str, Any], points: np.ndarray, marker_description: str
) -> np.ndarray | None:
    """Re-derive a tip from the object cloud when its direction marker is visible."""
    if len(points) < 20:
        return None
    marker = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=marker_description, max_results=2
    )
    if not marker.get("masks") or not marker.get("scores") or float(marker["scores"][0]) < 0.04:
        return None
    marker_cloud = ctx.tool(
        "geometry.mask_to_world_points",
        mask=np.asarray(marker["masks"][0], dtype=np.uint8),
        depth=camera["depth"],
        intrinsics=camera["intrinsics"],
        camera_pose=camera["pose"],
    )["points"]
    marker_points = _points(marker_cloud)
    if len(marker_points) < 6:
        return None
    return _directed_tip_frame(points, marker_points)


# --------------------------------------------------------------------------
# CAD-backed strategies (profile-driven; never reached without a profile)
# --------------------------------------------------------------------------


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
    import trimesh  # noqa: PLC0415

    mesh = trimesh.load_mesh(Path(model_path), process=False)
    # Dense sampling matters here: using only vertices and triangle centroids
    # left centimetre-scale holes on broad faces and biased nearest neighbours.
    points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    return float(model_scale) * points


def _refine_in_plane(observed: np.ndarray, initial: np.ndarray, profile: dict[str, Any], *,
                     cutoff_m: float, max_total_m: float, reducer) -> tuple[np.ndarray | None, float, int]:
    """Translate a CAD pose in its own plane until wrist depth agrees with it.

    The gripper hides part of a held object and a wrist depth image sees one
    face; a free 6-DoF ICP would slide the model by its thickness or turn a
    distant feature by centimetres. Orientation stays as given and only the
    in-plane translation moves -- the reducer (median for a planar object,
    mean for a rigid one) is the profile's choice.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    observed = observed[np.all(np.isfinite(observed), axis=1)]
    if len(observed) < 80:
        return None, float("inf"), 0
    pose = initial.copy()
    model = _model_points(str(profile["model_path"]), float(profile.get("model_scale", 1.0)))
    normal = pose[:3, :3] @ np.asarray(profile["normal_local"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    total = np.zeros(3)
    keep = np.zeros(len(observed), dtype=bool)
    distances = np.zeros(len(observed))
    for _ in range(15):
        world_model = (pose[:3, :3] @ model.T).T + pose[:3, 3]
        distances, indices = cKDTree(world_model).query(observed, k=1)
        cutoff = min(cutoff_m, float(np.quantile(distances, 0.70 if cutoff_m <= 0.010 else 0.75)))
        keep = distances <= max(cutoff, 0.0015)
        if int(np.count_nonzero(keep)) < 60:
            return None, float("inf"), int(np.count_nonzero(keep))
        delta = reducer(observed[keep] - world_model[indices[keep]], axis=0)
        delta -= normal * float(delta @ normal)
        pose[:3, 3] += delta
        total += delta
        if float(np.linalg.norm(delta)) < 2e-5:
            break
    rmse = float(np.sqrt(np.mean(np.square(distances[keep]))))
    inliers = int(np.count_nonzero(keep))
    if float(np.linalg.norm(total)) > max_total_m or rmse > 0.006:
        return None, rmse, inliers
    return pose, rmse, inliers


def _refine_scissors_from_wrist(observed, initial, profile):
    """Recover in-plane settling without letting one-sided depth move the plane."""
    return _refine_in_plane(observed, initial, profile, cutoff_m=0.010, max_total_m=0.020, reducer=np.median)


def _refine_wrench_from_wrist(observed, initial, profile):
    """Refine only observable translation from the partial wrist views."""
    return _refine_in_plane(observed, initial, profile, cutoff_m=0.012, max_total_m=0.015, reducer=np.mean)


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


def _initial_wrench_pose(reference_cloud: dict[str, Any], feature: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    points = _points(reference_cloud)
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
    result[:3, 3] = ring - rotation @ np.asarray(profile["feature_local"], dtype=np.float64)
    return result


def _register_wrench_cad(observed: np.ndarray, initial: np.ndarray, profile: dict[str, Any]):
    """Dense trimmed rigid ICP, used on the complete pre-grasp observation."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

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


def _anchor_wrench_plane(pose: np.ndarray, coarse: np.ndarray, reference_points: np.ndarray,
                         profile: dict[str, Any]) -> np.ndarray:
    """Keep ICP's in-plane fit but recover plane normal/thickness from RGB-D.

    A flat object gives weak point-to-point constraints along its thickness,
    which allowed a few degrees of pitch to move a distant ring by 5--10 mm.
    The support plane supplies the normal; the top-depth mode supplies the CAD
    mid-plane coordinate.
    """
    old_normal = pose[:3, 1].copy()
    new_normal = np.array([0.0, 0.0, 1.0 if coarse[2, 1] >= 0.0 else -1.0])
    old_normal /= max(float(np.linalg.norm(old_normal)), 1e-12)
    if float(old_normal @ new_normal) < 0.0:
        new_normal = -new_normal
    correction, _ = Rotation.align_vectors([new_normal], [old_normal])
    pose = pose.copy()
    pose[:3, :3] = correction.as_matrix() @ pose[:3, :3]
    points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    # Ring holes contribute table pixels; the upper world-Z mode is the top
    # face and is cleanly separated by roughly the object's thickness.
    tool_face = points[points[:, 2] > float(np.quantile(points[:, 2], 0.60))]
    normal = pose[:3, 1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    upward = float(normal[2]) >= 0.0
    surface = float(np.median(tool_face @ normal))
    model = _model_points(str(profile["model_path"]), float(profile.get("model_scale", 1.0)))
    normal_index = int(profile.get("normal_axis_index", 1))
    support = float(np.max(model[:, normal_index]) if upward else np.min(model[:, normal_index]))
    pose[:3, 3] += normal * ((surface - support) - float(pose[:3, 3] @ normal))
    return pose


def _feature_from_wrench_pose(wrench: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    x_axis = wrench[:3, 0]
    z_axis = wrench[:3, :3] @ np.asarray(profile["normal_local"], dtype=np.float64)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1.0e-12)
    x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-12)
    y_axis = np.cross(z_axis, x_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = wrench[:3, :3] @ np.asarray(profile["feature_local"], dtype=np.float64) + wrench[:3, 3]
    return result


def _recover_loop_from_overview(ctx, cameras, predicted, description, profile):
    """Recover a held loop when the close wrist view is self-occluded.

    Overview detections are associated geometrically with the rigid prediction
    rather than by detection index, which stays valid when another object has
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
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=description,
                          max_results=int(profile.get("overview_max_results", 3)))
        for mask, raw_score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
            score = float(raw_score)
            if score < float(profile.get("overview_minimum_score", 0.05)):
                continue
            cloud = ctx.tool("geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
                             depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]
            if len(_points(cloud)) < 20:
                continue
            try:
                fit = ctx.tool("geometry.fit_planar_feature", points=cloud,
                               normal_hint={"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])},
                               fit_circle_center=True)
            except Exception:
                continue
            # A distant planar loop localises its centre well, but its normal
            # and in-plane roll are ambiguous under partial occlusion: keep the
            # rigid prior orientation and take only the associated centre.
            observed = predicted.copy()
            center = np.array([fit["pose"]["position"][key] for key in ("x", "y", "z")], dtype=np.float64)
            jump = float(np.linalg.norm(center - predicted[:3, 3]))
            if jump <= maximum_jump:
                observed[:3, 3] = center
                candidates.append((jump, -score, observed, score))
    if not candidates:
        return None, 0.0
    _, _, observed, score = min(candidates, key=lambda item: item[:2])
    return observed, score


def _wrist_clouds(ctx, cameras, wrist_name, profile, world_tcp, *, remote_fallback: bool):
    """Segment the held object from the wrist camera(s); (clouds, scores)."""
    clouds, scores = [], []
    wrists = [c for c in cameras if str(c.get("name", "")).startswith("eye_in_hand")]
    wrists.sort(key=lambda c: c.get("name") != wrist_name)
    tcp_position = world_tcp[:3, 3]
    for camera in wrists:
        if camera.get("name") != wrist_name and (clouds or not remote_fallback):
            break
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=str(profile["query"]),
                          max_results=int(profile.get("max_results", 3)))
        localized = []
        for mask, raw_score in zip(result.get("masks") or [], result.get("scores") or [], strict=False):
            score = float(raw_score)
            if score < 0.08:
                continue
            cloud = ctx.tool("geometry.mask_to_world_points", mask=np.asarray(mask, dtype=np.uint8),
                             depth=camera["depth"], intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]
            array = _points(cloud)
            if len(array) < 40:
                continue
            distance = float(np.linalg.norm(np.median(array, axis=0) - tcp_position))
            if distance <= 0.30:
                localized.append((distance, -score, array, score))
            if not remote_fallback:
                break  # the planar strategy takes the first mask only
        if localized:
            _, _, array, score = min(localized, key=lambda item: item[:2])
            clouds.append(array)
            scores.append(score)
            if camera.get("name") != wrist_name:
                print(f"[register_held] active wrist occluded; using {camera.get('name')} remote observation")
    return clouds, scores


def _first_pass_loop(ctx, prior: np.ndarray, ranked, first_cloud: dict[str, Any], profile: dict[str, Any]):
    """The best-scoring loop mask whose fitted centre lies where the grasp left it.

    On the first pass nothing but the carried pre-grasp perception says where
    the loop is, and a language mask can rank the object's OTHER end first: from
    the wrist camera a wrench's jaw opening reads as a "ring" within a few
    hundredths of score of the ring itself, and sub-millimetre differences in
    how the object sits in the jaws decide the order. A rigidly grasped loop
    cannot be farther from its pre-grasp position than the profile's
    ``maximum_translation_jump_m`` -- the bound the later passes already apply
    -- so the masks are tried best-first and the first whose fitted centre lies
    within it is the observation. Without a profile every mask qualifies, as
    before. ``first_cloud`` is the cloud already lifted for the top mask.

    Returns ``(aligned_frame, centre, cloud, score)``, or ``None`` when no
    mask of reliable score fits.
    """
    bound = float(profile.get("maximum_translation_jump_m", 0.025)) if profile else float("inf")
    for index, (score, camera, mask) in enumerate(ranked):
        if score < 0.20:
            break
        if index == 0:
            cloud = first_cloud
        else:
            cloud = ctx.tool(
                "geometry.mask_to_world_points",
                mask=np.asarray(mask, dtype=np.uint8),
                depth=camera["depth"],
                intrinsics=camera["intrinsics"],
                camera_pose=camera["pose"],
            )["points"]
            if len(_points(cloud)) < 8:
                continue
        aligned, centre = _fit_loop_center(ctx, prior, cloud)
        jump = float(np.linalg.norm(centre - prior[:3, 3]))
        if jump <= bound:
            return aligned, centre, cloud, score
        print(f"[register_held] rejected loop mask {index} (score {score:.2f}): fitted centre "
              f"{1000 * jump:.0f} mm from the pre-grasp loop, bound {1000 * bound:.0f} mm")
    return None


def run(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    reference_cloud: dict[str, Any],
    functional_feature: dict[str, Any],
    object_description: str,
    prior_feature_in_tcp: dict[str, Any] | None = None,
    prior_object_in_tcp: dict[str, Any] | None = None,
    prior_attached_object: dict[str, Any] | None = None,
    attachment_fit_type: str = "morphit",
    attachment_source: str = "reference",
    grasp_pose: dict[str, Any] | None = None,
    direction_marker_description: str = "",
    camera_name_filter: str = "eye_in_hand",
    arm_id: int | None = None,
) -> Output:
    if attachment_source not in {"reference", "observed"}:
        raise ValueError("attachment_source must be 'reference' or 'observed'")
    profile = dict(functional_feature.get("registration_profile") or {})
    strategy = str(profile.get("strategy", "semantic_feature"))
    if strategy not in {"semantic_feature", "cad_distal_loop", "planar_cad_landmark"}:
        raise ValueError(f"unsupported registration strategy {strategy!r}")
    if arm_id is None and prior_attached_object is not None and "arm_id" in prior_attached_object:
        arm_id = int(prior_attached_object["arm_id"])
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}
    # The wrist that holds the object: the profile may name it; a named arm
    # implies its own; otherwise the filter selects as it always did.
    wrist_name = str(profile.get("wrist_camera", ""))
    if not wrist_name and arm_id is not None:
        wrist_name = "eye_in_hand_left" if int(arm_id) == 0 else "eye_in_hand_right"
    ee = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    world_tcp = _matrix(ee)
    kind = functional_feature.get("kind")
    # The reference cloud and the pre-grasp feature were observed before the
    # object moved.  When the commanded grasp frame is known, carry both by the
    # rigid grasp transform so a later hand pose (after a lift) does not turn a
    # missed wrist detection into a false in-hand displacement.
    carry = np.eye(4)
    if grasp_pose is not None:
        carry = world_tcp @ np.linalg.inv(_matrix(grasp_pose))
    reference = _transform_cloud(reference_cloud, carry) if grasp_pose is not None else reference_cloud

    # ---- CAD-backed pre-passes (profile-driven) ----------------------------
    wrench_registration = None
    wrench_confidence = 0.0
    preserve_wrench_prior = False
    if strategy == "cad_distal_loop":
        clouds, scores = _wrist_clouds(ctx, cameras, wrist_name, profile, world_tcp, remote_fallback=True)
        if clouds:
            if prior_object_in_tcp is not None:
                initial = world_tcp @ _matrix(prior_object_in_tcp)
            else:
                coarse = _initial_wrench_pose(reference_cloud, _matrix(functional_feature["pose"]), profile)
                initial, ref_rmse, ref_inliers = _register_wrench_cad(_points(reference_cloud), coarse, profile)
                if initial is None:
                    initial = coarse
                    print(f"[register_held] pre-grasp CAD ICP rejected: rmse={1000 * ref_rmse:.2f}mm inliers={ref_inliers}")
                else:
                    initial = _anchor_wrench_plane(initial, coarse, _points(reference_cloud), profile)
                    print(f"[register_held] pre-grasp CAD ICP accepted: rmse={1000 * ref_rmse:.2f}mm inliers={ref_inliers}")
            wrench_registration, rmse, inliers = _refine_wrench_from_wrist(np.vstack(clouds), initial, profile)
            if wrench_registration is not None:
                wrench_confidence = max(scores) * float(np.exp(-rmse / 0.006))
                print(f"[register_held] wrist CAD refinement accepted: rmse={1000 * rmse:.2f}mm inliers={inliers} confidence={wrench_confidence:.3f}")
            elif prior_object_in_tcp is None:
                # First pass: the pre-grasp CAD estimate is still the safest
                # prior; the generic loop fitter below may still correct it.
                wrench_registration, wrench_confidence = initial, 0.25
                print(f"[register_held] wrist CAD refinement rejected; preserving pre-grasp CAD prior: rmse={1000 * rmse:.2f}mm inliers={inliers}")
            else:
                # Jaws closed: feature_in_tcp is rigid, and a partial ring arc
                # fed to a circle fit moved a distant ring by centimetres.
                # Preserve the last validated transform.
                preserve_wrench_prior = True
                print(f"[register_held] wrist CAD refinement rejected; preserving rigid TCP prior: rmse={1000 * rmse:.2f}mm inliers={inliers}")
    scissors_registration = None
    scissors_confidence = 0.0
    if strategy == "planar_cad_landmark" and functional_feature.get("local_center") is not None:
        clouds, scores = _wrist_clouds(ctx, cameras, wrist_name, profile, world_tcp, remote_fallback=False)
        if clouds:
            initial = (world_tcp @ _matrix(prior_object_in_tcp) if prior_object_in_tcp is not None
                       else _scissors_pose_from_feature(functional_feature))
            scissors_registration, rmse, inliers = _refine_scissors_from_wrist(np.vstack(clouds), initial, profile)
            if scissors_registration is not None:
                scissors_confidence = max(scores) * float(np.exp(-rmse / 0.006))
                print(f"[register_held] planar CAD refinement accepted: rmse={1000 * rmse:.2f}mm inliers={inliers} confidence={scissors_confidence:.3f}")
            else:
                print(f"[register_held] planar CAD refinement rejected: rmse={1000 * rmse:.2f}mm inliers={inliers}")

    # ---- the generic path (main's body) ------------------------------------
    best = None
    ranked: list[tuple[float, dict[str, Any], Any]] = []
    for camera in cameras:
        name = camera.get("name", "")
        if wrist_name and profile:
            # A profiled graph measures a held loop from the holding wrist only
            # and everything else from the overview camera.
            if (name != wrist_name) if kind == "loop" else (name != str(profile.get("overview_camera", "overhead"))):
                continue
        elif camera_name_filter and camera_name_filter not in name:
            continue
        if kind == "loop":
            # A loop is localized from its own description on every pass: the
            # circle fit below needs the ring, not the whole object.
            query = functional_feature.get("description") or object_description
        elif kind == "tip" or prior_feature_in_tcp is None:
            # Thin tips are unreliable language-segmentation targets. Segment
            # the complete held object and recover the distal endpoint in 3D.
            query = object_description
        else:
            query = functional_feature.get("description") or object_description
        result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=2)
        if result.get("masks") and result.get("scores"):
            score = float(result["scores"][0])
            if best is None or score > best[0]:
                best = (score, camera, result["masks"][0])
            ranked.extend(
                (float(raw), camera, mask)
                for mask, raw in zip(result["masks"], result["scores"], strict=False)
            )
    ranked.sort(key=lambda item: -item[0])
    cloud = reference
    confidence = 0.25
    observed_points = None
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
        points = _points(observed)
        if len(points) >= 8:
            accept_observed = True
            if prior_feature_in_tcp is None and kind != "loop":
                # Segmentation can merge a held tool with the arm. Reject a
                # grossly inconsistent 3D extent before it becomes an enormous
                # attached collision model and clearance waypoint.
                reference_size = float(np.linalg.norm(np.ptp(_points(reference), axis=0)))
                observed_size = float(np.linalg.norm(np.ptp(points, axis=0)))
                ratio = observed_size / max(reference_size, 1.0e-6)
                if not 0.55 <= ratio <= 1.80:
                    accept_observed = False
                    confidence = 0.25
            if accept_observed:
                cloud = observed
                observed_points = points
    reliable_registration = best is not None and best[0] >= 0.20 and not preserve_wrench_prior
    tip_frame = None
    if kind == "tip" and direction_marker_description and observed_points is not None:
        tip_frame = _observed_tip_frame(ctx, best[1], observed_points, direction_marker_description)

    overview_loop = None
    overview_confidence = 0.0
    if (
        profile
        and prior_feature_in_tcp is not None
        and kind == "loop"
        and bool(profile.get("overview_loop_fallback", True))
        and (
            preserve_wrench_prior
            or best is None
            or best[0] < 0.20
            or (wrench_registration is not None
                and wrench_confidence < float(profile.get("overview_trigger_confidence", 0.20)))
        )
    ):
        overview_loop, overview_confidence = _recover_loop_from_overview(
            ctx, cameras, world_tcp @ _matrix(prior_feature_in_tcp),
            str(functional_feature.get("description", object_description)), profile,
        )
        if overview_loop is not None:
            print(f"[register_held] wrist feature occluded; accepted associated overview loop observation confidence={overview_confidence:.3f}")

    registration_correction = np.zeros(3, dtype=np.float64)
    if overview_loop is not None:
        predicted = world_tcp @ _matrix(prior_feature_in_tcp)
        world_feature = overview_loop
        registration_correction = world_feature[:3, 3] - predicted[:3, 3]
        confidence = overview_confidence
        preserve_wrench_prior = False
    elif scissors_registration is not None:
        world_feature = _feature_from_scissors_pose(scissors_registration, functional_feature["local_center"])
        confidence = scissors_confidence
    elif wrench_registration is not None:
        world_feature = _feature_from_wrench_pose(wrench_registration, profile)
        confidence = wrench_confidence
    elif prior_feature_in_tcp is not None and reliable_registration:
        # Near a fixture, whole-object matching is easily contaminated by the
        # fixture. Localize the functional feature directly, correct its center,
        # and preserve the already-reached feature orientation in the hand.
        world_feature = world_tcp @ _matrix(prior_feature_in_tcp)
        predicted_center = world_feature[:3, 3].copy()
        if kind == "loop":
            world_feature, observed_center = _fit_loop_center(ctx, world_feature, cloud)
        elif tip_frame is not None:
            world_feature = _align_normal_preserving_roll(world_feature, tip_frame)
            observed_center = tip_frame[:3, 3]
        else:
            observed_center = np.median(_points(cloud), axis=0)
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
        world_feature = carry @ _matrix(functional_feature["pose"])
        if tip_frame is not None:
            world_feature = tip_frame
        elif reliable_registration and kind == "loop":
            accepted = _first_pass_loop(ctx, world_feature, ranked, cloud, profile)
            if accepted is None:
                # No reliable mask put the loop where the pre-grasp perception
                # left it: the semantic draw latched onto the object's other
                # end. Keep the carried pre-grasp loop at fallback confidence
                # rather than register a feature 20 cm from where it is.
                reliable_registration = False
                confidence = 0.25
                cloud = reference
            else:
                world_feature, observed_center, cloud, confidence = accepted
                world_feature[:3, 3] = observed_center
        elif reliable_registration:
            correction = np.median(_points(cloud), axis=0) - np.median(_points(reference), axis=0)
            if float(np.linalg.norm(correction)) <= 0.02:
                world_feature[:3, 3] += correction
            else:
                confidence = 0.25
    feature_tcp = _pose(np.linalg.inv(world_tcp) @ world_feature)

    preserve_prior = False
    if profile and prior_feature_in_tcp is not None:
        # A rigidly grasped feature is fixed in TCP coordinates. Wrist
        # re-observation may refine it by millimetres, but a centimetre-scale
        # or large angular jump means the mask/ICP latched onto another
        # symmetric surface. Reject it before mate planning turns the error
        # into a distant, apparently unreachable approach pose.
        prior_matrix, proposed = _matrix(prior_feature_in_tcp), _matrix(feature_tcp)
        translation_jump = float(np.linalg.norm(proposed[:3, 3] - prior_matrix[:3, 3]))
        rotation_jump = float(Rotation.from_matrix(proposed[:3, :3] @ prior_matrix[:3, :3].T).magnitude())
        if (translation_jump > float(profile.get("maximum_translation_jump_m", 0.025))
                or rotation_jump > np.deg2rad(float(profile.get("maximum_rotation_jump_deg", 20.0)))):
            print(f"[register_held] rejected non-rigid TCP feature jump: translation={1000 * translation_jump:.1f}mm "
                  f"rotation={np.rad2deg(rotation_jump):.1f}deg; preserving prior")
            feature_tcp = dict(prior_feature_in_tcp)
            preserve_prior = True
            confidence = 0.25
    for key in ("kind", "radius_inner", "radius_outer"):
        if key in functional_feature:
            feature_tcp[key] = functional_feature[key]

    # The wrist mask may isolate only the functional feature; the reference
    # cloud is the complete pre-grasp object. ``attachment_source`` chooses
    # which one bounds the collision model.
    attachment_cloud = cloud if attachment_source == "observed" else reference
    attachment = prior_attached_object
    if attachment is None:
        # MORPHIT is cuRobo's fitter and lives in the curobo bundle; the CPU
        # geometry bundle fits the surface and voxel kinds.
        fit_kind = str(profile.get("attachment_fit_type", attachment_fit_type)).strip().lower()
        # Both fitters are named literally, so a static allowlist check can read
        # this call (gap-self-learning's guard rejects a tool name that arrives
        # as an expression).
        fit_options = dict(
            points=attachment_cloud,
            tcp_pose=ee,
            surface_radius=0.002,
            margin=0.002,
            max_spheres=64,
        )
        if fit_kind == "morphit":
            attachment = ctx.tool("curobo.cloud_to_attachment", **fit_options)["attached_object"]
        else:
            attachment = ctx.tool("geometry.cloud_to_attachment", fit_type=fit_kind, **fit_options)["attached_object"]
    if preserve_prior and prior_object_in_tcp is not None:
        obj = world_tcp @ _matrix(prior_object_in_tcp)
    elif scissors_registration is not None:
        obj = scissors_registration
    elif wrench_registration is not None:
        obj = wrench_registration
    elif prior_object_in_tcp is not None:
        obj = world_tcp @ _matrix(prior_object_in_tcp)
        obj[:3, 3] += registration_correction
    else:
        obj = np.eye(4)
        obj[:3, 3] = np.median(_points(attachment_cloud), axis=0)

    if overview_loop is not None:
        method = "overview_loop_feature"
    elif wrench_registration is not None:
        method = "cad_distal_loop"
    elif scissors_registration is not None:
        method = "planar_cad_landmark"
    elif reliable_registration:
        method = "semantic_feature"
    else:
        method = "rigid_grasp_prior"
    uncertainty = float(profile.get(
        "fallback_uncertainty_m" if confidence <= 0.25 else "measurement_uncertainty_m",
        0.010 if confidence <= 0.25 else 0.004,
    ))
    if profile or arm_id is not None:
        attachment = dict(attachment)
        if arm_id is not None:
            attachment["arm_id"] = int(arm_id)
        if profile:
            attachment["translation_uncertainty_m"] = uncertainty
            attachment["registration_method"] = method
    return {
        "feature_in_tcp": feature_tcp,
        "object_in_tcp": _pose(np.linalg.inv(world_tcp) @ obj),
        "attached_object": attachment,
        "registration_confidence": float(confidence),
        "registration_method": method,
        "fallback_used": bool(preserve_prior or confidence <= 0.25),
        "translation_uncertainty_m": uncertainty,
    }
