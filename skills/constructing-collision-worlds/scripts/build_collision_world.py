"""Construct a profile-driven collision world from calibrated RGB-D views."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    world_config: dict[str, Any]
    mesh_names: list[str]
    removed_mesh_names: list[str]
    strategy: str


def _select_profile(target_kind: str,
                    collision_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = {str(item["source_kind"]): item for item in collision_profiles}
    if target_kind not in profiles:
        raise ValueError(f"no collision profile declared for {target_kind!r}")
    return profiles[target_kind]


def _semantic_mask(ctx: NodeContext, camera: dict[str, Any], query: str,
                   threshold: float, max_results: int = 2) -> np.ndarray | None:
    result = ctx.tool(
        "sam3.segment_text", image=camera["rgb"], query=query,
        max_results=max_results,
    )
    candidates = [
        (float(score), np.asarray(mask) > 0)
        for mask, score in zip(result.get("masks") or [],
                               result.get("scores") or [], strict=False)
        if float(score) >= threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1].astype(np.uint8) * 255


def _triangle_origin_distance(triangle: np.ndarray) -> float:
    a, b, c = triangle
    cross = lambda u, v: float(u[0] * v[1] - u[1] * v[0])
    signs = [cross(b-a, -a), cross(c-b, -b), cross(a-c, -c)]
    if all(value >= -1.0e-9 for value in signs) or all(
            value <= 1.0e-9 for value in signs):
        return 0.0
    distances = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        t = float(np.clip(-(start @ edge) / max(float(edge @ edge), 1.0e-12), 0.0, 1.0))
        distances.append(float(np.linalg.norm(start + t * edge)))
    return min(distances)


def _filter_meshes(config: dict[str, Any], profile: dict[str, Any]
                   ) -> tuple[dict[str, Any], list[str], list[str]]:
    rules = profile.get("obstacle_filter") or {}
    max_vertices = int(rules.get("max_sparse_vertices", 31))
    sheet_thickness = float(rules.get("min_sheet_thickness_m", 0.004))
    sparse_span = float(rules.get("max_sparse_span_m", 0.15))
    compact_span = float(rules.get("max_compact_sparse_span_m", 0.060))
    floor_z = float(rules.get("valid_workspace_floor_z_m", 0.30))
    scene_span = float(rules.get("max_scene_span_m", 2.0))
    kept, retained, removed = [], [], []
    for index, mesh in enumerate(config.get("meshes") or []):
        name = str(mesh.get("name", f"mesh_{index}"))
        raw_vertices = mesh.get("vertices")
        vertices = np.asarray(
            [] if raw_vertices is None else raw_vertices, dtype=np.float64
        ).reshape(-1, 3)
        extent = np.ptp(vertices, axis=0) if len(vertices) else np.zeros(3)
        artifact = bool(
            0 < len(vertices) <= max_vertices
            and ((float(extent.min()) < sheet_thickness
                  and float(extent.max()) < sparse_span)
                 or float(extent.max()) < compact_span)
        )
        invalid = bool(len(vertices) and (
            float(vertices[:, 2].max()) < floor_z or float(extent.max()) > scene_span
        ))
        if artifact or invalid:
            removed.append(name)
        else:
            retained.append(name)
            kept.append(mesh)
    result = dict(config)
    result["meshes"] = kept
    return result, retained, removed


def _carve_corridor(config: dict[str, Any], feature_point: dict[str, float],
                    fixture_axis: dict[str, float], profile: dict[str, Any]
                    ) -> dict[str, Any]:
    corridor = profile.get("approach_corridor") or {}
    if not corridor.get("enabled", False):
        return config
    origin = np.array([feature_point[key] for key in ("x", "y", "z")], dtype=np.float64)
    axis = float(corridor.get("axis_sign", 1.0)) * np.array(
        [fixture_axis[key] for key in ("x", "y", "z")], dtype=np.float64,
    )
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    axial_min = float(corridor.get("axial_min_m", -0.012))
    axial_max = float(corridor.get("axial_max_m", 0.25))
    radius = float(corridor.get("radius_m", 0.055))
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1.0e-12)
    basis_v = np.cross(axis, basis_u)
    carved = []
    for mesh in config.get("meshes") or []:
        raw_vertices, raw_faces = mesh.get("vertices"), mesh.get("faces")
        vertices = np.asarray(
            [] if raw_vertices is None else raw_vertices, dtype=np.float64
        ).reshape(-1, 3)
        faces = np.asarray(
            [] if raw_faces is None else raw_faces, dtype=np.int64
        ).reshape(-1, 3)
        if not len(vertices) or not len(faces):
            carved.append(mesh)
            continue
        relative = vertices - origin
        axial = relative @ axis
        projected = np.column_stack((relative @ basis_u, relative @ basis_v))
        keep = []
        for face in faces:
            values = axial[face]
            overlaps = float(values.max()) >= axial_min and float(values.min()) <= axial_max
            intersects = overlaps and _triangle_origin_distance(projected[face]) <= radius
            keep.append(not intersects)
        faces = faces[np.asarray(keep, dtype=bool)]
        if not len(faces):
            continue
        used = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        updated = dict(mesh)
        updated["vertices"] = vertices[used].astype(np.float32)
        updated["faces"] = remap[faces].astype(np.int32)
        carved.append(updated)
    result = dict(config)
    result["meshes"] = carved
    return result


def _append_keep_out(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Append declared axis-aligned keep-out boxes as planner mesh geometry."""
    boxes = list((profile.get("fixture_keep_out") or {}).get("boxes") or [])
    if not boxes:
        return config
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)
    meshes = list(config.get("meshes") or [])
    allowed = set(profile.get("allowed_contact_surfaces") or [])
    for index, box in enumerate(boxes):
        name = str(box.get("name", f"fixture_keep_out_{index}"))
        if name in allowed:
            continue
        center = np.array([float((box["center"])[key]) for key in ("x", "y", "z")])
        half = 0.5 * np.array([float((box["size"])[key]) for key in ("x", "y", "z")])
        meshes.append({"name": name, "vertices": (center + signs * half).astype(np.float32),
                       "faces": faces.copy()})
    result = dict(config)
    result["meshes"] = meshes
    return result


def run(ctx: NodeContext, observation: dict[str, Any], target_mask: Any,
        fixture_mask: Any = None, hook_tip: dict[str, float] | None = None,
        fixture_axis: dict[str, float] | None = None, target_kind: str = "object",
        collision_profiles: list[dict[str, Any]] | None = None,
        robot_spheres: list[dict[str, Any]] | None = None) -> Output:
    profile = _select_profile(target_kind, collision_profiles or [])
    strategy = str(profile.get("strategy", "rgbd_mesh"))
    if strategy == "disabled":
        return {"world_config": {"meshes": []}, "mesh_names": [],
                "removed_mesh_names": [], "strategy": strategy}
    if strategy != "rgbd_mesh":
        raise ValueError(f"unsupported collision-world strategy {strategy!r}")
    cameras = list(observation.get("cameras") or [])
    if not cameras:
        raise ValueError("collision reconstruction requires RGB-D cameras")

    exclusions = profile.get("excluded_masks") or {}
    object_masks: list[dict[str, Any]] = []
    target_query = str(exclusions.get("target_query", target_kind))
    for index, camera in enumerate(cameras):
        is_reference = camera.get("name") == str(profile.get("reference_camera", "overhead"))
        if is_reference and exclusions.get("target", True):
            object_masks.append({"name": "manipulated_object", "mask": target_mask,
                                 "camera_index": index})
        elif exclusions.get("cross_view_target", True):
            mask = _semantic_mask(
                ctx, camera, target_query,
                float(exclusions.get("target_score_min", 0.08)),
            )
            if mask is not None:
                object_masks.append({"name": f"manipulated_object_{index}",
                                     "mask": mask, "camera_index": index})
        allowed_surfaces = set(profile.get("allowed_contact_surfaces") or [])
        if (is_reference and fixture_mask is not None
                and "fixture_mask" in allowed_surfaces):
            mask = np.asarray(fixture_mask, dtype=np.uint8)
            if mask.shape == camera["depth"].shape and float(np.mean(mask > 0)) < 0.5:
                object_masks.append({"name": "allowed_contact_fixture", "mask": mask,
                                     "camera_index": index})
        if exclusions.get("robot", True):
            merged = None
            for query in exclusions.get("robot_queries", ["robot arm", "robot gripper"]):
                mask = _semantic_mask(
                    ctx, camera, str(query),
                    float(exclusions.get("robot_score_min", 0.10)),
                )
                if mask is not None:
                    candidate = mask > 0
                    merged = candidate if merged is None else merged | candidate
            if merged is not None:
                object_masks.append({"name": f"robot_{index}",
                                     "mask": merged.astype(np.uint8) * 255,
                                     "camera_index": index})

    robot_spheres = list(robot_spheres or [])
    fixture_keep_out = profile.get("fixture_keep_out") or {}
    reconstruction = profile.get("reconstruction") or {}
    response = ctx.tool(
        "geometry.build_world_config", cameras=cameras, object_masks=object_masks,
        voxel_size=float(reconstruction.get("voxel_size_m", 0.008)),
        noise_eps=float(reconstruction.get("noise_eps_m", 0.025)),
        noise_min_samples=int(reconstruction.get("noise_min_samples", 4)),
        mesh_alpha=float(reconstruction.get("mesh_alpha_m", 0.04)),
        robot_spheres=robot_spheres,
        robot_sphere_margin=float(fixture_keep_out.get("robot_sphere_margin_m", 0.015)),
    )
    config, retained, removed = _filter_meshes(response.get("config") or {"meshes": []}, profile)
    config = _append_keep_out(config, profile)
    if hook_tip is not None and fixture_axis is not None:
        config = _carve_corridor(config, hook_tip, fixture_axis, profile)
    if not config.get("meshes"):
        raise ValueError("RGB-D collision reconstruction produced no scene mesh")
    return {"world_config": config, "mesh_names": retained,
            "removed_mesh_names": removed, "strategy": strategy}
