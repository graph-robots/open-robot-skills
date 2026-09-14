"""Reconstruct a camera-only planner world that keeps the goal reachable.

Every selected RGB-D view contributes depth. The grasp target is excluded by
its perceived mask in the mask camera and by text segmentation in the other
views (so its returns do not become a static obstacle where the attached
object later rotates); the robot is excluded by text segmentation per view and
by its model-backed collision spheres when the connector provides them; a
perceived fixture mask can be excluded from the mask camera too.

Two intentional-contact zones can be carved out of the reconstruction so the
alpha-shape meshes do not bridge the free space the goal needs: a narrow
approach tube along a fixture axis ending at its tip, and a vertical corridor
through an aperture in a lid. Nearly zero-thickness depth-edge slivers and
unmistakable invalid-depth sheets are dropped because a collision checker
treats each one as exact occupied volume.

**The collision profile.** A graph that carries one profile per object kind
passes ``collision_profiles`` (keyed by ``source_kind``) and ``target_kind``.
A profile may switch the whole thing off (``strategy: disabled`` -- an empty
world, for an executor that has declared it will not consult one), name the
reference camera and the exclusion queries and thresholds, override the
reconstruction and artifact-filter constants, append declared keep-out boxes,
and drive the approach-tube carve from its own ``approach_corridor`` block.
Every profile key is optional and an absent key leaves the constant below in
force, so a graph that passes no profile gets exactly the world and the calls
it always got.

**The TSDF world (opt-in).** ``tsdf=True`` (or a profile's ``strategy:
rgbd_tsdf``) asks the connector's ``motion.build_world_tsdf`` for an ESDF voxel
grid integrated on device instead of the alpha-shape meshes, and returns
``{"use_tsdf": true, ...}``; the grid stays on the motion backend and a planner
reaches it through that flag. The aperture corridor is cleared on the grid as a
vertical cylinder. Measured on the ``sharps_disposal`` benchmark (graph
``gap_perception_v2``): the CPU mesh path costs 42-44 s per episode on the
overhead camera; the TSDF path is GPU kernels end to end. Whenever the tool is
absent or fails -- or the request is one a flat mask list and vertical
cylinders cannot express -- the mesh world below is built instead, and the
fallback is printed: a silent one there spent a debugging session reading
mesh-path timings as TSDF ones. Off by default, so no call changes for a graph
that does not ask.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    world_config: dict[str, Any]
    mesh_names: list[str]
    removed_mesh_names: list[str]
    strategy: str


def _semantic_mask(
    ctx: NodeContext, camera: dict[str, Any], query: str, threshold: float, max_results: int = 2
) -> np.ndarray | None:
    """The best-scoring mask for *query* at or above *threshold*, or ``None``."""
    result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query, max_results=max_results)
    candidates = [
        (float(score), np.asarray(mask) > 0)
        for mask, score in zip(result.get("masks") or [], result.get("scores") or [], strict=False)
        if float(score) >= threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1].astype(np.uint8) * 255


def _visible_robot_mask(
    ctx: NodeContext, camera: dict[str, Any], queries: list[str], threshold: float
) -> np.ndarray | None:
    """Segment robot pixels so the depth reconstruction is not self-obstacle."""
    merged = None
    for query in queries:
        mask = _semantic_mask(ctx, camera, str(query), threshold)
        if mask is not None:
            candidate = mask > 0
            merged = candidate if merged is None else merged | candidate
    return None if merged is None else merged.astype(np.uint8) * 255


def _visible_target_mask(
    ctx: NodeContext, camera: dict[str, Any], description: str, threshold: float
) -> np.ndarray | None:
    """Exclude the future attachment from every reconstructed camera view."""
    result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=description, max_results=2)
    masks, scores = result.get("masks") or [], result.get("scores") or []
    if not masks or not scores or float(scores[0]) < threshold:
        return None
    return (np.asarray(masks[0]) > 0).astype(np.uint8) * 255


def _origin_triangle_distance_2d(triangle: np.ndarray) -> float:
    """Distance from the origin to a projected 2D triangle."""
    a, b, c = triangle

    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    signs = [cross(b - a, -a), cross(c - b, -b), cross(a - c, -c)]
    if all(value >= -1.0e-9 for value in signs) or all(value <= 1.0e-9 for value in signs):
        return 0.0
    origin = np.zeros(2)
    return min(
        _distance_to_segment_xy(origin, a, b),
        _distance_to_segment_xy(origin, b, c),
        _distance_to_segment_xy(origin, c, a),
    )


def _distance_to_segment_xy(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    t = float(np.clip((point - start) @ edge / max(float(edge @ edge), 1.0e-12), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * edge)))


def _point_in_triangle_xy(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Whether point lies in a projected triangle, including its boundary."""
    a, b, c = triangle[:, :2]
    v0, v1, v2 = c - a, b - a, point - a
    dot00, dot01, dot02 = v0 @ v0, v0 @ v1, v0 @ v2
    dot11, dot12 = v1 @ v1, v1 @ v2
    denominator = float(dot00 * dot11 - dot01 * dot01)
    if abs(denominator) < 1.0e-12:
        return False
    u = float((dot11 * dot02 - dot01 * dot12) / denominator)
    v = float((dot00 * dot12 - dot01 * dot02) / denominator)
    return u >= -1.0e-9 and v >= -1.0e-9 and u + v <= 1.0 + 1.0e-9


def _mesh_arrays(mesh: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_vertices, raw_faces = mesh.get("vertices"), mesh.get("faces")
    vertices = np.asarray([] if raw_vertices is None else raw_vertices, dtype=np.float64)
    faces = np.asarray([] if raw_faces is None else raw_faces, dtype=np.int64)
    return vertices.reshape(-1, 3), faces.reshape(-1, 3)


def _keep_faces(
    mesh: dict[str, Any], vertices: np.ndarray, faces: np.ndarray, keep: list[bool]
) -> dict[str, Any] | None:
    kept_faces = faces[np.asarray(keep, dtype=bool)]
    if not len(kept_faces):
        return None
    used = np.unique(kept_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    updated = dict(mesh)
    updated["vertices"] = vertices[used].astype(np.float32)
    updated["faces"] = remap[kept_faces].astype(np.int32)
    return updated


def _carve_approach_tube(
    meshes: list[dict[str, Any]],
    tip: np.ndarray,
    outward: np.ndarray,
    *,
    axial_min: float = -0.012,
    axial_max: float = 0.25,
    radius: float = 0.055,
) -> list[dict[str, Any]]:
    """Remove faces inside a narrow tube along ``outward`` ending at ``tip``.

    The 55 mm default admits the gripper collision envelope at the intentional
    mate while retaining nearby fixture geometry.
    """
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ outward)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(outward, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(outward, basis_u)
    carved = []
    for mesh in meshes:
        vertices, faces = _mesh_arrays(mesh)
        if not len(vertices) or not len(faces):
            carved.append(mesh)
            continue
        relative = vertices - tip
        axial = relative @ outward
        projected = np.column_stack((relative @ basis_u, relative @ basis_v))
        keep = []
        for face in faces:
            face_axial = axial[face]
            overlaps = float(face_axial.max()) >= axial_min and float(face_axial.min()) <= axial_max
            intersects = overlaps and _origin_triangle_distance_2d(projected[face]) <= radius
            keep.append(not intersects)
        updated = _keep_faces(mesh, vertices, faces, keep)
        if updated is not None:
            carved.append(updated)
    return carved


def _carve_corridor(
    meshes: list[dict[str, Any]], center: np.ndarray, radius: float, rim_z: float
) -> list[dict[str, Any]]:
    """Remove faces crossing a vertical corridor of ``radius`` through ``center``."""
    carved = []
    for mesh in meshes:
        vertices, faces = _mesh_arrays(mesh)
        if not len(vertices) or not len(faces):
            carved.append(mesh)
            continue
        keep = []
        for face in faces:
            triangle = vertices[face]
            z_overlap = (
                float(triangle[:, 2].max()) >= rim_z - 0.035
                and float(triangle[:, 2].min()) <= rim_z + 0.18
            )
            distances = [float(np.linalg.norm(vertex[:2] - center)) for vertex in triangle]
            for start, end in (
                (triangle[0, :2], triangle[1, :2]),
                (triangle[1, :2], triangle[2, :2]),
                (triangle[2, :2], triangle[0, :2]),
            ):
                distances.append(_distance_to_segment_xy(center, start, end))
            crosses = min(distances) <= radius or _point_in_triangle_xy(center, triangle)
            keep.append(not (z_overlap and crosses))
        updated = _keep_faces(mesh, vertices, faces, keep)
        if updated is not None:
            carved.append(updated)
    return carved


def _keep_out_boxes(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Declared axis-aligned keep-out boxes as planner mesh geometry."""
    boxes = list((profile.get("fixture_keep_out") or {}).get("boxes") or [])
    if not boxes:
        return []
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)
    allowed = set(profile.get("allowed_contact_surfaces") or [])
    out = []
    for index, box in enumerate(boxes):
        name = str(box.get("name", f"fixture_keep_out_{index}"))
        if name in allowed:
            continue
        center = np.array([float(box["center"][key]) for key in ("x", "y", "z")])
        half = 0.5 * np.array([float(box["size"][key]) for key in ("x", "y", "z")])
        out.append({"name": name, "vertices": (center + signs * half).astype(np.float32), "faces": faces.copy()})
    return out


def _pack(meshes: list[dict[str, Any]]) -> dict[str, Any]:
    """One indexed mesh from many components, without bridging triangles."""
    all_vertices, all_faces, offset = [], [], 0
    for mesh in meshes:
        vertices = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(mesh["faces"], dtype=np.int32).reshape(-1, 3)
        all_vertices.append(vertices)
        all_faces.append(faces + offset)
        offset += len(vertices)
    return {
        "name": "perceived_scene",
        "vertices": np.concatenate(all_vertices, axis=0),
        "faces": np.concatenate(all_faces, axis=0),
        "pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([value[k] for k in ("x", "y", "z")], dtype=np.float64)


#: TSDF defaults, the values the ``sharps_disposal/gap_perception_v2`` graph
#: ran in sweep s8. Each one is a measured failure, recorded where it is used.
_TSDF_VOXEL_SIZE_M = 0.01
_TSDF_CLEAR_RADIUS_FACTOR = 1.1
_TSDF_CLEAR_RADIUS_MIN_M = 0.05
_TSDF_CLEAR_BELOW_RIM_M = 0.20
_TSDF_CLEAR_ABOVE_RIM_M = 0.25


def _tsdf_world(
    ctx: NodeContext,
    cameras: list[dict[str, Any]],
    object_masks: list[dict[str, Any]],
    keep_out: list[dict[str, Any]],
    corridor_center: dict[str, float] | None,
    corridor_radius: float,
    corridor_rim_z: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """The backend's TSDF grid with the aperture corridor cleared; raises to fall back."""
    if len(cameras) != 1:
        # The tool zeroes every mask in every depth image of the same shape --
        # it has no camera index -- so a mask from one view would erase free
        # depth in another. One camera, as the graph ran it, or the mesh path.
        raise ValueError(f"TSDF mode takes exactly one camera, {len(cameras)} selected")
    clear = []
    if corridor_center is not None and float(corridor_radius) > 0.0:
        clear.append({
            "x": float(corridor_center["x"]), "y": float(corridor_center["y"]),
            # The corridor is for the *descent*, not the hole: the held object
            # hangs below the TCP and the jaws straddle the rim, so the cleared
            # cylinder spans the hand's approach (a 50 mm floor) rather than
            # the aperture alone.
            "radius": max(
                float(settings.get("clear_radius_min_m", _TSDF_CLEAR_RADIUS_MIN_M)),
                float(settings.get("clear_radius_factor", _TSDF_CLEAR_RADIUS_FACTOR)) * float(corridor_radius),
            ),
            # ...and down *through* the cavity, not merely to the rim. A mesh
            # world leaves unseen interior space free; a TSDF assigns unseen
            # voxels small distances, so a held syringe's tip 75 mm below the
            # TCP at the approach waypoint, already inside the box's mouth,
            # read as colliding and the planner refused a waypoint the mesh
            # world had passed. The top reaches well above the rim, where the
            # approach waypoint itself sits.
            "z_lo": float(corridor_rim_z) - float(settings.get("clear_below_rim_m", _TSDF_CLEAR_BELOW_RIM_M)),
            "z_hi": float(corridor_rim_z) + float(settings.get("clear_above_rim_m", _TSDF_CLEAR_ABOVE_RIM_M)),
        })
    summary = ctx.tool(
        "motion.build_world_tsdf",
        cameras=cameras,
        exclude_masks=[entry["mask"] for entry in object_masks],
        # 10 mm: an ESDF fattens every surface by about a voxel, and at 20 mm
        # that sealed the aperture to a single cleared column -- the planner
        # then refused the approach waypoint over the lid on a layout the mesh
        # world had passed. The backend's default grid extent is kept: a
        # taller grid left the carry detour statistically unchanged.
        voxel_size=float(settings.get("voxel_size_m", _TSDF_VOXEL_SIZE_M)),
        clear_cylinders=clear,
    )
    # Declared keep-out boxes ride beside the grid as meshes; the planner
    # loads both into one scene.
    return {"use_tsdf": True, "meshes": keep_out, "tsdf": summary}


def _select_profile(target_kind: str, collision_profiles: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not collision_profiles:
        return {}
    profiles = {str(item["source_kind"]): item for item in collision_profiles}
    if target_kind not in profiles:
        raise ValueError(f"no collision profile declared for {target_kind!r}")
    return dict(profiles[target_kind])


def run(
    ctx: NodeContext,
    observation: dict[str, Any],
    target_mask: Any,
    target_description: str = "",
    fixture_mask: Any = None,
    fixture_tip: dict[str, float] | None = None,
    fixture_axis: dict[str, float] | None = None,
    outward_sign: float = 1.0,
    corridor_center: dict[str, float] | None = None,
    corridor_radius: float = 0.0,
    corridor_rim_z: float = 0.0,
    camera_names: str = "",
    mask_camera_name: str = "overhead",
    voxel_size: float = 0.008,
    pack_meshes: bool = False,
    sheet_floor_z: float = 0.30,
    sheet_max_extent_m: float = 2.0,
    target_kind: str = "",
    collision_profiles: list[dict[str, Any]] | None = None,
    robot_spheres: list[dict[str, Any]] | None = None,
    tsdf: bool = False,
    tsdf_voxel_size: float = _TSDF_VOXEL_SIZE_M,
    tsdf_clear_radius_factor: float = _TSDF_CLEAR_RADIUS_FACTOR,
    tsdf_clear_radius_min_m: float = _TSDF_CLEAR_RADIUS_MIN_M,
    tsdf_clear_below_rim_m: float = _TSDF_CLEAR_BELOW_RIM_M,
    tsdf_clear_above_rim_m: float = _TSDF_CLEAR_ABOVE_RIM_M,
) -> Output:
    """Build the world from the selected views and carve the goal's free space.

    ``camera_names`` is a comma-separated selection (empty = every camera).
    ``target_mask`` and ``fixture_mask`` apply to ``mask_camera_name``; the
    other selected views exclude the target by segmenting
    ``target_description`` when it is non-empty. The approach tube is carved
    when both ``fixture_tip`` and ``fixture_axis`` are given, along
    ``outward_sign * fixture_axis`` (use ``-1`` for an axis that points into
    the fixture). The corridor is carved when ``corridor_center`` is given
    with a positive ``corridor_radius``, between ``corridor_rim_z - 35 mm``
    and ``+180 mm``. ``sheet_floor_z``/``sheet_max_extent_m`` drop components
    lying entirely below or spanning more than the workcell (0 disables).

    ``collision_profiles`` + ``target_kind`` select a profile (see the module
    docstring); ``robot_spheres`` supplies the robot's collision spheres when
    the caller already has them, instead of asking the planner.

    ``tsdf`` asks ``motion.build_world_tsdf`` for the world instead (see the
    module docstring) and returns ``strategy: rgbd_tsdf`` when it was built.
    The corridor becomes one cleared cylinder of radius
    ``max(tsdf_clear_radius_min_m, tsdf_clear_radius_factor * corridor_radius)``
    from ``corridor_rim_z - tsdf_clear_below_rim_m`` to ``+ tsdf_clear_above_rim_m``.
    It falls back to the mesh world, printing why, when the tool raises or is
    absent, when more than one camera is selected, or when an approach tube is
    requested (a tube along an arbitrary axis is not a vertical cylinder). A
    profile's ``tsdf`` block overrides the tunables by ``voxel_size_m``,
    ``clear_radius_factor``, ``clear_radius_min_m``, ``clear_below_rim_m`` and
    ``clear_above_rim_m``.
    """
    profile = _select_profile(target_kind, collision_profiles)
    strategy = str(profile.get("strategy", "rgbd_mesh"))
    if strategy == "disabled":
        return {"world_config": {"meshes": []}, "mesh_names": [], "removed_mesh_names": [], "strategy": strategy}
    if strategy == "rgbd_tsdf":
        tsdf = True
    elif strategy != "rgbd_mesh":
        raise ValueError(f"unsupported collision-world strategy {strategy!r}")
    exclusions = profile.get("excluded_masks") or {}
    reconstruction = profile.get("reconstruction") or {}
    filters = profile.get("obstacle_filter") or {}
    keep_out = profile.get("fixture_keep_out") or {}
    corridor = profile.get("approach_corridor") or {}
    allowed_surfaces = set(profile.get("allowed_contact_surfaces") or [])
    if "reference_camera" in profile:
        mask_camera_name = str(profile["reference_camera"])
    if not target_description and exclusions.get("cross_view_target", False) and target_kind:
        target_description = str(exclusions.get("target_query", target_kind))

    all_cameras = list(observation.get("cameras") or [])
    if isinstance(observation.get("cameras"), dict):
        all_cameras = list(observation["cameras"].values())
    selected = [name.strip() for name in str(camera_names).split(",") if name.strip()]
    cameras = [c for c in all_cameras if not selected or c.get("name") in selected]
    if not cameras:
        raise ValueError("collision reconstruction requires RGB-D cameras")
    object_masks: list[dict[str, Any]] = []
    for index, camera in enumerate(cameras):
        if camera.get("name") == mask_camera_name:
            if exclusions.get("target", True):
                object_masks.append({"name": "grasp_target", "mask": target_mask, "camera_index": index})
            # A fixture mask is excluded as before; a profile may also say the
            # fixture is an allowed contact surface, which means the same thing.
            if fixture_mask is not None and (not profile or "fixture_mask" in allowed_surfaces):
                contact_mask = np.asarray(fixture_mask, dtype=np.uint8)
                # Some geometric fixture detectors return a full-frame mask as
                # a placeholder. Excluding that would erase the entire world.
                if contact_mask.shape == camera["depth"].shape and float(np.mean(contact_mask > 0)) < 0.5:
                    object_masks.append({"name": "contact_fixture", "mask": contact_mask, "camera_index": index})
        elif target_description and exclusions.get("cross_view_target", True):
            # Reconstruction happens before grasping. If the target is
            # removed only from one view, its returns in the others become a
            # static obstacle exactly where the attached object later rotates.
            target_view = _visible_target_mask(
                ctx, camera, target_description, float(exclusions.get("target_score_min", 0.08))
            )
            if target_view is not None:
                object_masks.append({"name": f"grasp_target_view_{index}", "mask": target_view, "camera_index": index})
        if exclusions.get("robot", True):
            robot_mask = _visible_robot_mask(
                ctx, camera,
                list(exclusions.get("robot_queries") or ["robot arm", "robot gripper"]),
                float(exclusions.get("robot_score_min", 0.10)),
            )
            if robot_mask is not None:
                object_masks.append({"name": f"robot_view_{index}", "mask": robot_mask, "camera_index": index})
    tube_requested = fixture_tip is not None and fixture_axis is not None and corridor.get("enabled", True)
    if tsdf:
        settings = {
            "voxel_size_m": tsdf_voxel_size,
            "clear_radius_factor": tsdf_clear_radius_factor,
            "clear_radius_min_m": tsdf_clear_radius_min_m,
            "clear_below_rim_m": tsdf_clear_below_rim_m,
            "clear_above_rim_m": tsdf_clear_above_rim_m,
            **(profile.get("tsdf") or {}),
        }
        try:
            if tube_requested:
                raise ValueError("an approach tube along the fixture axis is not a vertical clear cylinder")
            keep_out_meshes = _keep_out_boxes(profile)
            world = _tsdf_world(
                ctx, cameras, object_masks, keep_out_meshes,
                corridor_center, corridor_radius, corridor_rim_z, settings,
            )
            return {
                "world_config": world,
                "mesh_names": [str(mesh["name"]) for mesh in keep_out_meshes],
                "removed_mesh_names": [],
                "strategy": "rgbd_tsdf",
            }
        except Exception as error:  # noqa: BLE001 -- every failure falls back, loudly
            # Tool absent (a backend with no mapper) or integration failed:
            # the mesh reconstruction below is the world the planner always had.
            print(f"[build_collision_world] TSDF path unavailable ({error}); falling back to mesh reconstruction",
                  flush=True)
            strategy = "rgbd_mesh"
    if robot_spheres is None:
        # Capture robot geometry with the same observation. Unlike a
        # segmentation mask, this includes occluded and visually ambiguous
        # links and remains aligned with the arm pixels even after the robot
        # subsequently moves.
        try:
            sphere_result = ctx.tool("motion.get_robot_collision_spheres", arm_id=-1)
            robot_spheres = list(sphere_result.get("spheres") or [])
        except Exception:
            # Portable to robots without a model-backed planner; their visual
            # masks still provide the established fallback.
            robot_spheres = []
    response = ctx.tool(
        "geometry.build_world_config",
        cameras=cameras,
        object_masks=object_masks,
        voxel_size=float(reconstruction.get("voxel_size_m", voxel_size)),
        noise_eps=float(reconstruction.get("noise_eps_m", 0.025)),
        noise_min_samples=int(reconstruction.get("noise_min_samples", 4)),
        mesh_alpha=float(reconstruction.get("mesh_alpha_m", 0.04)),
        robot_spheres=list(robot_spheres),
        robot_sphere_margin=float(keep_out.get("robot_sphere_margin_m", 0.015)),
    )
    config = response.get("config") or {"meshes": []}
    # Alpha-shape reconstruction can emit isolated, nearly zero-thickness
    # sheets at depth discontinuities. A collision checker treats each such
    # sheet as an exact obstacle; a 2 mm sliver detached from an otherwise
    # represented surface can invalidate a whole roadmap even though it is
    # not a closed occupied volume. Keep real thin objects and compact
    # clutter; reject only small, sparse sheet fragments.
    max_vertices = int(filters.get("max_sparse_vertices", 31))
    sheet_thickness = float(filters.get("min_sheet_thickness_m", 0.004))
    sparse_span = float(filters.get("max_sparse_span_m", 0.15))
    compact_span = float(filters.get("max_compact_sparse_span_m", 0.0))
    floor_z = float(filters.get("valid_workspace_floor_z_m", sheet_floor_z))
    scene_span = float(filters.get("max_scene_span_m", sheet_max_extent_m))
    cleaned, removed = [], []
    for index, mesh in enumerate(config.get("meshes") or []):
        vertices, _ = _mesh_arrays(mesh)
        extent = np.ptp(vertices, axis=0) if len(vertices) else np.zeros(3)
        depth_edge_sliver = bool(
            0 < len(vertices) <= max_vertices
            and ((float(extent.min()) < sheet_thickness and float(extent.max()) < sparse_span)
                 or (compact_span > 0.0 and float(extent.max()) < compact_span))
        )
        # Invalid/far depth from an oblique or wrist camera can back-project
        # into a giant sheet near z=0. It is not part of the workcell, but a
        # collision checker treats it as a real wall or floor.
        invalid_depth_sheet = bool(
            len(vertices)
            and ((floor_z > 0.0 and float(vertices[:, 2].max()) < floor_z)
                 or (scene_span > 0.0 and float(extent.max()) > scene_span))
        )
        if depth_edge_sliver or invalid_depth_sheet:
            removed.append(str(mesh.get("name", f"mesh_{index}")))
        else:
            cleaned.append(mesh)
    meshes = cleaned + _keep_out_boxes(profile)
    if tube_requested:
        # The fixture contact zone is intentionally occupied at the goal, and
        # RGB-D alpha shapes can also bridge the thin free space around a
        # shaft. Carve only a narrow perceived approach tube ending at the
        # feature; the rest of the fixture remains an obstacle.
        outward = float(corridor.get("axis_sign", outward_sign)) * _vec(fixture_axis)
        outward /= max(float(np.linalg.norm(outward)), 1.0e-12)
        meshes = _carve_approach_tube(
            meshes, _vec(fixture_tip), outward,
            axial_min=float(corridor.get("axial_min_m", -0.012)),
            axial_max=float(corridor.get("axial_max_m", 0.25)),
            radius=float(corridor.get("radius_m", 0.055)),
        )
    if corridor_center is not None and float(corridor_radius) > 0.0:
        # Alpha-shape reconstruction may bridge a real lid opening. Preserve
        # a vertical corridor of the caller's measured radius.
        center = np.array([corridor_center["x"], corridor_center["y"]], dtype=np.float64)
        meshes = _carve_corridor(meshes, center, float(corridor_radius), float(corridor_rim_z))
    if not meshes:
        raise ValueError("RGB-D collision reconstruction produced no scene mesh")
    if pack_meshes:
        # A planner's scene cache may admit only a few dozen meshes. RGB-D
        # clustering can legitimately return more disconnected components;
        # collision checking does not need each as a separate named obstacle.
        meshes = [_pack(meshes)]
    config["meshes"] = meshes
    mesh_names = [str(mesh["name"]) for mesh in meshes if mesh.get("name")]
    if not mesh_names:
        mesh_names = list(response.get("mesh_names") or [])
    return {"world_config": config, "mesh_names": mesh_names, "removed_mesh_names": removed, "strategy": strategy}
