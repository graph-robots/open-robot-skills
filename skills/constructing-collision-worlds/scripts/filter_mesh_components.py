"""Conservatively remove unmistakable RGB-D reconstruction artifacts."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class Output(TypedDict):
    world_config: dict[str, Any]
    retained_names: list[str]
    removed_names: list[str]


def run(ctx, world_config: dict[str, Any], max_sparse_vertices: int = 31,
        min_sheet_thickness_m: float = 0.004,
        max_sparse_span_m: float = 0.15,
        max_compact_sparse_span_m: float = 0.060,
        valid_workspace_floor_z_m: float = 0.30,
        max_scene_span_m: float = 2.0) -> Output:
    del ctx
    kept, retained, removed = [], [], []
    for index, mesh in enumerate(world_config.get("meshes") or []):
        name = str(mesh.get("name", f"mesh_{index}"))
        raw_vertices = mesh.get("vertices")
        vertices = np.asarray(
            [] if raw_vertices is None else raw_vertices, dtype=np.float64
        ).reshape(-1, 3)
        extent = np.ptp(vertices, axis=0) if len(vertices) else np.zeros(3)
        depth_edge_sheet = bool(
            0 < len(vertices) <= int(max_sparse_vertices)
            and float(extent.min()) < float(min_sheet_thickness_m)
            and float(extent.max()) < float(max_sparse_span_m)
        )
        compact_sparse_fragment = bool(
            0 < len(vertices) <= int(max_sparse_vertices)
            and float(extent.max()) < float(max_compact_sparse_span_m)
        )
        invalid_depth = bool(
            len(vertices)
            and (float(vertices[:, 2].max()) < float(valid_workspace_floor_z_m)
                 or float(extent.max()) > float(max_scene_span_m))
        )
        if depth_edge_sheet or compact_sparse_fragment or invalid_depth:
            removed.append(name)
        else:
            kept.append(mesh)
            retained.append(name)
    if not kept:
        raise ValueError("collision reconstruction filtering removed every mesh")
    output = dict(world_config)
    output["meshes"] = kept
    return {"world_config": output, "retained_names": retained,
            "removed_names": removed}
