"""Select the next source instance using declarative categories and geometry.

Materialized from the reusable perceiving-relational-correspondences skill.
Task semantics and workspace calibration are supplied by the graph inputs.
"""

from __future__ import annotations

from hashlib import sha1
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext


class Output(TypedDict):
    status: str
    instruction: str
    source_id: str
    target_kind: str
    target_mask: np.ndarray
    destination_anchor_y: float
    arm_id: int


def _empty() -> np.ndarray:
    return np.zeros((1, 1), dtype=np.uint8)


def _camera(observation: dict[str, Any], camera_name: str) -> dict[str, Any]:
    return next(c for c in observation["cameras"] if c.get("name") == camera_name)


def _category(instruction: str, categories: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(instruction).casefold()
    matched = []
    for category in categories:
        aliases = category.get("aliases") or [category.get("kind", "")]
        if any(str(alias).casefold() in text for alias in aliases if str(alias)):
            matched.append(category)
    if len(matched) == 1:
        return matched[0]
    if not matched and len(categories) == 1:
        return categories[0]
    if not matched:
        raise ValueError("instruction does not identify any declared source category")
    raise ValueError("instruction ambiguously identifies multiple source categories")


def _inside_workspace(center: np.ndarray, workspace: dict[str, Any]) -> bool:
    for axis, index in {"x": 0, "y": 1, "z": 2}.items():
        if f"{axis}_min" in workspace and center[index] < float(workspace[f"{axis}_min"]):
            return False
        if f"{axis}_max" in workspace and center[index] > float(workspace[f"{axis}_max"]):
            return False
    return True


def _stable_id(kind: str, center: np.ndarray, quantization_m: float) -> str:
    quantized = np.rint(center / max(float(quantization_m), 1.0e-6)).astype(np.int64)
    token = f"{kind}:" + ":".join(str(int(v)) for v in quantized)
    return f"{kind}-{sha1(token.encode('utf-8')).hexdigest()[:10]}"


def run(ctx: NodeContext, observation: dict[str, Any], instruction: str,
        source_categories: list[dict[str, Any]], source_workspace: dict[str, Any],
        correspondence_axis: str = "y", arm_partition: dict[str, Any] | None = None,
        completed_source_ids: list[str] | None = None,
        camera_name: str = "overhead") -> Output:
    category = _category(instruction, source_categories)
    kind, query = str(category["kind"]), str(category["query"])
    camera = _camera(observation, camera_name)
    result = ctx.tool("sam3.segment_text", image=camera["rgb"], query=query,
                      max_results=int(category.get("max_results", 6)))
    scores = list(result.get("scores") or [])
    minimum_score = float(category.get("minimum_score", 0.08))
    minimum_pixels = int(source_workspace.get("minimum_mask_pixels", 80))
    minimum_points = int(source_workspace.get("minimum_cloud_points", 20))
    quantization = float(source_workspace.get("id_quantization_m", 0.01))
    completed = set(completed_source_ids or [])
    valid = []
    for index, raw_mask in enumerate(result.get("masks") or []):
        score = float(scores[index]) if index < len(scores) else 0.0
        mask = np.asarray(raw_mask, dtype=np.uint8)
        if score < minimum_score or int(np.count_nonzero(mask)) < minimum_pixels:
            continue
        cloud = ctx.tool("geometry.mask_to_world_points", mask=mask, depth=camera["depth"],
                         intrinsics=camera["intrinsics"], camera_pose=camera["pose"])["points"]
        points = np.asarray(cloud["points"], dtype=float).reshape(-1, 3)
        if len(points) < minimum_points:
            continue
        center = np.median(points, axis=0)
        if not _inside_workspace(center, source_workspace):
            continue
        source_id = _stable_id(kind, center, quantization)
        if source_id not in completed:
            valid.append((score, source_id, mask, center))
    if not valid:
        return {"status": "finished", "instruction": instruction, "source_id": "",
                "target_kind": kind,
                "target_mask": _empty(), "destination_anchor_y": 0.0, "arm_id": 0}
    _, source_id, mask, center = max(valid, key=lambda item: item[0])
    axis_index = {"x": 0, "y": 1, "z": 2}
    if correspondence_axis not in axis_index:
        raise ValueError(f"unsupported correspondence_axis {correspondence_axis!r}")
    anchor = float(center[axis_index[correspondence_axis]])
    partition = arm_partition or {"split": 0.0, "positive_arm": 0, "negative_arm": 1}
    arm_id = int(partition.get("positive_arm", 0) if anchor >= float(partition.get("split", 0.0))
                 else partition.get("negative_arm", 1))
    return {"status": "found", "instruction": instruction, "source_id": source_id,
            "target_kind": kind, "target_mask": mask,
            "destination_anchor_y": anchor, "arm_id": arm_id}
