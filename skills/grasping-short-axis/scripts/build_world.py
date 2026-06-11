"""Thin adapter: extract joint state from observation, call build_world_config.

Calls ``geometry.build_world_config`` to reconstruct the scene as a set of
collision meshes, excluding the robot body. Unwraps
``observation["arms"][0]["joint_state"]`` (the ``$ref`` DSL cannot index
into repeated fields via bracket notation) and forwards the rest.

Target isolation, in priority order:

1. ``target_mask`` (preferred): the pixel-accurate segmentation mask from
   perception (e.g. SAM3 output via ``perceiving-*`` skills). Wrapped into
   an object-mask entry ``{name: target_name, mask, camera_index: 0}``. The
   target mesh captures the object's true silhouette so the rest of the
   scene mesh is free of object leakage.

2. ``target_obb`` (fallback): the geometry bundle projects the OBB's 8
   corners onto the first camera and fills the axis-aligned image-space
   rectangle as a mask. Works when ``target_mask`` is unavailable but is
   lossy for degenerate OBBs.

Pass BOTH when available — ``target_mask`` wins for isolation, and the
OBB remains available elsewhere (top_down_grasp_candidates, etc.).
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import Mask, Observation, OrientedBoundingBox, WorldConfig


class Output(TypedDict):
    config: WorldConfig


def run(
    ctx: NodeContext,
    observation: Observation,
    target_mask: Mask | None = None,
    target_obb: OrientedBoundingBox | None = None,
    target_name: str = "target",
    robot_distance_threshold: float = 0.15,
    robot_file: str = "",
) -> Output:
    kwargs = dict(
        cameras=observation["cameras"],
        robot_distance_threshold=robot_distance_threshold,
    )
    if observation.get("arms"):
        kwargs["robot_joint_state"] = observation["arms"][0]["joint_state"]

    if target_mask is not None:
        kwargs["object_masks"] = [
            {"name": target_name, "mask": target_mask, "camera_index": 0}
        ]
    elif target_obb is not None:
        kwargs["target_obb"] = target_obb
        kwargs["target_obb_name"] = target_name

    if robot_file:
        kwargs["robot_file"] = robot_file

    resp = ctx.tool("geometry.build_world_config", **kwargs)
    return {"config": resp["config"]}
