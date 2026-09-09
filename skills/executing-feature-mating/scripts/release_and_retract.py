"""Release at the mate, then retreat away from the fixture without sweeping it."""

from typing import Any, TypedDict

from gap import NodeContext
import numpy as np


class Output(TypedDict):
    released: bool
    retreat_pose: dict[str, Any]
    release_report: dict[str, Any]


def run(ctx: NodeContext, final_pose: dict[str, Any], retract_m: float = 0.08,
        retreat_axis: dict[str, float] | None = None,
        attached_object: dict[str, Any] | None = None,
        relation: str = "loop_over_shaft", arm_id: int | None = None,
        contact_profile: dict[str, Any] | None = None) -> Output:
    profile = contact_profile or {}
    if arm_id is None:
        arm_id = int((attached_object or {}).get("arm_id", 0))
    release_steps = int(profile.get("release_settle_steps", 80))
    ctx.tool("robot.open_gripper", settle_steps=release_steps, arm_id=int(arm_id))
    retreat = {
        "position": dict(final_pose["position"]),
        "rotation": dict(final_pose["rotation"]),
    }
    axis = np.array([0.0, 0.0, 1.0])
    reverse_relations = set(profile.get(
        "reverse_retreat_relations",
        ["shaft_into_aperture", "tip_through_aperture", "insert_through"],
    ))
    if retreat_axis is not None:
        axis = np.array([retreat_axis[k] for k in ("x", "y", "z")], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        if relation in reverse_relations:
            axis = -axis
    spheres = list((attached_object or {}).get("spheres") or [])
    extent = max((float(np.linalg.norm([s["center"][k] for k in ("x", "y", "z")]))
                  + float(s.get("radius", 0.0)) for s in spheres), default=0.0)
    distance = max(float(profile.get("retract_m", retract_m)),
                   extent + float(profile.get("object_clearance_m", 0.01)))
    for key, component in zip(("x", "y", "z"), axis, strict=True):
        retreat["position"][key] = float(retreat["position"][key]) + distance * component
    if retreat_axis is not None:
        retreat["position"]["z"] = (
            float(retreat["position"]["z"])
            + float(profile.get("vertical_retreat_fraction", 0.5)) * distance
        )
    ctx.tool("robot.go_to_pose_cartesian", pose=retreat, arm_id=int(arm_id))
    wait_steps = int(profile.get("post_release_wait_steps", 120))
    ctx.tool("robot.wait_steps", steps=wait_steps)
    return {
        "released": True,
        "retreat_pose": retreat,
        "release_report": {
            "relation": relation,
            "arm_id": int(arm_id),
            "release_settle_steps": release_steps,
            "retreat_distance_m": distance,
            "post_release_wait_steps": wait_steps,
        },
    }
