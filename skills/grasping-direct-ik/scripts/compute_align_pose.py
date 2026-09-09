"""Compute a safe pre-rotated pose above a top-down grasp."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    align_pose: dict[str, Any]
    grasp_pose: dict[str, Any]


def _profile(grasp_profile: dict[str, Any] | None, target_kind: str,
             grasp_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve a declarative profile without embedding category-specific logic.

    ``grasp_profile`` is the canonical interface.  The keyed profile list is a
    compatibility adapter for already-materialized graphs and can be removed
    once their orchestration layer passes the selected profile directly.
    """
    if grasp_profile is not None:
        return dict(grasp_profile)
    if not target_kind and not grasp_profiles:
        return {}
    profiles = {str(item["source_kind"]): item for item in grasp_profiles}
    if target_kind not in profiles:
        raise ValueError(f"no grasp profile declared for profile key {target_kind!r}")
    return dict(profiles[target_kind])


def run(ctx: NodeContext, grasp_pose: dict[str, Any], target_obb: dict[str, Any],
        grasp_profile: dict[str, Any] | None = None, arm_id: int = 0,
        target_kind: str = "",
        grasp_profiles: list[dict[str, Any]] | None = None) -> Output:
    profile = _profile(grasp_profile, target_kind, grasp_profiles or [])
    grasp_pose = {"position": dict(grasp_pose["position"]), "rotation": dict(grasp_pose["rotation"])}
    use_calibration = bool(profile.get(
        "use_embodiment_calibration", bool(target_kind and grasp_profiles)
    ))
    if use_calibration:
        try:
            gripper = ctx.tool("robot.describe_gripper", arm_id=int(arm_id))
        except Exception:
            # Geometry-only fallback remains valid on connectors that do not
            # expose embodiment metadata. The supplied pose is retained.
            gripper = {}
    else:
        gripper = {}
    # Infer yaw from the perceived geometry and active hand metadata.  Align
    # the gripper's declared local closing axis with the shorter horizontal OBB
    # axis.  (YAM closes along local X; several reusable skills assume local Y.)
    # This works for an arbitrarily rotated tool and avoids selecting the first
    # (world-aligned) pose from the generic grasp-candidate fan.
    q = target_obb["orientation"]
    obb_rotation = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    extents = np.array([target_obb["extent"][k] for k in ("x", "y", "z")], dtype=np.float64)
    vertical = int(np.argmax(np.abs(obb_rotation.T @ np.array([0.0, 0.0, 1.0]))))
    horizontal = [i for i in range(3) if i != vertical]
    short = min(horizontal, key=lambda i: extents[i])
    short_axis = obb_rotation[:, short].copy()
    short_axis[2] = 0.0
    if np.linalg.norm(short_axis) > 1.0e-6:
        short_axis /= np.linalg.norm(short_axis)
        # Build the quaternion through the embodiment's calibrated grasp-frame
        # composition. ``close_axis`` is measured in the physical hand/tool
        # frame, while commanded poses are stated at the TCP. YAM's TCP has a
        # calibrated z twist, so treating the former as a TCP-local axis can
        # turn a locked grasp away from the perceived object's short axis.
        heading_deg = float(np.degrees(np.arctan2(short_axis[1], short_axis[0])))
        if gripper:
            grasp_frame = ctx.tool(
                "robot.grasp_frame",
                approach={"x": 0.0, "y": 0.0, "z": -1.0},
                close_heading_deg=heading_deg,
                arm_id=int(arm_id),
            )
            grasp_pose["rotation"] = dict(grasp_frame["rotation"])
    # Keep the fingertip below the TCP from entering the support surface. Thin
    # tools otherwise command a geometrically centred grasp that jams the jaws
    # against the table before they can close. Both reach and clearance come
    # from the active gripper, so the correction survives a hand change.
    finger = gripper.get("finger") or {}
    if finger.get("stated"):
        support_z = float(target_obb["center"]["z"]) - float(target_obb["extent"]["z"])
        min_tcp_z = (
            support_z
            + float(finger.get("reach_m", 0.0))
            + float(finger.get("clearance_m", 0.0))
        )
        grasp_pose["position"]["z"] = max(float(grasp_pose["position"]["z"]), min_tcp_z)
    pose = {"position": dict(grasp_pose["position"]), "rotation": dict(grasp_pose["rotation"])}
    top = float(target_obb["center"]["z"]) + float(target_obb["extent"]["z"])
    clearance = float(profile.get("approach_clearance_m", 0.15))
    pose["position"]["z"] = max(float(pose["position"]["z"]) + clearance,
                                top + clearance)
    return {"align_pose": pose, "grasp_pose": grasp_pose}
