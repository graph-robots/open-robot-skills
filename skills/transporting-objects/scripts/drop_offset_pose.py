"""Apply a held-object → grasp-point XY offset correction to a drop pose.

Runs AFTER ``compute_drop_pose`` and BEFORE ``move_above`` / ``release``.
``compute_drop_pose`` places the **TCP (fingertip)** at the placement
zone's XY centroid — which puts the **held grasp point** there.  When the
robot grasped a *subpart* (e.g. a frypan handle, a kettle spout, a bottle
neck) the rest of the object body hangs off-centre and lands off the
target.

This node computes the rigid XY displacement between the grasp point
(``held_obb``, usually the perceived subpart used for the grasp) and the
full object centroid (``parent_obb``, usually the parent perception
output) at grasp time, transports it through the gripper-rotation
difference between grasp and drop, and shifts the drop pose by ``-Δxy``
so the **parent centroid** — not the grasp point — lands at the zone
centroid.

Inputs:
- ``drop_pose``: ``Se3Pose`` from ``compute_drop_pose.drop_pose``.
- ``ee_pose_at_grasp``: ``Se3Pose`` of the EE when the gripper closed
  (passed through from the grasp subgraph's ``ee_pose_at_grasp`` output).
- ``held_obb``: ``OrientedBoundingBox`` of the subpart actually grasped
  (the same OBB plumbed into ``compute_drop_pose.held_obb`` — typically
  ``target_sg.subpart_obb`` or ``target_sg.obb`` when grasping a subpart).
- ``parent_obb``: ``OrientedBoundingBox`` of the full object — the
  parent geometry that should land at the zone centre. When the
  perception subgraph used ``perceiving-object-parts`` this is the parent
  output; when only a single OBB is available, wire ``parent_obb =
  held_obb`` and this node degrades to identity (no shift).
- ``approach_height``: ``float`` Z offset for ``approach_pose`` above the
  corrected drop. Default 0.20 m matches ``compute_drop_pose``.

Returns ``drop_position``, ``drop_pose``, ``approach_pose`` — all
shifted in XY by the same ``-Δxy``. Z is preserved (``compute_drop_pose``
already accounts for held-object thickness in Z).

Why a separate node and not folded into ``compute_drop_pose``:
``compute_drop_pose`` only sees ONE OBB (the held one). Adding the
parent-OBB plumbing there would force every legacy caller to perceive +
wire a second OBB even when they grasp the object centroid directly.
Splitting the offset into its own node lets the coordinator omit it on
center-grasp tasks while keeping the contract clean for subpart-grasp
tasks (frypan handle → stove, kettle spout → trivet, etc.).
"""

from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap.types import OrientedBoundingBox, Se3Pose, Vec3
from scipy.spatial.transform import Rotation as _R


class Output(TypedDict):
    drop_position: Vec3
    drop_pose: Se3Pose
    approach_pose: Se3Pose


def _quat_to_R(q) -> np.ndarray:
    """gap Quaternion (wxyz dict) → 3×3 rotation matrix."""
    return _R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()


def _vec3(x: float, y: float, z: float) -> Vec3:
    return {"x": float(x), "y": float(y), "z": float(z)}


def run(
    ctx: NodeContext,
    drop_pose: Se3Pose,
    ee_pose_at_grasp: Se3Pose,
    held_obb: OrientedBoundingBox,
    parent_obb: OrientedBoundingBox,
    approach_height: float = 0.20,
) -> Output:
    # World-frame XY offset from the grasp point (held centroid) to the
    # full-object centroid AT GRASP TIME. The Z component is intentionally
    # ignored — compute_drop_pose already corrects Z via held_obb extent.z
    # and ee_pose_at_grasp.
    delta_world_grasp = np.array([
        parent_obb["center"]["x"] - held_obb["center"]["x"],
        parent_obb["center"]["y"] - held_obb["center"]["y"],
        parent_obb["center"]["z"] - held_obb["center"]["z"],
    ], dtype=np.float64)

    # The held object is rigid w.r.t. the gripper. In the gripper's local
    # frame the offset is constant; in the world frame it rotates with
    # the gripper. We need the offset in the world frame at *drop* time,
    # which is (R_drop · R_graspᵀ) applied to the grasp-time offset.
    R_grasp = _quat_to_R(ee_pose_at_grasp["rotation"])
    R_drop = _quat_to_R(drop_pose["rotation"])
    R_delta = R_drop @ R_grasp.T
    delta_world_drop = R_delta @ delta_world_grasp

    # Shift the drop pose by -Δxy so the parent centroid lands where the
    # original drop_position was targeted. Z is untouched.
    corrected_x = drop_pose["position"]["x"] - float(delta_world_drop[0])
    corrected_y = drop_pose["position"]["y"] - float(delta_world_drop[1])
    corrected_z = drop_pose["position"]["z"]

    drop_position = _vec3(corrected_x, corrected_y, corrected_z)
    corrected_drop_pose: Se3Pose = {
        "position": drop_position, "rotation": drop_pose["rotation"],
    }
    approach_pose: Se3Pose = {
        "position": _vec3(corrected_x, corrected_y, corrected_z + approach_height),
        "rotation": drop_pose["rotation"],
    }

    return {
        "drop_position": drop_position,
        "drop_pose": corrected_drop_pose,
        "approach_pose": approach_pose,
    }
