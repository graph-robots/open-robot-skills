"""Linear-descent variant of ``descend_release`` using the connector's TCP-aware
``robot.go_to_pose_cartesian``.

Drop-in replacement for ``descend_release.py`` — same node-level contract:
takes ``drop_position`` (and optionally ``drop_rotation``), descends, opens
the gripper, retracts.

Routes through ``robot.go_to_pose_cartesian`` (gap.connector.ik.CuRoboBackend):
this applies the configured TCP offset / TCP rotation, plans a straight
Cartesian line at the IK link, and falls back internally to a single-pose
collision-aware plan (``plan_to_pose``) when the linear plan cannot solve.
The previous ``curobo.plan_linear`` bundle call interpreted ``drop_position``
as a panda_hand link target, which silently dropped the TCP offset — fine
when the workflow encoded the link-frame offset into pose math, but a
source of vertical misses when the workflow really did mean "put the
fingertips at this XYZ".

When to use this variant:
- For ANY subpart-grasp + place-ON task (frypan handle → stove, kettle
  spout → trivet). The yaw-preserving compute_drop_pose + linear descent
  combination gives the cleanest possible release dynamics.
- A repair pass can flip ``descend_release`` → ``descend_release_linear``
  when the place stage shows visible release artefacts (object tilting on
  touch-down) even though the geometric drop pose is correct.

When NOT to use it:
- When the descent path needs to avoid an obstacle. ``go_to_pose_cartesian``
  refuses to plan around obstacles — it only checks the straight line and
  falls back to a collision-free single-pose plan if the line fails. Use
  the ``descend_release.py`` variant for cluttered scenes that need
  smoother IK-only descent.

Inputs:
- ``drop_position`` (Vec3): TCP target XYZ (fingertip position), same as
  descend_release.
- ``drop_rotation`` (Quaternion, optional): TCP orientation held through
  the descent. Defaults to canonical top-down ``(w=0, x=1, y=0, z=0)``.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3


# Canonical top-down orientation (gripper Z aligned with world -Z).
_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
) -> Output:
    rotation = drop_rotation if drop_rotation is not None else _DOWN
    end_pose: Se3Pose = {"position": drop_position, "rotation": rotation}

    ctx.tool("robot.go_to_pose_cartesian", pose=end_pose)

    # Open the gripper and settle long enough for the object to land
    # before the retract starts moving the arm laterally.
    ctx.tool("robot.open_gripper", settle_steps=60)

    # Retract home so the just-placed object isn't disturbed.
    ctx.tool("robot.go_home")

    return {"drop_position": drop_position}
