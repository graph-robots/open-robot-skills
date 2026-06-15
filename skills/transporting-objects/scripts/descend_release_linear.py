"""Linear-descent variant of ``descend_release`` using cuRobo ``plan_linear``.

Drop-in replacement for ``descend_release.py`` — same node-level
contract: takes ``drop_position`` (and optionally ``drop_rotation``),
descends, opens the gripper, and retracts. The difference is in HOW
the descent is executed:

- ``descend_release.py`` uses ``robot.go_to_pose`` for the descent,
  which goes through TrajOpt-style IK and may produce a curved
  end-effector path (the solver is free to deviate from a straight
  Cartesian line as long as start + goal pose constraints are met). On
  asymmetric loads (subpart-grasped frypan / kettle), it sometimes
  picks an IK solution that swings the elbow during the descent — the
  held object visibly tilts before release.
- ``descend_release_linear.py`` (this file) calls ``curobo.plan_linear``
  with the current EE pose as ``start_pose`` and ``(drop_position,
  drop_rotation)`` as ``end_pose``. plan_linear constrains the
  trajectory to a straight Cartesian line in EE space, holding the
  orientation between the two endpoints — the held object descends
  vertically with no spurious rotation. Faster to execute because
  the planner doesn't search for IK alternatives; cheaper to verify
  visually because the EE trace is a pure line.

When to use this variant:
- For ANY subpart-grasp + place-ON task (frypan handle → stove, kettle
  spout → trivet). The yaw-preserving compute_drop_pose + linear
  descent combination gives the cleanest possible release dynamics.
- A repair pass can flip ``descend_release`` →
  ``descend_release_linear`` when the place stage shows visible release
  artefacts (object tilting on touch-down) even though the geometric
  drop pose is correct.

When NOT to use it:
- When the descent path needs to avoid an obstacle. plan_linear refuses
  to plan around obstacles — it only checks the straight line and fails
  if any waypoint collides. Use the go_to_pose-based
  ``descend_release.py`` for cluttered scenes.

Inputs:
- ``drop_position`` (Vec3): final TCP target XYZ, same as descend_release.
- ``drop_rotation`` (Quaternion, optional): EE orientation held through
  the descent. Defaults to canonical top-down ``(w=0, x=1, y=0, z=0)``
  when unset.
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

    obs = ctx.tool("robot.get_observation")
    arm = obs["arms"][0]
    start_pose = arm["ee_pose"]
    start_joints = arm["joint_state"]

    end_pose: Se3Pose = {"position": drop_position, "rotation": rotation}

    plan = ctx.tool(
        "curobo.plan_linear",
        start_pose=start_pose,
        end_pose=end_pose,
        start_joint_position=start_joints,
    )
    if not plan["success"]:
        # Linear plan failed. Fall back to the IK-based go_to_pose so the
        # workflow still completes — the descent may not be a clean line
        # but it should still reach the target.
        ctx.tool("robot.go_to_pose", pose=end_pose)
    else:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=plan["trajectory"],
            subsample=2,
            max_steps_per_waypoint=8,
            tolerance=0.02,
        )

    # Open the gripper and settle long enough for the object to land
    # before the retract starts moving the arm laterally.
    ctx.tool("robot.open_gripper", settle_steps=60)

    # Retract home so the just-placed object isn't disturbed.
    ctx.tool("robot.go_home")

    return {"drop_position": drop_position}
