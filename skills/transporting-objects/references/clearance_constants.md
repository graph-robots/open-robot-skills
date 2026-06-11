# Approach / lift / drop clearance constants

The transport bundle's scripts use a few empirical clearance constants
that need explanation rather than burial in code.

## `_OBB_APPROACH_CLEARANCE = 0.15` (in `approach_above.py`)

Distance to lift above the OBB top before any lateral motion. Same value
as the grasp bundles' align-pose clearance — the gripper is ~5–8 cm tall
on Franka, plus margin to avoid clipping the OBB during rotation.

## `_OBB_LIFT_CLEARANCE = 0.05` (in `lift_grasped.py`)

When `target_obb` is supplied, the lift height is set to
`target_obb.center.z + target_obb.extent.z + 0.05`. This is used after
the gripper has closed on the object, so we no longer need full
approach-clearance — just enough to clear the OBB top with the held
object hanging below the gripper.

## `_MIN_LIFT_DELTA = 0.10` (in `lift_grasped.py`)

Minimum absolute Z displacement during the lift, regardless of OBB-derived
height. Prevents pathological cases where the OBB is so flat that the
OBB-derived lift would be a few centimeters and the held object would
still drag.

## Hardcoded `0.45` waypoint height (in `waypoint_move.py`)

Safety height for lateral motion during transport. Tuned for LIBERO-style
tabletop scenes; on tasks with overhead obstacles (shelves, lighting
rigs) this needs adjustment. A future revision should accept this as a
parameter.

## Settle behavior in `descend_release.py`

- The descend (`robot.go_to_pose` to the drop_position) blocks until the
  controller converges, so the arm is stable before the gripper opens.
- `settle_steps=60` on `robot.open_gripper` — gripper-controller settling
  plus drop-settling time so the released object lands before the
  retract starts moving the arm laterally.
- `robot.go_home` retracts only after the open-settle completes, so the
  freshly-placed object isn't disturbed by the retract motion.
