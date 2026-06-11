# Gripper settle-step constants

Gripper open/close commands send a target position to the controller, but
the controller's joint motion takes time. Without explicit settling, the
next state runs while the fingers are still in flight — the descent state
fires before the gripper has actually opened, and the close state fires
before it has actually gripped. Both failure modes are silent.

The default settle constants used across `grasping-*` skills:

- `_OPEN_SETTLE_STEPS = 40` — typical 40-tick simulation step at 200 Hz
  (≈200 ms on Franka in MuJoCo) is enough for the fingers to fully
  retract from the previous grip width.
- `_CLOSE_SETTLE_STEPS = 60` — closing under contact is slower because
  the fingers stop on the target instead of at their hardware limits;
  60 ticks (~300 ms) gives the controller time to apply the holding force.

These are simulator-validated; on real hardware the appropriate values
depend on the gripper's bandwidth and the target's compliance. The
`robot.open_gripper` / `robot.close_gripper` tools accept any
non-negative `settle_steps` int; passing zero falls back to the connector
defaults (40 open / 60 close) rather than skipping settling — but
declaring the values explicitly in the workflow keeps the tuning visible.
