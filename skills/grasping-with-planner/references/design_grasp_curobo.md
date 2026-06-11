# Why `use_grasp_approach=False` in `plan_grasp.py`

CuRobo's `curobo.plan_to_grasp_poses` exposes a flag, `use_grasp_approach`,
that biases the trajectory to follow a two-phase profile:

1. Free-space motion to a *pre-grasp* pose offset back along the gripper's
   approach axis by ``grasp_approach_offset`` (default 12 cm).
2. Linear-Cartesian descent from the pre-grasp pose to the grasp pose,
   over the last ``grasp_approach_tstep_fraction`` (default 70 %) of the
   trajectory.

In theory this enforces "approach the target along the gripper Z axis,"
which avoids drag-across-target failure modes. In practice the metric's
internal ``reach_vec_weight`` heavily downweights the Z component of
position (~0.2 vs the ~1.0 weight for X, Y, and orientation), so the
optimizer is happy to "complete" the trajectory at the pre-grasp pose
without ever executing the descent. The result: the gripper closes
above the target. Slip every time.

Our workflow already pre-rotates the gripper into grasp orientation via
the `approach` state (script `approach_above.py`) that runs between
`compute_grasp` and `observe` in this skill's canonical flow. Once the
gripper is correctly oriented above the target, CuRobo's job is just to
descend from there to the grasp pose — there is no rotation-during-descent
to guard against. Disabling `use_grasp_approach` and letting the planner
do a direct goal-set plan to the grasp pose works reliably.

If you ever re-enable `use_grasp_approach`:

- Verify the issued plan actually descends. Print the last waypoint's
  Z coordinate.
- Tune `reach_vec_weight` upward in CuRobo's metric config (the planner
  source, not the tool request).
- Don't combine with the in-skill `approach` state — pick one approach
  strategy and remove the other.
