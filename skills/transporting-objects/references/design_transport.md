# Why no planner — direct waypoint motion suffices

The transport bundle uses a fast 2-waypoint pattern (`waypoint_move.py`):

1. Lift to safe height at current XY.
2. Move laterally to the drop XY.

No collision-aware planning. This works because:

1. **The pick is over.** The gripper is holding the target. Once we are
   above ~0.45 m, the only obstacles to lateral motion are the same
   obstacles the perception saw — typically static scene clutter that's
   well below transport altitude.
2. **The release path is a straight descent.** The drop XY is centered
   on the destination container's OBB top face; descent is vertical from
   the safe height down to the release point. No planning required for
   the descent either.

This pattern fails when:

- The destination has overhead obstacles (a shelf above the container).
  The 0.45 m safety height is calibrated for LIBERO-style open scenes;
  shelves break the assumption.
- The held object is large enough that the lift clearance + the held
  object's bottom collides with scene objects en route. In LIBERO this
  hasn't been observed; on real cluttered shelves, this assumption needs
  reconsideration.

A `transport_curobo` composite is on the roadmap for those cases. For
the default open-tabletop scope, the direct waypoint motion is faster
(no planning latency) and equally reliable.
