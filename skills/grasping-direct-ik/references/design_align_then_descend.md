# Why pre-rotate-then-descend instead of blended rotate+descend

A naive direct-IK grasp would call `robot.go_to_pose` once with the grasp
pose as target. The IK solver returns a joint trajectory that interpolates
between the current pose and the goal: both position **and** rotation change
simultaneously along the path. For a top-down grasp on a small object this
produces a corkscrew motion — the gripper twists into orientation while
also descending — and tends to drag the gripper across the target's top
surface.

`grasping-direct-ik` solves this by splitting the motion into two segments:

1. **Pre-rotate at the align-pose.** Move to `(grasp.x, grasp.y, top + 0.15)`
   with the *grasp orientation*. The position lies directly above the
   target; the rotation matches the final grasp; only translation and
   rotation change here. Because the path is mostly translation in free
   space, the corkscrew effect is harmless.

2. **Descend straight down.** Move from the align-pose to the grasp pose.
   The rotation stays fixed; only Z decreases. The gripper drops vertically
   onto the target.

This decomposition is also what `grasping-with-planner` does internally — it
runs `approach_above` (which sets the orientation high) before
running the actual planner. The principle is the same; the implementation
differs because the CuRobo path can rely on a planner to avoid obstacles
on the way down whereas direct-IK assumes the descent is unobstructed.

## Clearance choice

The 0.15 m clearance above the OBB top is empirical. It must be at least:

- The end-effector tooltip-to-flange distance (~5–8 cm on Franka with the
  default fingers).
- Plus a few cm of safety margin to avoid clipping the OBB during
  rotation.

A tighter clearance trades safety for cycle time. A looser clearance is
always safe but slower. The constant in `compute_align_pose.py` is the
documented default; per-task overrides should ship as a SKILL.md
frontmatter parameter override or as a workflow-level `inputs:` field.
