---
name: executing-feature-mating
description: Execute a supplied feature-mating motion plan for an already-held rigid object, then release and retract. Use for constrained loop-over-shaft, shaft-into-aperture, and support-seating operations; do not use for ordinary container drops.
compatibility: requires gap>=0.1
metadata: {category: motion, tags: [motion, placement, insertion, fixtures]}
gap:
  allowed_tools:
    - robot.get_ee_pose
    - robot.execute_trajectory
    - robot.describe_arm
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.move_cartesian_until_contact
    - robot.open_gripper
    - robot.go_home
    - robot.wait_steps
    - motion.plan_to_pose
    - motion.plan_linear
  exit_conditions:
    placed: The feature-mating waypoint sequence completed and the object was released.
    blocked: A waypoint or release motion failed before completion.
  required_inputs:
    placement_plan: PoseSequence
    relation: string
  produces_outputs:
    final_pose: Se3Pose
    fallback_count: int
  canonical_scripts:
    - execute_waypoints: scripts/execute_waypoints.py
    - release_and_retract: scripts/release_and_retract.py
  streaming: false
---

# executing-feature-mating

Execute a previously computed feature-mating plan for an already-grasped rigid
object, then release it. The upstream localization/planning node supplies
ordered poses; this skill does not locate a fixture or compute mating poses and
does not depend on object names, task names, simulator state, or evaluator
predicates.

Use this instead of `transporting-objects` when the goal is a constrained mate,
not a drop into a volume or onto a broad surface. Examples are putting a loop
over a rod, inserting a rod through a loop, or lowering a handle gap onto a
support point.

## Recommended subgraph

```text
execute_waypoints -> release -> placed
```

- `execute_waypoints` runs `scripts/execute_waypoints.py` with
  `placement_plan = Ref("in.placement_plan")`. It also accepts `relation`,
  `contact_profile`, `arm_id`, and `registration_uncertainty_m`. Use
  `planned_joint` for a
  collision-aware transit, `planned_linear` for an orientation-locked
  approach/insertion, and `contact_seat` only for the final intended-contact
  leg. The plan carries its collision world and attached-object spheres.
- When insertion intentionally ends in contact, use
  `robot.move_cartesian_until_contact` for that final straight leg. It stops
  when the target is reached or measured TCP progress stalls, avoiding a long
  exact-pose timeout without relying on object names or simulator contacts.
- `release` runs `scripts/release_and_retract.py` with the final waypoint pose.
- Route any raised motion error through `on_error: blocked`.

Every completed leg returns a report containing its waypoint index and mode,
attempt count, translational and rotational residual, contact status, and any
fallback used. Profiles declare tolerances, retry budgets, contact margins,
release settling, and retreat geometry. Do not select these values from object
or task names. Registration uncertainty is diagnostic by default; a profile
may use it upstream to generate a wider-clearance plan, but execution must not
silently loosen pose acceptance because perception is uncertain.

## Boundaries

- This skill does not infer fixture geometry or choose feature landmarks.
- Do not query `sim.*`, cameras, rewards, or goal predicates inside this skill.
- Preserve the waypoint order. Insertion plans encode clearance first, mating
  second, seating third; shortcutting between them can cross solid geometry.
- Never fall back to unchecked robot motion after a collision-aware planner
  rejects a waypoint unless the plan/profile explicitly authorizes a bounded
  local Cartesian recovery and the executor verifies the resulting pose.
- Use `transporting-objects` for bins, baskets, and unconstrained surface drops.
