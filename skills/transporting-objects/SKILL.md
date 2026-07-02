---
name: transporting-objects
description: Move the currently-held object to a destination and release. The
  gripper enters this subgraph holding the object; on exit the object has been
  placed. For a walled container (basket / bin / box / tote) the default is a
  straight-Z descent to inside the walls (transport_descend_linear → release);
  for placement ONTO a surface or into a described sub-region it computes an
  explicit drop pose (compute_drop → move_above → release). Use after a
  successful grasp when the destination has a known OBB. When the destination is
  a sub-region described in natural language (e.g. "the left compartment of the
  caddy", "to the left of the plate", "the inside of the top drawer"), an
  optional VLM-grounded perceive_zone state localizes the zone before the drop
  pose is computed.
compatibility: requires gap>=0.1
metadata: {category: motion, tags: [motion, transport, place, drop]}
gap:
  allowed_tools:
    - robot.go_to_pose
    - robot.go_to_pose_cartesian
    - robot.go_home
    - robot.open_gripper
    - robot.get_ee_pose
    - robot.get_observation
    - robot.execute_trajectory
    - geometry.compute_drop_position
    - geometry.mask_to_world_points
    - geometry.filter_and_compute_obb
    - geometry.build_world_config
    - curobo.plan_with_grasped_object
    - curobo.plan_directed_linear
    - grounding-dino.detect
    - vlm.query
    - sam3.segment_box
    - sam3.segment_text
  exit_conditions:
    placed: Object released at the destination.
    blocked: Path blocked or motion failed; aborted before release. Coordinator routes to abort.
  required_inputs:
    container_obb: OrientedBoundingBox
    container_mask: Mask
    target_obb: OrientedBoundingBox
    target_mask: Mask
    ee_pose_at_grasp: Se3Pose
  canonical_scripts:
    - transport_descend_linear: scripts/transport_descend_linear.py
    - place_release: scripts/place_release.py
    - compute_drop_pose: scripts/compute_drop_pose.py
    - drop_offset_pose: scripts/drop_offset_pose.py
    - approach_above: scripts/approach_above.py
    - descend_release: scripts/descend_release.py
    - descend_release_linear: scripts/descend_release_linear.py
    - lift_grasped: scripts/lift_grasped.py
    - waypoint_move: scripts/waypoint_move.py
    - waypoint_move_carve: scripts/waypoint_move_carve.py
    - perceive_placement_zone: scripts/perceive_placement_zone.py
  prompts:
    vlm_select_zone: prompts/vlm_select_zone.md
  references:
    - title: Approach / lift / drop clearance constants
      path: references/clearance_constants.md
    - title: Why no planner — direct waypoint motion suffices
      path: references/design_transport.md
  examples:
    - title: Canonical walled-container placement (transport_descend_linear → release)
      path: examples/canonical_subgraph.json
  streaming: false
---

# transporting-objects

The pick has happened; the gripper is holding the target. This subgraph
moves the held object above the destination container and releases it.

## When to use

- After a successful grasp (`grasped` end state of any `grasping-*` skill).
- When the destination is a container with a known OBB (cup, bin, basket).

## When NOT to use

- Cluttered transport paths where the lifted object risks colliding with
  other scene objects in transit — but see the `waypoint_move_carve`
  variant below, which routes the lift/translate through
  `curobo.plan_with_grasped_object` against a rebuilt collision world.

## Recommended subgraph state flow

Choose the flow by WHERE the object must land.

### Default — a walled container (basket / bin / box / tote): 2 states

**HARD RULE — pick this path whenever the destination is a container with walls**
(a basket, bin, tote, box, or cup) — i.e. essentially every "put/place X **in**
the <container>" task. Use the 2-state `transport_move → release` flow below and
do **NOT** use the `compute_drop → move_above → release` drop-pose path (that
path releases at/above the rim for a walled container and the object misses).
The drop-pose path in the next section is ONLY for placing **onto** a surface or
into a **named sub-region**.

The object must end up INSIDE the walls, so descend straight down into it. This
is the reference VAB packing recipe and needs only `container_obb`:

```text
transport_move → release
```

- **`transport_move`** — `type: script`,
  `scripts/<sg>/transport_descend_linear.py`. Use the canonical script as-is
  (materialized automatically; do **not** re-emit it). Inputs:
  `container_obb = Ref("in.container_obb")`, `place_offset = -0.06` (the TCP
  descends to just above the rim so the held object clears the walls and drops
  in). It lifts, moves over the container, and does an axis-locked straight-Z
  descend (`curobo.plan_directed_linear`, fingertip-frame). Returns
  `place_position`.
- **`release`** — `type: script`, `scripts/<sg>/place_release.py`.
  Input `place_position = Ref("transport_move.place_position")`. Opens the
  gripper with a settle so the object lands, then a linear straight-up retract.
  On success → exit `placed`; on failure → `blocked`.

Emit `examples/canonical_subgraph.json` verbatim for this path. Do **not** add a
`compute_drop`/`move_above` here — the descend already places into the walls.

### Placement ONTO a surface, or a described sub-region/zone, or a subpart grasp: the drop-pose path

Here the target is a surface top or a localized zone (not "inside walls"), so
compute an explicit drop pose. Either **3 states** or **4 states** (when a
sub-region must be localized at drop time). Do **not** add more.

3 states:

```text
compute_drop → move_above → release
```

4 states (when the task description names a placement sub-region):

```text
perceive_zone → compute_drop → move_above → release
```

For this path prefer the `descend_release_linear` release variant (TCP-aware
straight Cartesian line) over `descend_release` — the latter targets the
panda_hand link and drops the ~0.10 m TCP offset, a common source of vertical
placement misses.

**Hard rule — no re-perception of the container.** Do NOT add states
named `re_perceive_container`, `reobserve_container`, `re_observe_*`,
`re_filter_obb`, or any equivalent that re-detect the container. The
container's `OrientedBoundingBox` and `Mask` flow in via
`in.container_obb` / `in.container_mask` from the upstream perception
subgraph and are reliable for the drop.

**Carve-out — placement-zone perception is allowed.** When the task
description names a sub-region (e.g. *"the left compartment of the
caddy"*, *"the inside of the top drawer of the cabinet"*, *"on top of
the cabinet shelf"*), insert ONE state named `perceive_zone` of
`type: script` running the canonical
`scripts/<sg>/perceive_placement_zone.py`. The script calls
`robot.get_observation` once and grounds the zone with a
two-path pipeline that mirrors `perceiving-objects`'s picking pattern:
(A) `grounding-dino.detect` + `vlm.query` letter-pick over labeled
boxes + `sam3.segment_box`; (B) `sam3.segment_text` fallback. The VLM
only ever returns a single letter (or `none`) — never pixel
coordinates. Its output flows into `compute_drop` as
`container_interior_obb` and is the ONLY downstream consumer. When
the placement is unambiguously the bare container (e.g. *"put X in
the basket"*), omit the `perceive_zone` state entirely (3-state
flow). Do not add any other perception/observation states inside this
subgraph.

State details:

0. **`perceive_zone`** *(optional — include only when the task names a
   placement sub-region)* — `type: script`, file
   `scripts/<sg>/perceive_placement_zone.py`. Use the canonical script
   as-is; do **not** re-emit a ``` ```python:``` ``` block for this
   path. Returns `{placement_zone_obb: OrientedBoundingBox | None}`.

   **Hard rule on `placement_description`:** like `object_name` in
   `perceiving-objects`, this is a literal Python string — the natural
   noun phrase describing **where the held object should land**, drawn
   from this subgraph's description. It is a constant per subgraph
   instance, NOT a binding. **DO NOT** write
   `Ref("in.placement_description")` or any `$ref`; the
   coordinator does not declare `placement_description` as a subgraph
   input. Write the string directly.

   The phrase should describe the placement REGION, not the named
   reference object. Examples:
   - Task: "Pick up the book and place it in the **left compartment of
     the caddy**." → `"the left compartment of the caddy"`.
   - Task: "Put the chocolate pudding **to the left of the plate**." →
     `"the area to the left of the plate"`.
   - Task: "Put the ketchup **in the top drawer of the cabinet**." →
     `"the inside of the top drawer of the cabinet"`.
   - Task: "Pick up the book and place it **on top of the shelf**." →
     `"on top of the cabinet shelf"`.

   ```json
   "perceive_zone": {
     "type": "script",
     "script": "scripts/<sg>/perceive_placement_zone.py",
     "inputs": {
       "placement_description": "the left compartment of the caddy",
       "container_obb":  Ref("in.container_obb"),
       "container_mask": Ref("in.container_mask")
     }
   }
   ```

   Both `container_obb` and `container_mask` are OPTIONAL but strongly
   recommended:
   * `container_mask` filters DINO detections to those whose box center
     falls inside the container (caddy/drawer/shelf), which removes
     noise boxes elsewhere in the image and frees the limited
     labeled-box slots for actual sub-region candidates.
   * `container_obb` lets the script reject candidate zones that drift
     too far in XY from the container center (a common failure mode
     when the VLM's chosen mask projects to background through bad
     depth). It's also used to choose between the DINO and SAM3-text
     paths — the candidate closer to the container center wins.

   When the script returns `None` (zone not visible, low confidence,
   sanity-check failure), the downstream `compute_drop` falls back to
   the bare-container path automatically — no extra graph wiring
   needed.

1. **`compute_drop`** — `type: script`, file
   `scripts/<sg>/compute_drop_pose.py`. Use the canonical script as-is;
   do **not** emit a ``` ```python:scripts/<sg>/compute_drop_pose.py``` ```
   block — the bundle's canonical script is materialized into the
   workflow directory automatically and re-emitting it overrides the
   correct implementation with an LLM reimplementation.

   **Hard rule on parameter names:** the canonical script's `def run`
   signature is `(ctx, container_obb, container_interior_obb=None,
   ee_pose_at_grasp=None, drop_clearance=0.05, approach_height=0.20,
   held_obb=None, ...)`. Bind the held object as `held_obb`, **not**
   `target_obb`. Renaming `held_obb → target_obb` causes the runtime
   to silently drop the value (extra kwargs are warned and discarded);
   the script then falls into the no-held-geometry branch and the drop
   pose is wrong by the held object's full height.

   When the optional `perceive_zone` state is present, bind
   `container_interior_obb` to its `placement_zone_obb` output so the
   drop targets the named sub-region instead of the bare container
   center:

   ```json
   "compute_drop": {
     "type": "script",
     "script": "scripts/<sg>/compute_drop_pose.py",
     "inputs": {
       "container_obb":           Ref("in.container_obb"),
       "held_obb":                Ref("in.target_obb"),
       "ee_pose_at_grasp":        Ref("in.ee_pose_at_grasp"),
       "container_interior_obb":  Ref("perceive_zone.placement_zone_obb")
     }
   }
   ```

   When `perceive_zone` is omitted (3-state flow), drop the
   `container_interior_obb` line:

   ```json
   "compute_drop": {
     "type": "script",
     "script": "scripts/<sg>/compute_drop_pose.py",
     "inputs": {
       "container_obb":    Ref("in.container_obb"),
       "held_obb":         Ref("in.target_obb"),
       "ee_pose_at_grasp": Ref("in.ee_pose_at_grasp")
     }
   }
   ```

   `ee_pose_at_grasp` is required for the LIBERO `In(obj, region)`
   predicate to fire after release — the script uses it to convert the
   desired held-object Z into a TCP target accounting for the panda
   hand-to-tcp offset. The upstream `grasping-with-planner` subgraph
   publishes it as a cross-subgraph output (`produces_outputs.ee_pose_at_grasp`);
   this subgraph declares `ee_pose_at_grasp` in its `required_inputs`
   so the coordinator wires the binding by name.

   Returns `drop_position` (Vec3), `drop_pose` (Se3Pose), `approach_pose`
   (Se3Pose).
1a. **`drop_offset`** — `type: script`, file
    `scripts/<sg>/drop_offset_pose.py`. **REQUIRED whenever the
    upstream grasp was on a SUBPART** (e.g. frypan handle, kettle
    spout, bottle neck, tool grip). Omit ONLY when the grasp was on
    the object's geometric centroid (`parent_obb == held_obb` case).

    **HOW TO TELL: the upstream perception subgraph is
    `perceiving-object-parts`**, OR the task description names the grasp
    location explicitly ("grasp the pan by its handle", "lift it by
    the spout"). In both cases the gripper closes on a subpart and
    the parent body hangs off-axis — without `drop_offset` the
    `compute_drop` script puts the *grasp point* at the placement
    zone, which means the *body* lands off the support. This is the
    dominant failure mode for handle-grasp-then-place-ON tasks: the
    goal predicate reports `pan.bottom_z << burner.top_z` (pan
    centroid below the support) and `xy_coverage_over` < 0.5 even
    though the workflow itself returned `placed`.

    **Repair trigger.** When repairing a transport subgraph that
    failed with the above signature AND the upstream is
    `perceiving-object-parts`, **insert `drop_offset` even if a prior
    iteration omitted it**. Do not assume the absence of `drop_offset`
    in the existing workflow is intentional; it is the single most
    common omission.

    Inputs:
    `drop_pose = Ref("compute_drop.drop_pose")`,
    `ee_pose_at_grasp = Ref("in.ee_pose_at_grasp")`,
    `held_obb = Ref("in.target_obb")` (the grasped subpart OBB —
    same as compute_drop.held_obb),
    `parent_obb = Ref("in.parent_obb")` (the full object OBB — the
    coordinator must declare `parent_obb` in this subgraph's `inputs`
    and wire it from the perception subgraph's parent output, e.g.
    `perception_sg.parent_obb`).
    Returns `drop_position`, `drop_pose`, `approach_pose` — all shifted
    in XY so the parent centroid lands at the original drop XY.
    Downstream `move_above` / `release` then reference
    `drop_offset.drop_position` instead of `compute_drop.drop_position`.
2. **`move_above`** — `type: script`, file
   `scripts/<sg>/waypoint_move.py`. Inputs:
   `drop_x = Ref("compute_drop.drop_position.x")`,
   `drop_y = Ref("compute_drop.drop_position.y")`
   (or `Ref("drop_offset.drop_position.x")` / `.y` when the optional
   `drop_offset` node is present). Lifts to a safe height at the current
   XY, then moves laterally to above the drop XY.

   **Variant — collision-aware lift/translate (`waypoint_move_carve`)**.
   Same inputs (`drop_x`, `drop_y`) and same return shape, but the node
   rebuilds the world from a fresh observation and routes through
   `curobo.plan_with_grasped_object` instead of the connector's
   `robot.go_to_pose_cartesian` (which is TCP-aware but not world-aware).
   Use when a known obstacle sits on the transport path between the
   grasp pose and the drop XY (an oven door, a shelf above the table, a
   tall bottle the lift would clip). A repair pass may flip the
   transport subgraph to this variant when the `transport` stage
   pass-rate drops below the configured threshold — its hypothesis is
   "free-space transport plowed through a perceived obstacle".
3. **`release`** — `type: script`, file `scripts/<sg>/descend_release.py`.
   Inputs: `drop_position = Ref("compute_drop.drop_position")`
   (or `Ref("drop_offset.drop_position")` when `drop_offset` is present).
   Descends, opens the gripper, retracts home. On success → exit `placed`;
   on failure → exit `blocked`.

   **Variant — linear descent (`descend_release_linear`)**. Same
   node-level contract, but the descent goes through the connector's
   TCP-aware `robot.go_to_pose_cartesian` (cuRobo linear plan with a
   `plan_to_pose` fallback built into the backend), so the held object
   descends on a straight Cartesian line with the orientation held — the
   cleanest release dynamics for subpart-grasp + place-ON tasks (frypan
   handle → stove).

## Required end states

| End state | Meaning |
|---|---|
| `placed` | Object released at the destination. Route to the next subgraph or to `done`. |
| `blocked` | Path blocked or motion failed; aborted before release. Coordinator routes to abort. |


## See also

- `references/clearance_constants.md` — the magic numbers.
- `references/design_transport.md` — why no planner is needed for the
  default scope.
- `scripts/{compute_drop_pose,drop_offset_pose,waypoint_move,waypoint_move_carve,descend_release,descend_release_linear,approach_above,lift_grasped,perceive_placement_zone}.py`
  — canonical scripts.
- `prompts/vlm_select_zone.md` — VLM prompt template for the optional
  `perceive_zone` state.
