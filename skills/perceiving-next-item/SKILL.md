---
name: perceiving-next-item
description: >
  Loop-head perception for pack-all / clean-all-items tasks. From ONE
  observation it localizes BOTH the destination container (basket, bin, box)
  AND the next remaining target item, using the pairwise VLM crop tournament
  with a container-excluding description so the target is never confused with
  the basket, then makes a clean found / none decision: an item was found
  (grasp it) or only the container remains (all items packed — exit the loop
  to done). Use when a workflow must pick up EVERY object and place each into a
  container in a loop (pack-all / clean-all-items) and each pass must reliably
  answer "is there still a graspable item, or are we done?" while also exposing
  a fresh container OBB for the downstream place. This subgraph is
  self-contained and takes NO inputs (inputs: {}); the container and item
  phrases are literal strings written inside the perception nodes, never
  subgraph parameters — do not declare item_name / container_name /
  item_description as subgraph inputs.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, dino, vlm, sam3, loop, pack-all, clean-all, container]
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - vlm.query
    - vlm.query_yes_no
    - sam3.segment_box
    - sam3.segment_text
    - geometry.mask_to_world_points
    - geometry.exclude_robot_points
    - geometry.filter_and_compute_obb
  exit_conditions:
    found: A graspable target item distinct from the container was localized; target_obb/mask and container_obb/mask/cloud bound in the subgraph outputs. Route to a grasp skill.
    none: No target item remains — only the container is left, so every item is packed. This is the clean loop terminator; route to done (NOT abort — it is a success).
    perception_failed: Perception raised an actual error (empty detection is NOT this — that is the normal `none` path). Route to abort. Named so it is never confused with "nothing left"; do NOT route it to done.
  produces_outputs:
    target_obb: OrientedBoundingBox
    target_mask: Mask
    container_obb: OrientedBoundingBox
    container_mask: Mask
    container_cloud: PointCloud
  hard_rules:
    - perception_pipeline_invariants.md#emit-both-obb-and-mask
    - geometry_calling_conventions.md#obb-field-binding
    - >
      The item perception MUST pass an `object_description` that names the
      grocery item categories AND explicitly excludes the container (e.g.
      "a packaged grocery product such as a can, box, carton, jar, or bottle.
      Never the wicker basket or storage container."). This exclusion is what
      makes the tournament return "no item" once only the basket remains — it
      is the whole basis of the clean `none` loop-exit. Do NOT rely on
      geometry to reject the basket.
    - >
      This subgraph declares NO inputs (`inputs: {}`) and is fully
      self-contained. Every value that tells the perceive nodes WHAT to look
      for is a LITERAL string hardcoded in the node — use exactly the node input
      names `object_name` and `object_description` from
      examples/canonical_subgraph.json (do not rename them to `item_name`,
      `container_name`, `item_description`, etc.). DO NOT lift ANY of them to a
      subgraph input and DO NOT write `Ref("in.<anything>")` for them: there is
      no upstream producer, so it fails validation (W8: no upstream producer).
      The basket and item names are known at authoring time — write them in.
    - >
      The `decide` router (scripts/<sg>/decide_next_item.py) sits BETWEEN
      `perceive_item` and `filter_obb_item`, and maps `found -> filter_obb_item`,
      `none -> none`. Declare `success_values: ["found", "none"]` so BOTH are
      clean exits; only genuine perception errors fall through to
      `on_error: perception_failed`.
    - >
      TOP-LEVEL ROUTING (do not get this wrong): map this subgraph's
      `found -> grasp`, `none -> done` (all items packed — a SUCCESS), and
      `perception_failed -> abort`. `none` is the loop terminator, NOT
      `perception_failed`; never route `perception_failed` to done.
    - >
      Perceive the item from the raw `observe.cameras`, but perceive the
      container through `exterior_view` (drop the wrist cam) — the angled
      eye-in-hand view bloats the container OBB by fusing neighbouring points.
    - >
      WIRE A SINGLE CHAIN, never parallel branches: observe → exterior_view →
      perceive_container → filter_obb_container → perceive_item → decide.
      Do NOT add a second edge from `observe` straight to `perceive_item` to
      "parallelize" the two perceptions: the runtime scheduler has no join
      barrier, so `decide` (which consumes `filter_obb_container.obb`) can be
      scheduled by `perceive_item`'s completion before `filter_obb_container`
      has run — the $ref fails at runtime and the subgraph aborts on every
      trial (validator rule S12 rejects this shape).
    - >
      RESTRICTED ITEM SETS: when the task's target set is restricted — a
      SUBSET ("pack the milk and the tomato sauce…") OR an EXCLUSION ("pack
      everything except the milk…") — i.e. whenever some visible object must
      NOT be packed, the `perceive_item` node MUST also pass
      `reject_unverified: true` (a boolean node input, next to
      object_name/object_description), and its `object_name`/
      `object_description` literals MUST spell out the allowed set and every
      exclusion (excluded items AND the container). Why: the default perceive
      favors recall — an unverified best-guess pick is kept so a pack-ALL
      loop never stops early — but under a restricted query that same
      fallback returns EXCLUDED items: a subset loop packs things the task
      forbade, and an exclusion loop ends by grasping at the excluded item
      and aborting instead of exiting `none`. `reject_unverified: true`
      flips the gate to precision: no verified allowed-item ⇒ `found=false`
      ⇒ the loop exits `none` exactly when only excluded objects remain.
      Set it ONLY for restricted sets — for true pack-all tasks ("all
      objects", "every item") leave it unset; recall mode is what keeps
      those loops from terminating early. The allowed set must be a
      CONCRETE CATEGORY, never a tautology: write "a packaged grocery
      product such as a can, box, carton, jar, or bottle — never the milk
      carton, and never the wicker basket", NOT "any object except the
      milk" — "any object" is satisfied by every spurious detection
      (table patch, shadow, robot part), so the precision gate can never
      reject the leftovers and the loop cannot exit.
  canonical_scripts:
    - perceive_dino_vlm: scripts/perceive_dino_vlm.py
    - exterior_view: scripts/exterior_view.py
    - decide_next_item: scripts/decide_next_item.py
  prompts:
    vlm_pairwise: prompts/vlm_pairwise.md
  references:
    - title: Perception pipeline invariants (emit obb + mask + cloud)
      path: references/perception_pipeline_invariants.md
    - title: Geometry tool calling conventions (output field binding)
      path: references/geometry_calling_conventions.md
  examples:
    - title: Canonical loop-head subgraph (observe → container + item → decide)
      path: examples/canonical_subgraph.json
  streaming: false
---

# perceiving-next-item

The repeating **head of a pack-all loop**. One `robot.get_observation` feeds two
perceptions — the destination **container** and the **next remaining item** —
and a `decide` router turns the item verdict into the loop's continue/terminate
signal. It is `perceiving-objects` (the reliable pairwise-tournament perception)
composed with (a) a second perception for the container and (b) a clean `none`
exit, so a "pick up every object and place it in the basket" workflow has one
subgraph that answers *"is there still an item, and where is the basket?"* every
pass.

Detection uses the same pairwise VLM crop tournament as `perceiving-objects`
(~30% → 97% object-ID over a one-shot Set-of-Marks pick on the LIBERO-PosVar
study). The clean loop terminator comes from the item's **container-excluding
`object_description`**: once only the basket remains, the tournament + verify
gate answer "no grocery item", `perceive_item` returns `found=False`, and the
`decide` router emits `none` → the loop exits to `done`.

## When to use

- The repeating head of a **pack-all / clean-all-items loop** (`transport`
  routes its success edge back here), where each pass must reliably decide
  "grasp the next item" vs "everything is packed, stop".
- When the downstream place needs a **fresh container OBB** every pass (this
  skill emits `container_obb`/`container_mask`/`container_cloud` alongside the
  target).

## When NOT to use

- **Single pick-and-place** (grab ONE named object). Use `perceiving-objects`
  (one target, no loop, no container co-perception).
- **You want the container localized once, out of the loop.** If the container
  never moves and you prefer to perceive it a single time before the loop, use a
  plain `perceiving-objects` subgraph for the basket + `perceiving-objects-oneshot`
  for the looping target. This skill deliberately re-localizes both each pass
  (robust to a nudged basket, one observation, one clean `none`).
- **Cluttered scenes with look-alike distractors.** Prefer
  `perceiving-objects` for the target identity — its pairwise-tournament plus
  `object_description` hints disambiguate look-alikes (it lacks the clean
  loop `none`, so you would add your own `decide`).

### vs. `perceiving-objects-oneshot`

`oneshot` also has a clean `not_found` loop terminator, but it identifies with a
**single Set-of-Marks VLM pick** (weaker on small/similar items) and perceives
**only the target** (no container). This skill uses the **pairwise tournament**
and **co-localizes the container**, so both the "is anything left?" decision and
the downstream place are more reliable — at the cost of a second perception per
pass.

## Recommended subgraph state flow

```text
observe → exterior_view → perceive_container → filter_obb_container
        → perceive_item → decide ──found──▶ filter_obb_item ──▶ found
                                └──none──▶ none
```

See `examples/canonical_subgraph.json` for the exact node/edge/output wiring —
emit it verbatim, adapting only the `object_name`/`object_description` literals
to the task's items and container.

State details:

1. **`observe`** — `type: tool`, `robot.get_observation`, `inputs: {}`.
2. **`exterior_view`** — `type: script`, `scripts/<sg>/exterior_view.py`,
   `inputs: {cameras: Ref("observe.cameras")}`. Drops the wrist cam so the
   container OBB is not bloated by the angled eye-in-hand view.
3. **`perceive_container`** — `type: script`, `scripts/<sg>/perceive_dino_vlm.py`,
   `inputs: {cameras: Ref("exterior_view.cameras"), object_name: "basket"}`.
4. **`filter_obb_container`** — `type: tool`, `geometry.filter_and_compute_obb`,
   `inputs: {points: Ref("perceive_container.cloud")}`.
5. **`perceive_item`** — `type: script`, `scripts/<sg>/perceive_dino_vlm.py`,
   `inputs: {cameras: Ref("observe.cameras"), object_name: "grocery item",
   object_description: "…never the wicker basket…"}` (the exclusion is a HARD
   rule above). For SUBSET tasks (specific items only), add
   `reject_unverified: True` and name exactly the allowed items in the
   literals — see the subset-scoping HARD rule.
6. **`decide`** — `type: router`, `scripts/<sg>/decide_next_item.py`,
   `inputs: {found: Ref("perceive_item.found"), cloud: Ref("perceive_item.cloud"),
   container_obb: Ref("filter_obb_container.obb")}`. Maps
   `found → filter_obb_item`, `none → none`.
7. **`filter_obb_item`** — `type: tool`, `geometry.filter_and_compute_obb`,
   `inputs: {points: Ref("perceive_item.cloud")}`.

### Wiring the exit (HARD)

```python
sg.add_exit("found")
sg.add_exit("none")
sg.set_exit(success_values=["found", "none"])   # BOTH are clean exits
sg.set_on_error("perception_failed")

sg.set_outputs(
    target_obb=Ref("filter_obb_item.obb"),
    target_mask=Ref("perceive_item.mask"),
    container_obb=Ref("filter_obb_container.obb"),
    container_mask=Ref("perceive_container.mask"),
    container_cloud=Ref("perceive_container.cloud"),
)
```

At the top level, route this subgraph's `found → grasp`, `none → done`,
`perception_failed → abort`.

## Required end states

| End state | Meaning |
|---|---|
| `found` | A graspable item distinct from the container was localized. Route to a grasp skill. |
| `none` | Only the container remains — all items packed. Route to `done` (a **success**, not an abort). |
| `perception_failed` | Unexpected perception error (not "nothing left" — that is `none`). Route to `abort`. Lives only in `on_error`. |

## See also

- `scripts/perceive_dino_vlm.py` — the pairwise-tournament perception (shared with `perceiving-objects`).
- `scripts/decide_next_item.py` — the unprivileged loop router: per-pass VLM
  all-packed check (primary stop) + same-target no-progress guard + perception
  verdict; `sim.check_success` is telemetry only (`GAP_DECIDE_TRUST_ENV=1` opts
  into trusting it on benchmarks).
- `scripts/exterior_view.py` — wrist-cam drop for a clean container OBB.
- `prompts/vlm_pairwise.md` — the pairwise-tournament VLM prompt template.
