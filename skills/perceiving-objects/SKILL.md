---
name: perceiving-objects
description: >
  Fast single-path 3D object perception. Runs Grounding-DINO broad
  detection, a pairwise VLM crop tournament to identify the target box,
  SAM3 box segmentation, and depth back-projection to a world-frame point
  cloud, finished by geometry.filter_and_compute_obb for a clean oriented
  bounding box, mask, and cloud. Multi-camera rigs get a verified
  wrist-camera fallback gate. Use when a manipulation workflow needs to
  localize one named object quickly — uncluttered scenes with visually
  distinct targets, latency-bound loops, or platforms deploying only
  DINO + VLM + SAM3 + geometry.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, dino, vlm, sam3, single-method]
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
    found: Target detected; OBB and mask bound in subgraph outputs.
    not_found: Target not visible in any view. In clean-all-items loops route to done; in normal pick-and-place, route to abort.
  produces_outputs:
    "<name>_obb": OrientedBoundingBox
    "<name>_mask": Mask
    "<name>_cloud": PointCloud
  errors:
    - "NOT_FOUND: Object not detected in any camera view."
  hard_rules:
    - perception_pipeline_invariants.md#emit-both-obb-and-mask
    - geometry_calling_conventions.md#obb-field-binding
  canonical_scripts:
    - perceive_dino_vlm: scripts/perceive_dino_vlm.py
  prompts:
    vlm_pairwise: prompts/vlm_pairwise.md
  references:
    - title: When to pick perceiving-objects vs perceiving-objects-multiview
      path: references/single_vs_multi.md
    - title: Perception pipeline invariants (emit obb + mask + cloud)
      path: references/perception_pipeline_invariants.md
    - title: Geometry tool calling conventions (output field binding)
      path: references/geometry_calling_conventions.md
  examples:
    - title: Canonical perception subgraph (observe → perceive → filter_obb)
      path: examples/canonical_subgraph.json
  streaming: false
---

# perceiving-objects

Single-path perception: detect → disambiguate (pairwise crop tournament)
→ segment → fuse to 3D → extract OBB. Each DINO detection is cropped and
upscaled, and the target is found via binary "A or B?" comparisons of
crop pairs — far more reliable on small targets than a one-shot
Set-of-Marks letter pick (~30% → 97% on the LIBERO-PosVar object-ID
study).

Multi-camera handling uses a **`safe` wrist-fallback gate** (not blind
KD-tree fusion): identification defaults to the exterior view; the
wrist (eye-in-hand) view is consulted ONLY when the exterior pick fails
its own close-up verify AND the wrist pick passes its own. On the
4-suite / 200-frame regression study this was the only zero-regression
policy (+2.5% net, 0/189 frames regressed; blind fuse/verify→wrist/
wrist-only all regressed). See `perceive_dino_vlm.run` docstring.

On the verified-exterior path the wrist views still contribute **cloud
geometry** (never identity): wrist clouds of the same object — gated by
the multiview intersection check, with a geometry-seeded SAM fallback
(exterior cloud projected into the wrist frame) — are fused into the
output cloud so the OBB recovers the top face / far side a single front
view misses. A lone front view yields a sliver OBB biased toward the
camera by half the object depth, and that off-centre pinch is the
measured slip-during-transport failure mode on tall bottles/cartons.

## When to use

- Uncluttered scenes with visually distinct targets.
- Platforms where only DINO + VLM + SAM3 + geometry are deployed.
- When `perceiving-objects-multiview` is not in the available skill catalog.

## When NOT to use

- Cluttered scenes with similar nearby distractors. Prefer
  `perceiving-objects-multiview`.
- Clean-all-items / multi-item loops that need a clean "no match" loop
  terminator. Prefer `perceiving-objects-oneshot`.

## Recommended subgraph state flow

3 states:

```text
observe → perceive → filter_obb
```

State details:

> **About `object_name` below:** it is a literal Python string — the natural
> noun phrase for the object you are perceiving, drawn from this subgraph's
> description (e.g. `"alphabet soup can"`, `"basket"`, `"red bowl"`). It is
> a constant per subgraph instance, NOT a binding. **DO NOT** write
> `Ref("in.object_name")` or any other `Ref(...)`; the coordinator does
> not declare `object_name` as a subgraph input. Write the string directly,
> e.g. `"object_name": "basket"`.
>
> **About `object_description` (wire it whenever the task gives hints):**
> also a literal Python string. When the task/workflow description carries
> shape or appearance hints for the target (e.g. an "Object context" block
> with `shape_hint` / `expected_label`, or adjectives in the instruction),
> pass them through verbatim, e.g.
> `"object_description": "small rectangular box, blue and white packaging,
> ~5 cm wide"`. The description is injected into BOTH the pairwise
> tournament prompt ("It looks like: …") and the close-up verification
> question ("It should look like: …"). This is what disambiguates
> look-alike packaging (several LIBERO grocery items are small blue/white
> boxes) and keeps the verify gate from rejecting a correct pick whose
> rendered asset reads as a generic box — a rejection forces the
> wrist-camera fallback, whose single top-down view degrades the OBB
> height and downstream grasps. Omit it (default `""`) only when the task
> provides no hints.

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`,
   `inputs: {}`. Connector tool; flat name only.
2. **`perceive`** — `type: script`, file `scripts/<sg>/perceive_dino_vlm.py`
   from this bundle. Inputs:
   `cameras=Ref("observe.cameras")`,
   `object_name="basket"` (replace with the actual target noun phrase
   from this subgraph's description),
   `object_description="..."` (the task's shape/appearance hints — see the
   note above; strongly recommended whenever hints exist),
   plus any optional fields (`dino_prompt`, etc.).
   Returns `{found, cloud, mask, score}`.
3. **`filter_obb`** — `type: tool`,
   `tool: "geometry.filter_and_compute_obb"`,
   `inputs={"points": Ref("perceive.cloud")}`. Returns `{"obb": <OrientedBoundingBox>}`.

### Wiring the exit (HARD)

Use the linear edge `filter_obb → found → END`. When the target isn't
found, `perceive` returns an empty cloud and `filter_obb` raises on it,
so the subgraph's `on_error: "not_found"` catches that path
automatically. Do **NOT** add any conditional edges on `perceive` — the
linear path plus `set_on_error` is sufficient.

✅ Correct (the literal `gap.builder` calls you should emit):

```python
sg.add_node("perceive", type="script",
            script="scripts/<sg>/perceive_dino_vlm.py",
            inputs={"cameras": Ref("observe.cameras"),
                    "object_name": "small blue and white cream cheese",
                    # from the task's Object context / shape_hint block:
                    "object_description": ("small rectangular box, blue "
                                           "and white packaging, ~5 cm wide")})

sg.add_node("filter_obb", type="tool",
            tool="geometry.filter_and_compute_obb",
            inputs={"points": Ref("perceive.cloud")})

# add_exit() creates the success-marker noop node AND registers the
# exit value. Do NOT also call sg.add_node("found", type="noop") — that
# would conflict with the node add_exit created.
sg.add_exit("found")

sg.add_edge("perceive", "filter_obb")
sg.add_edge("filter_obb", "found")
sg.add_edge("found", END)

sg.set_on_error("not_found")
```

Bind the subgraph outputs (ALL THREE — required, no exceptions):

```python
sg.set_outputs(
    target_obb=Ref("filter_obb.obb"),
    target_mask=Ref("perceive.mask"),
    target_cloud=Ref("perceive.cloud"),
)
```

(Replace `target_*` with this subgraph's actual name prefix — e.g.
`container_obb`, `container_mask`, `container_cloud` when authoring
the container subgraph.)

All three bindings walk into a field of the producing node's output
dict: `geometry.filter_and_compute_obb` returns `{"obb": ...}` (bind
`Ref("filter_obb.obb")`, NOT a bare `Ref("filter_obb")`), while
`<name>_mask` and `<name>_cloud` walk into fields of `perceive`'s
output dict. `perceive_dino_vlm.py` already produces all three;
emitting them unconditionally lets downstream subgraphs that need any
of them (e.g. learned-grasp skills require `<name>_cloud`) wire up
without you having to anticipate which skill they'll use.

## Hard rules

1. Subgraph-level outputs MUST emit ALL THREE: `<name>_obb`,
   `<name>_mask`, AND `<name>_cloud`. The cloud is the fused world-frame
   point cloud needed by learned-grasp skills; emit it unconditionally
   so the downstream agent can wire it without round-tripping. See
   `references/perception_pipeline_invariants.md`.
2. `geometry.filter_and_compute_obb` returns `{"obb": OrientedBoundingBox}`;
   bind via `Ref("filter_obb.obb")` (walk into the `obb` field).
   See `references/geometry_calling_conventions.md`.

## Required end states

| End state | Meaning |
|---|---|
| `found` | OBB + mask bound; route to next subgraph (typically a grasp skill). |
| `not_found` | Route to `abort` (or to `done` in clean-all-items loops). |


## See also

- `references/single_vs_multi.md` — the choice between single- and multi-
  method perception.
- `prompts/vlm_pairwise.md` — the pairwise-tournament VLM prompt template.
- `scripts/perceive_dino_vlm.py` — the canonical perception script.
