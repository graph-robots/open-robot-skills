---
name: perceiving-objects-oneshot
description: >
  Lightweight one-shot 3D object perception. Runs Grounding-DINO broad
  detection, a SINGLE VLM set-of-marks letter pick over the labeled
  boxes, SAM3 box segmentation, depth back-projection, and
  geometry.filter_and_compute_obb. No pairwise tournament, no multi-view
  safe-gate. Returns a clean not_found output (no exception) when the
  VLM answers "none" or DINO emits no detections — making this the right
  skill for clean-all-items loops whose natural termination signal is
  "no more matching objects in view". Use when a multi-item loop needs a
  clean no-match exit, or for generic target descriptions on uncluttered
  scenes with reasonably sized targets.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, dino, vlm, one-shot, set-of-marks]
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - vlm.query
    - sam3.segment_box
    - geometry.mask_to_world_points
    - geometry.filter_and_compute_obb
  exit_conditions:
    found: Target detected; OBB and mask bound in subgraph outputs.
    not_found: VLM replied "none" or DINO emitted no detections. In clean-all-items loops route to done; in normal pick-and-place route to abort.
  produces_outputs:
    "<name>_obb": OrientedBoundingBox
    "<name>_mask": Mask
    "<name>_cloud": PointCloud
  errors:
    - "NOT_FOUND: No detection matched the target description."
  hard_rules:
    - perception_pipeline_invariants.md#emit-both-obb-and-mask
    - geometry_calling_conventions.md#obb-field-binding
  canonical_scripts:
    - perceive_simple: scripts/perceive_simple.py
  prompts:
    vlm_one_shot: prompts/vlm_one_shot.md
  references:
    - title: Perception pipeline invariants (emit obb + mask + cloud)
      path: references/perception_pipeline_invariants.md
    - title: Geometry tool calling conventions (output field binding)
      path: references/geometry_calling_conventions.md
  examples:
    - title: Canonical perception subgraph (observe → perceive → filter_obb)
      path: examples/canonical_subgraph.json
  streaming: false
---

# perceiving-objects-oneshot

Single VLM call over a set-of-marks overlay. Pipeline:

```text
observe → perceive → filter_obb
```

`perceive` runs:

1. ``grounding-dino.detect`` with a broad ``object.`` text prompt
2. One ``vlm.query`` showing the image with letter-labeled boxes:
   "Which letter is the *<target>*? Reply with one letter or 'none'."
3. On ``none``: emit ``found: False`` so the subgraph exits
   ``not_found``. On a letter: ``sam3.segment_box`` on the chosen box,
   ``geometry.mask_to_world_points`` for the cloud.

## When to use

- Clean-all-items / multi-item loops where the cycle needs a clean
  "no match" signal to terminate via ``target.not_found → done``.
- Tasks where the target description is generic ("any item on the
  floor", "the next remaining grocery item") rather than a specific
  scene-spec id.
- Uncluttered scenes with distinct, reasonably sized targets where the
  set-of-marks letter pick is reliable.

## When NOT to use

- Small / cluttered targets (< 40 px wide) — prefer ``perceiving-objects``
  whose pairwise crop tournament is far more reliable in that regime.

## Recommended subgraph state flow

3 states: ``observe → perceive → filter_obb`` (mirrors ``perceiving-objects``).

State details:

> **About `object_name` below:** it is a literal Python string — the
> natural noun phrase for the object you are perceiving, drawn from this
> subgraph's description (e.g. `"alphabet soup"`, `"basket"`,
> `"any grocery item on the floor"`). It is a constant per subgraph
> instance, NOT a binding. **DO NOT** write `Ref("in.object_name")` or
> any other `Ref(...)`; the coordinator does not declare `object_name`
> as a subgraph input. Write the string directly,
> e.g. `"object_name": "any grocery item on the floor"`.
> The same rule applies to `object_description` if you set it.

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`,
   `inputs: {}`. Connector tool; flat name only.
2. **`perceive`** — `type: script`, file `scripts/<sg>/perceive_simple.py`
   from this bundle. Inputs:
   `cameras=Ref("observe.cameras")`,
   `object_name="<noun phrase from the subgraph description>"`,
   plus any optional literals (`object_description`, `dino_prompt`).
   Returns `{found, cloud, mask, score}`. When the VLM picks "none" or
   DINO emits no detections, ``found`` is `False` and the downstream
   `filter_obb` step then raises (empty cloud) — caught by the
   subgraph's `on_error: "not_found"` exit.
3. **`filter_obb`** — `type: tool`,
   `tool: "geometry.filter_and_compute_obb"`,
   `inputs={"points": Ref("perceive.cloud")}`. Returns
   `{"obb": <OrientedBoundingBox>}`.

### Wiring the exit (HARD)

Linear `perceive → filter_obb → found → END`. The `filter_obb` tool
raises on empty clouds (the not-found path), and the subgraph's
`on_error: "not_found"` catches that. Do NOT add conditional edges on
`perceive` — the linear path plus `set_on_error` is sufficient.

```python
sg.add_node("filter_obb", type="tool",
            tool="geometry.filter_and_compute_obb",
            inputs={"points": Ref("perceive.cloud")})
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

Note that `geometry.filter_and_compute_obb` returns `{"obb": ...}`, so
the OBB binding walks into the `obb` field (`Ref("filter_obb.obb")`,
NOT a bare `Ref("filter_obb")`). See
`references/geometry_calling_conventions.md`.
