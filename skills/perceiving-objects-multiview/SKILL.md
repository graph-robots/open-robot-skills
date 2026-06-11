---
name: perceiving-objects-multiview
description: >
  Robust three-method 3D object perception. Bare Grounding-DINO, a Molmo
  point-prompt pipeline, and DINO+VLM box selection all run on the same
  observation; a VLM disambiguator picks the best mask when the methods
  disagree, and the winning cloud is passed through
  geometry.filter_and_compute_obb for the OBB. The Molmo point-prompt
  path needs a self-hosted Molmo vLLM endpoint (GAP_MOLMO_BASE_URL);
  without one the merge still works on the two DINO-based detectors, and
  gemini-er.detect is the hosted-API detection alternative. Runs three
  detectors per call so it is slower than perceiving-objects, and it has
  no clean not_found loop signal. Use when single-shot pick-and-place on
  a cluttered scene needs maximum robustness, one detector alone has
  proven unreliable, and the extra latency is acceptable.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, multi-method, robust, dino, vlm, molmo, sam3]
gap:
  allowed_tools:
    - robot.get_observation
    - molmo.point_prompt
    - grounding-dino.detect
    - vlm.query
    - sam3.segment_box
    - sam3.segment_point
    - sam3.segment_text
    - geometry.mask_to_world_points
    - geometry.filter_and_compute_obb
  exit_conditions:
    found: Target detected; OBB and mask bound in subgraph outputs.
    not_found: Target not visible after exhausting all three methods. Coordinator routes to abort.
  produces_outputs:
    "<name>_obb": OrientedBoundingBox
    "<name>_mask": Mask
  errors:
    - "NOT_FOUND: Object not detected by any of the three perception methods."
  hard_rules:
    - perception_pipeline_invariants.md#emit-both-obb-and-mask
    - geometry_calling_conventions.md#obb-field-binding
  canonical_scripts:
    - perceive_dino: scripts/perceive_dino.py
    - perceive_point: scripts/perceive_point.py
    - perceive_dino_vlm: scripts/perceive_dino_vlm.py
    - merge: scripts/merge.py
    - select_best: scripts/select_best.py
  prompts:
    vlm_select_box: prompts/vlm_select_box.md
    vlm_select_best: prompts/vlm_select_best.md
  references:
    - title: When to pick perceiving-objects vs perceiving-objects-multiview
      path: references/single_vs_multi.md
    - title: How the multi-method merge works
      path: references/design_multi_method.md
    - title: Perception pipeline invariants (emit obb + mask + cloud)
      path: references/perception_pipeline_invariants.md
    - title: Geometry tool calling conventions (output field binding)
      path: references/geometry_calling_conventions.md
  examples:
    - title: Canonical perception subgraph (three detectors + merge)
      path: examples/canonical_subgraph.json
  streaming: false
---

# perceiving-objects-multiview

Three perception pipelines run in sequence on the same observation:

1. **`perceive_dino`** — Molmo point-prompt + GroundingDINO box selection +
   SAM3 box-prompted segmentation.
2. **`perceive_point`** — Molmo point-prompt + SAM3 point-prompted (text
   fallback).
3. **`perceive_dino_vlm`** — broad GroundingDINO detect + VLM box selection +
   SAM3 box-prompted segmentation.

`merge` then collects every method that found the target, drops
geometrically-implausible thin-face masks, and — if more than one
candidate survives — runs a VLM disambiguator (`select_best.py`) to pick
the best mask. The winning cloud is passed through
`geometry.filter_and_compute_obb` so `merge` emits an already-filtered
OBB. If **no** method found the target, `merge` raises
`PerceptionFailed` and the subgraph's `on_error: "not_found"` catches
it. This redundancy is exactly why the hand-validated
`graph_cartesian_obb` workflow uses this skill for its target subgraph.

## Model dependency: Molmo OR Gemini-ER

The point-prompt pathway (`perceive_point`, and the box selection in
`perceive_dino`) calls `molmo.point_prompt`, which requires a
**self-hosted Molmo vLLM endpoint** configured via `GAP_MOLMO_BASE_URL`
(Molmo has no hosted API — see the molmo tool bundle's SKILL.md for the
`vllm serve` recipe). Two degradation options:

- **No Molmo endpoint at all:** every `molmo.point_prompt` call raises;
  the scripts catch the error and fall back to their DINO-box /
  SAM3-text paths, so the skill keeps working with one fewer redundant
  detector. Nothing to configure.
- **API-only platforms:** `gemini-er.detect` (the gemini-er tool
  bundle, hosted Gemini Robotics-ER, zero GPU) is the API alternative
  for open-vocabulary 2D detection — it returns
  `{detections: [{box, label, score}]}` boxes that feed
  `sam3.segment_box` exactly like the DINO boxes do. Swap it in for the
  DINO detect call when authoring a variant subgraph for a platform
  with no local GPU detector.

## When to use

- Single-shot pick-and-place where one detector alone has proven
  unreliable and the extra cost of running three detectors is
  acceptable.

## When NOT to use

- Clean-all-items / multi-item loops — use `perceiving-objects-oneshot`,
  which has the clean `not_found` termination this skill lacks.
- Latency-sensitive paths — three detectors run per call (~5–8 s vs
  ~2–4 s for `perceiving-objects`).

## Recommended subgraph state flow

5 states (no separate `filter_obb` step — `merge` already returns a
filtered OBB):

```text
observe → perceive_dino → perceive_point → perceive_dino_vlm → merge → found
```

State details:

> **About `object_name` below:** it is a literal Python string — the natural
> noun phrase for the object you are perceiving, drawn from this subgraph's
> description (e.g. `"alphabet soup can"`, `"basket"`, `"red bowl"`). It is
> a constant per subgraph instance, NOT a binding. **DO NOT** write
> `Ref("in.object_name")` or any other `Ref(...)`; the coordinator does
> not declare `object_name` as a subgraph input. Write the string directly,
> e.g. `"object_name": "basket"`. Use the **same** literal string for all
> four script nodes below.

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`,
   `inputs: {}`. Connector tool; flat name only. Do NOT
   write `type: service` — the validator rejects it.
2. **`perceive_dino`** — `type: script`, file
   `scripts/<sg>/perceive_dino.py` from this bundle's canonical_scripts.
   Inputs: `cameras = Ref("observe.cameras")`,
   `object_name = "basket"` (replace with the actual target noun phrase
   from this subgraph's description). Returns `{found, cloud, mask, score}`.
3. **`perceive_point`** — `type: script`, file
   `scripts/<sg>/perceive_point.py`. Same input shape (use the **same**
   literal `object_name` string as step 2), same output shape.
4. **`perceive_dino_vlm`** — `type: script`, file
   `scripts/<sg>/perceive_dino_vlm.py`. Same input shape (same literal
   `object_name`), same output shape.
5. **`merge`** — `type: script`, file `scripts/<sg>/merge.py`. Inputs:
   `cameras = Ref("observe.cameras")`,
   `object_name = "basket"` (same literal string as steps 2–4),
   `dino_result = Ref("perceive_dino")`,
   `point_result = Ref("perceive_point")`,
   `vlm_result = Ref("perceive_dino_vlm")`.
   Returns **only** `{cloud, mask, obb}` — there is **no `found` field
   in merge's output**. If merge cannot find the target it raises;
   the subgraph's `on_error: "not_found"` catches the raise.

### Wiring the exit (HARD)

Use the linear edge `merge → found → END`. Do **NOT** add any
`conditional_edges` entry on `merge` — `merge` does not return a
`found` field, so a router on it raises "cannot read field 'found'"
and the subgraph routes to on_error every time. The linear path plus
`set_on_error` is sufficient.

✅ Correct (the literal `gap.builder` calls you should emit):

```python
sg.add_node("merge", type="script", script="scripts/merge.py",
            inputs={"cameras": Ref("observe.cameras"),
                    "object_name": "basket",
                    "dino_result": Ref("perceive_dino"),
                    "point_result": Ref("perceive_point"),
                    "vlm_result": Ref("perceive_dino_vlm")})

# add_exit() creates the success-marker noop node AND registers the
# exit value. Do NOT also call sg.add_node("found", type="noop") — that
# would conflict with the node add_exit created.
sg.add_exit("found")

sg.add_edge("perceive_dino_vlm", "merge")
sg.add_edge("merge", "found")
sg.add_edge("found", END)

sg.set_on_error("not_found")
```

❌ Wrong — runtime raises "cannot read field 'found' from output of
type dict" because `merge` doesn't return that field; the subgraph
then routes to on_error every time:

```python
sg.add_conditional_edges("merge", router_field="found",
                         mapping={"true": "found"})
```

Bind the subgraph outputs (BOTH required — this exactly matches the
hand-validated `graph_cartesian_obb` target subgraph):

```python
sg.set_outputs(
    target_obb=Ref("merge.obb"),
    target_mask=Ref("merge.mask"),
)
```

(Replace `target_*` with this subgraph's actual name prefix — e.g.
`container_obb`, `container_mask` when authoring the container
subgraph.)

`merge` also returns a fused world-frame `cloud`; if a downstream
cloud-consuming grasp skill (e.g. a learned-grasp skill) declares a
`<name>_cloud` input, additionally bind
`target_cloud=Ref("merge.cloud")`. Omit it otherwise — the curobo OBB
grasp skill (the `graph_cartesian_obb` grasp skill) only needs
`<name>_obb` + `<name>_mask`.

## Hard rules

1. Subgraph-level outputs MUST emit BOTH `<name>_obb` AND `<name>_mask`.
   See `references/perception_pipeline_invariants.md`.
2. `merge.obb` is the OBB output — it is *already* filtered through
   `geometry.filter_and_compute_obb` inside `merge.py`, so there is
   **no** separate `filter_obb` tool node in this skill (unlike
   `perceiving-objects`). Bind via `Ref("merge.obb")` (no extra
   trailing field).
3. Emit all four script nodes (`perceive_dino`, `perceive_point`,
   `perceive_dino_vlm`, `merge`). Dropping any detector node defeats the
   redundancy that makes this skill robust — that degenerate single-path
   graph is what `perceiving-objects` is for.

## Required end states

| End state | Meaning |
|---|---|
| `found` | OBB + mask bound; route to next subgraph (typically a grasp skill). |
| `not_found` | Route to `abort`. |

## See also

- `references/design_multi_method.md` — why three methods beat one.
- `references/single_vs_multi.md` — choosing single- vs multi-method.
- `prompts/{vlm_select_box,vlm_select_best}.md` — VLM templates.
- `scripts/{perceive_dino,perceive_point,perceive_dino_vlm,merge,select_best}.py`
  — canonical scripts.
