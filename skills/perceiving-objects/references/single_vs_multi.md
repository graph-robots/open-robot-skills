# Single-method vs multi-method perception

`perceiving-objects` and `perceiving-objects-multiview` solve the same
problem — "where is object X in 3D" — with different cost/robustness
profiles.

## `perceiving-objects`

One pipeline: DINO broad-detect → VLM picks the right box → SAM3 segments →
geometry fuses depth → geometry.filter_and_compute_obb.

- **Latency:** ~2–4 s per call (one DINO inference, one VLM call, one SAM3
  call, one geometry call).
- **Robustness:** good when the target is visually distinct in a clean
  scene.
- **Failure mode:** if DINO misses the target (small, occluded, similar
  background), the VLM has no candidate box to pick. The skill falls back
  to SAM3's text-prompt mode (one extra SAM3 call), but that is itself
  fragile in clutter.

## `perceiving-objects-multiview`

Three detector pipelines run in sequence on the same observation: bare
DINO (`perceive_dino`), Molmo point-prompt (`perceive_point`), and
DINO+VLM (`perceive_dino_vlm`). `merge` then keeps every method that
found the target, drops geometrically-implausible thin-face masks, and —
when more than one candidate survives — runs a VLM disambiguator
(`select_best.py`) to pick the best mask. The winning cloud is passed
through `geometry.filter_and_compute_obb`, so `merge` emits an
already-filtered OBB (there is no separate `filter_obb` node). If no
method found the target, `merge` raises and the subgraph routes to
`not_found`.

- **Latency:** ~5–8 s per call (three detectors + a VLM merge step).
- **Robustness:** strong on cluttered scenes, small / low-contrast
  targets, uneven lighting. The three methods have different failure
  surfaces, so the survivor-plus-VLM-disambiguation merge beats any
  single method in practice — this is why the hand-validated
  `graph_cartesian_obb` workflow uses it for the target subgraph.

## Picking between them

- Default to `perceiving-objects-multiview` for single-shot
  pick-and-place — it is the safer choice whenever the task is not
  latency-bound.
- `perceiving-objects` when speed dominates (e.g. continuous tracking
  loops), or for clean scenes with a visually distinct, isolated target.
- Molmo not deployed (no `GAP_MOLMO_BASE_URL` endpoint):
  `perceiving-objects-multiview` still runs — the Molmo point-prompt
  pathway simply contributes nothing while bare DINO and DINO+VLM still
  feed the merge — so it stays available and useful, just with one fewer
  redundant detector. `perceiving-objects` remains the lighter-weight
  option there.
