# Perception pipeline invariants

These rules apply to any subgraph whose skill is a `perceiving-*`
bundle. They encode pipeline-level invariants that downstream consumers
(grasp, world building) depend on.

## emit-both-obb-and-mask

The subgraph-level outputs MUST emit BOTH `<name>_obb` AND `<name>_mask`.
The OBB is used by downstream grasp pose computation
(`geometry.top_down_grasp_candidates`, approach-above-target clearance).
The mask is used by the curobo grasp skill's world-building step
(`geometry.build_world_config` with the target's mask in `object_masks`)
to subtract the target from the collision world — a thin or degenerate
OBB alone cannot accurately isolate the target silhouette, which leaks
the object into the scene mesh and blocks the grasp descent.

If a perception skill internally produces a `cloud` and `mask` (and an
OBB derived later via `geometry.filter_and_compute_obb`), the subgraph
binding shape is:

```python
sg.set_outputs(
    target_obb=Ref("<obb_state>.obb"),
    target_mask=Ref("<perceive_state>.mask"),
)
```

(Replace `target_*` with the actual subgraph name prefix —
`container_obb`/`container_mask` etc.)

Missing either one is a hard failure during subgraph validation, not at
runtime — bind both.
