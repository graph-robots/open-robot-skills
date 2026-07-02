# Geometry tool calling conventions

The geometry tool bundle's functions return TypedDict response wrappers —
a dict with named fields — never a bare value at the top level. The
`Ref(...)` binding shape must walk into the named field; getting it wrong
is the most common silent-fail mode in subgraph authoring.

## obb-field-binding

`geometry.filter_and_compute_obb` returns `{"obb": OrientedBoundingBox}`
— a response wrapper, NOT a bare `OrientedBoundingBox`. The
subgraph-level output binding for an OBB MUST walk into the `obb` field:

```python
sg.set_outputs(
    target_obb=Ref("filter_obb.obb"),    # CORRECT
    # target_obb=Ref("filter_obb"),      # WRONG — binds the whole {"obb": ...} wrapper
)
```

The same rule applies to the other geometry tools:

- `geometry.compute_obb` → `{"obb": OrientedBoundingBox}` — walk into `.obb`
- `geometry.filter_noise` / `geometry.mask_to_world_points` →
  `{"points": PointCloud}` — walk into `.points`
- `geometry.compute_drop_position` → `{"position": Vec3}` — walk into `.position`
- `geometry.top_down_grasp_from_obb` → `{"pose": Se3Pose}` — walk into `.pose`

Skill scripts that return *dicts* with named outputs (TypedDict in
Python) use trailing field names in `Ref(...)` exactly the same way:

```python
sg.set_outputs(
    target_mask=Ref("perceive.mask"),    # CORRECT — perceive returns {found, cloud, mask, score}
)
```

When in doubt, check the tool's result TypedDict in the geometry
bundle's `tools.py` (or the script's `Output` TypedDict) and walk into
the field you need.

Note on kwargs: the cloud-consuming geometry tools take `points=` (a
`gap.types.PointCloud`), not the legacy `point_cloud=` proto field —
e.g. `geometry.filter_and_compute_obb(points=Ref("perceive.cloud"))`.
