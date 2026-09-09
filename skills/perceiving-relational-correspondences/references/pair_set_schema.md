# PairSet schema

```json
{
  "status": "paired",
  "relation": {"operator": "corresponding_order", "axis": [0, 1, 0], "direction": 1},
  "pairs": [
    {
      "pair_id": "pair:source:...->destination:...",
      "source_id": "source:...",
      "destination_id": "destination:...",
      "source": {"mask": "Mask", "cloud": "PointCloud", "obb": "OrientedBoundingBox"},
      "destination": {"mask": "Mask", "cloud": "PointCloud", "obb": "OrientedBoundingBox"},
      "confidence": 0.91
    }
  ]
}
```

`operator` is one of `corresponding_order`, `nearest`, `matching_attribute`,
`explicit`, or `shared_destination`. Candidate IDs must be stable within the
scene and must be the only identifiers emitted by an interpreting VLM.

For `corresponding_order`, project both sets on `axis`, sort using `direction`,
and pair equal ranks. For `nearest`, solve a global minimum-cost bipartite
assignment rather than making greedy choices. `shared_destination` permits the
same destination ID in multiple pairs; other operators require unique IDs.
