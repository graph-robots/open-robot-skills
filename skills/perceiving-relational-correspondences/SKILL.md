---
name: perceiving-relational-correspondences
description: Perceives multiple source and destination instances and resolves language-conditioned one-to-one or many-to-one correspondences. Use when an instruction relates instances by spatial order, proximity, appearance, labels, or a shared destination and downstream manipulation needs stable candidate IDs, masks, point clouds, and OBBs. Do not use for single-instance localization or for functional-feature fitting.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, correspondence, rgb-d, multi-object]}
gap:
  allowed_tools: [sam3.segment_text, geometry.mask_to_world_points]
  required_inputs: {instruction: string}
  produces_outputs: {status: string, source_id: string, target_kind: string, target_mask: Mask, destination_anchor_y: float, arm_id: int}
  exit_conditions:
    found: One eligible source instance and its correspondence anchor were selected.
    finished: No eligible uncompleted source remains.
    ambiguous: The instruction maps to multiple declared categories or pairs.
  canonical_scripts:
    - select_next_source: scripts/select_next_source.py
    - match_candidates: scripts/match_candidates.py
  streaming: false
---

# Perceiving Relational Correspondences

Resolve **which source goes with which destination** independently of grasping,
feature fitting, and motion planning.

## Contract

Require `instruction`, `source_description`, and `destination_description`.
Accept optional `cardinality`, `source_workspace_description`,
`destination_workspace_description`, and `completed_source_ids`.

Return a `PairSet` containing stable candidate IDs and, for every selected pair,
the source/destination mask, point cloud, OBB, relation, and confidence. Exit as
`paired`, `finished`, `ambiguous`, `not_found`, or `cardinality_mismatch`.
See [the schema](references/pair_set_schema.md).

## Procedure

1. Propose **all** source and destination instances with Grounding DINO; segment
   each tight proposal with SAM3 and recover world-frame geometry from RGB-D.
2. Assign stable IDs from category plus quantized world centroid. Never use
   detector list order as identity.
3. Ask the VLM to select candidate IDs and a relation operator from the
   instruction. The VLM may select IDs; it must not invent pixels or 3-D poses.
4. Validate IDs, cardinality, uniqueness, mask/cloud support, and the claimed
   relation deterministically. Reject an unverifiable answer as `ambiguous`.
5. Preserve the complete `PairSet` while manipulation iterates over it. Refresh
   geometry for a selected ID when needed, but do not recompute semantic ranks
   after moving an earlier object.

Supported relation operators should remain domain-neutral: corresponding order
along an inferred axis, nearest, matching appearance/label, explicit ID, and
shared destination. Do not encode words such as `left`, tool names, or hook
names as fixed world-axis branches; language interpretation chooses the
operator and direction, while geometry validates it.

Keep functional-feature perception in a later skill. A destination mask can
identify the correct fixture instance without claiming where its shaft, hole,
or seating pose is.

Use [the canonical subgraph](examples/canonical_subgraph.json) as the composition
boundary and `scripts/match_candidates.py` for deterministic assignment and
validation after candidates have been perceived. For iterative manipulation,
`scripts/select_next_source.py` accepts declarative `source_categories`, metric
`source_workspace`, `correspondence_axis`, and `arm_partition` inputs. Those
task-domain declarations belong in the graph; the script intentionally contains
no category names, workcell coordinates, or fixed meaning for left/right.

`select_next_source.py` returns a geometry-derived stable `source_id`, the
selected instance mask, the continuous correspondence anchor, and an arm ID.
Pass `completed_source_ids` when retry logic retains a PairSet; otherwise a
pickup-workspace gate can exclude already placed instances.
