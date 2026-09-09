---
name: routing-manipulation-relations
description: Routes a typed functional relation to a declaratively named manipulation branch. Use after perception has produced a relation and before selecting a transport, feature-mating, insertion, or placement executor. The router contains no object names or task-specific geometry.
license: Apache-2.0
compatibility: requires gap>=0.1
metadata: {category: coordination, tags: [routing, relations, composition, zero-shot]}
gap:
  allowed_tools: []
  required_inputs: {relation: string}
  produces_outputs: {mode: string}
  exit_conditions:
    routed: Exactly one declared route matched the relation.
    unsupported: No route matched the relation.
    ambiguous: More than one route matched the relation.
  canonical_scripts:
    - choose_placement_mode: scripts/choose_placement_mode.py
  streaming: false
---

# Routing Manipulation Relations

Select an execution family from a typed functional relation, never from an
object category. A newly perceived object can therefore reuse an existing
execution chain whenever it exposes a supported relation.

## Contract

Input `relation` is a semantic relation such as `loop_over_shaft`,
`shaft_into_aperture`, or `feature_into_container`. The optional
`relation_routes` input is a list of records with a branch `mode` and a list of
`relations`. Exactly one record must match.

The canonical defaults preserve the common split:

- `loop_over_shaft`, `feature_to_fixture` -> `fixture`
- `shaft_into_aperture`, `tip_through_aperture`, `insert_through`,
  `feature_into_container` -> `drop`

Branch names are graph-level labels. A graph may supply a different
`relation_routes` table without changing the script. Unsupported or overlapping
tables fail explicitly so an agent cannot silently choose the wrong executor.

## Composition

Place this skill after `perceiving-functional-features`. Route fixture relations
to `computing-feature-mating-poses`; route container/aperture placement to an
appropriate insertion or `transporting-objects` subgraph. The task graph owns
the branch wiring while this skill owns deterministic relation dispatch.
