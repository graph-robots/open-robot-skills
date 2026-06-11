---
name: tracking-objects
description: Long-running skill that drives the SAM3 tracker from the
  graph-scoped observation stream. Seeds the tracker via text prompt on the
  first frame, then polls the stream at update_hz and advances via
  sam3.tracker_update until the workflow signals termination, publishing a
  tracker snapshot per tick. Use when a workflow needs the live mask + box
  of an object across many frames — e.g. a supervisor branch that monitors
  a target's location while a policy manipulates it.
compatibility: requires gap>=0.1
metadata: {category: tracking, tags: [tracking, long-running, sam3, class-based, streaming]}
gap:
  allowed_tools:
    - sam3.tracker_init
    - sam3.tracker_update
    - sam3.tracker_close
  streaming: true
  tools:
    - tracking-objects.track: Run the SAM3 tracker loop over the observation stream; returns the final mask/box/confidence.
---

# tracking-objects

Long-running tracker skill. Init on first frame, update per tick, close on
exit. Used as a parallel sibling to other long-running skills (e.g. a
policy) when continuous state estimation is needed.

The skill is **class-based and stateful**: the tracker session id and the
last good mask/box live on the skill instance, so repeated visits to the
same state within one workflow execution can resume the session instead of
re-seeding it (pass `close_on_exit: false` to keep the session open across
visits; the final visit — or the instance teardown — closes it).

It is also a **streaming** skill (`gap.streaming: true`): each update tick
publishes a tracker snapshot
`{mask, box, confidence, object_present, n_updates}` via `ctx.publish`, so
downstream `{"$ref": "<node>"}` consumers see the latest tracked state
while the loop is still running.

## Install

Depends on the **sam3** tool bundle:

```bash
uv sync --extra sam3   # (pip: pip install -e "open-robot-skills[sam3]")
```

## When to use

- A workflow that needs the live mask + box of an object across many
  frames (e.g., a supervisor that monitors a target's location while a
  policy manipulates it).
- Wrapped under a `parallel` state with a `join_policy` so the tracker is
  cooperatively cancelled when the sibling branch finishes (the loop
  checks `ctx.cancel_token` every tick).

## Output

Returns the final mask, box, confidence, and a flag indicating whether
the object was visibly present at exit. Intermediate updates are
published as streaming snapshots; the return value exposes only the
final state.

## Tool form

The bundle also exposes the loop as a flat tool —
`tracking-objects.track` — for callers that want to invoke it as a single
unit (one fresh tracker session per call) rather than as a workflow
state.
