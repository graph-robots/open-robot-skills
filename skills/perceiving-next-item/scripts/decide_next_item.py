"""Loop control for a pack-all / clean-all-items loop.

Decides, each pass, whether to grasp another item (`found`) or stop because
everything is packed (`none`). Robust termination is layered — perception
alone does NOT reliably end the loop (if a delivery fails the item stays on
the table and perception keeps seeing it, spinning until the perception-call
safety guard kills the episode), so we combine three signals, cheapest-first:

  1. **Env telemetry (log-only by default).** When the connector is a
     simulator, its ``sim.check_success`` completion_rate is logged for
     diagnostics — but it does NOT terminate the loop: the env's task can be
     narrower than the user's instruction (a LIBERO object suite completes on
     one target item while a pack-all instruction still has items on the
     table), and it is privileged information besides. Wrapped in try/except
     so a real-robot connector (no ``sim.*`` tools) skips it.
     **Opt-in fast exit:** set ``GAP_DECIDE_TRUST_ENV=1`` when the env's own
     task IS the instruction (e.g. benchmarking on the pack-all suites) to
     exit on ``task_completed`` immediately — saves the 2-3 post-completion
     passes the perception path needs before the no-progress guard fires
     (~2 min each: empty-table false-positives get grasped at air).
  2. **No-progress guard.** If the delivered fraction (``completion_rate``) has
     not increased for ``_STUCK_LIMIT`` consecutive passes, the loop can no
     longer make progress (e.g. one ungraspable item left) -> ``none`` rather
     than re-grasping forever. State is per-episode: keyed by the workflow's
     trace dir (unique per trial) so a long-lived benchmark worker that runs
     many trials in one process never carries one trial's (prev, stuck) tail
     into the next — a bare-PID key made every post-first trial on a worker
     read "no progress" on its first pass and exit before grasping anything.
  3. **Perception.** Otherwise route on the paired ``perceive_item`` verdict
     (a pairwise-tournament detect with a container-excluding
     ``object_description``): a real item was found -> ``found``; nothing
     distinct from the container remains -> ``none``.

We deliberately do NOT geometrically reject items near the container: a real
item beside the basket falls inside any reasonable radius and would be dropped.
``container_obb`` is accepted (the subgraph wires it) but used only for
logging — the container-exclusion lives in the perception prompt.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile

import numpy as np

from gap import NodeContext

logger = logging.getLogger(__name__)

# Stop a cyclic loop that can no longer make progress: if completion_rate has
# not risen for this many consecutive passes, exit cleanly rather than looping.
_STUCK_LIMIT = 3


def _progress_path(ctx: NodeContext | None) -> str:
    """Per-episode scratch file for the no-progress counter.

    Scripts are re-imported on every node execution, so cross-iteration
    state must live outside the module. Key it by the workflow's trace dir
    (unique per trial/run) — NOT just the PID: benchmark workers execute
    many trials in one process, and a PID-keyed file carries the previous
    trial's (prev, stuck) tail into the next trial's first pass.
    """
    trace = getattr(ctx, "_trace", None) if ctx is not None else None
    out_dir = getattr(trace, "_output_dir", None) if trace is not None else None
    if out_dir is not None:
        tag = hashlib.sha1(str(out_dir).encode()).hexdigest()[:12]
    else:  # no-trace run: fall back to the PID (single episode per process)
        tag = f"pid{os.getpid()}"
    return os.path.join(
        tempfile.gettempdir(), f".gap_next_item_progress_{tag}"
    )


def _no_progress(cr: float | None, ctx: NodeContext | None = None) -> bool:
    if cr is None:
        return False
    path = _progress_path(ctx)
    prev, stuck = -1.0, 0.0
    try:
        with open(path) as f:
            prev, stuck = (float(x) for x in f.read().split())
    except Exception:
        pass
    # completion_rate is monotonic within an episode (delivered items are
    # retired) — a DROP means this key is stale state from an earlier
    # episode (e.g. a rerun reusing the same trace dir). Start fresh.
    if cr < prev - 1e-6:
        prev, stuck = -1.0, 0.0
    stuck = 0 if cr > prev + 1e-6 else stuck + 1
    try:
        with open(path, "w") as f:
            f.write(f"{cr} {stuck}")
    except Exception:
        pass
    return stuck >= _STUCK_LIMIT


def run(
    ctx: NodeContext,
    found: bool,
    cloud: dict,
    container_obb: dict | None = None,
) -> dict:
    # 1 + 2: authoritative env completion / no-progress guard (sim only).
    try:
        sc = ctx.tool("sim.check_success")
        cr = sc.get("completion_rate")
        logger.info(
            "[decide_next_item] completion_rate=%s task_completed=%s perceived_item=%s",
            f"{cr:.3f}" if isinstance(cr, (int, float)) else cr,
            sc.get("task_completed"), bool(found),
        )
        # NOTE: task_completed is deliberately NOT an exit signal by
        # default. It verifies the ENV's task, which can be narrower than
        # the user's instruction (LIBERO object suites complete on one
        # target item while a pack-all instruction still has items left) —
        # and it is privileged info. Perception (#3) ends the loop; the
        # no-progress guard (#2) covers the perception-false-positive spin
        # case. GAP_DECIDE_TRUST_ENV=1 opts into the fast authoritative
        # exit for setups where the env task IS the instruction.
        if os.environ.get("GAP_DECIDE_TRUST_ENV", "").lower() in ("1", "true", "yes") \
                and sc.get("task_completed"):
            logger.info("[decide_next_item] env task_completed (trusted via "
                        "GAP_DECIDE_TRUST_ENV) -> none")
            return {"route": "none"}
        if _no_progress(cr if isinstance(cr, (int, float)) else None, ctx):
            logger.info("[decide_next_item] no completion progress for %d passes -> none",
                        _STUCK_LIMIT)
            return {"route": "none"}
    except Exception:
        pass  # non-sim connector (real robot) -> fall through to perception

    # 3: perception verdict.
    if not found:
        logger.info("[decide_next_item] no grocery item perceived -> none (loop exit)")
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        logger.info("[decide_next_item] empty item cloud -> none (loop exit)")
        return {"route": "none"}
    logger.info("[decide_next_item] grocery item perceived (%d pts) -> found", pts.shape[0])
    return {"route": "found"}
