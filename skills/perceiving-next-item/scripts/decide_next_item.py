"""Loop control for a pack-all / clean-all-items loop.

Decides, each pass, whether to grasp another item (`found`) or stop because
everything is packed (`none`). Robust termination is layered — perception
alone does NOT reliably end the loop (if a delivery fails the item stays on
the table and perception keeps seeing it, spinning until the perception-call
safety guard kills the episode), so we combine three signals, cheapest-first:

  1. **Authoritative env completion.** When the connector is a simulator, read
     its own ``sim.check_success`` verdict: ``task_completed`` True means every
     item is delivered -> ``none`` (a clean success). This never false-rejects,
     so the loop never spins on the basket after the table is clear nor stops
     early while items remain. Wrapped in try/except so a real-robot connector
     (no ``sim.*`` tools) falls through to perception.
  2. **No-progress guard.** If the delivered fraction (``completion_rate``) has
     not increased for ``_STUCK_LIMIT`` consecutive passes, the loop can no
     longer make progress (e.g. one ungraspable item left) -> ``none`` rather
     than re-grasping forever. State is per-process (one episode == one PID).
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

import logging
import os

import numpy as np

from gap import NodeContext

logger = logging.getLogger(__name__)

# Stop a cyclic loop that can no longer make progress: if completion_rate has
# not risen for this many consecutive passes, exit cleanly rather than looping.
_STUCK_LIMIT = 3


def _no_progress(cr: float | None) -> bool:
    if cr is None:
        return False
    path = f"/tmp/.gap_next_item_progress_{os.getpid()}"
    prev, stuck = -1.0, 0.0
    try:
        with open(path) as f:
            prev, stuck = (float(x) for x in f.read().split())
    except Exception:
        pass
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
        if sc.get("task_completed"):
            return {"route": "none"}
        if _no_progress(cr if isinstance(cr, (int, float)) else None):
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
