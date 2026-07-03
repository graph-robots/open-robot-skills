"""Loop control for a pack-all / clean-all-items loop.

Decides, each pass, whether to grasp another item (`found`) or stop because
everything is packed (`none`). Robust termination is layered — perception
alone does NOT reliably end the loop (if a delivery fails the item stays on
the table and perception keeps seeing it, spinning until the perception-call
safety guard kills the episode) — and every signal here is UNPRIVILEGED
(camera + VLM only), so the same policy runs on a real robot:

  1. **VLM completion check (primary).** Every pass, show the exterior
     (agentview) frame to the VLM and ask whether every item is now inside
     the container. A confident YES exits the loop cleanly on ``none``. The
     check only ever forces a STOP — a NO or an unavailable VLM never forces
     the loop to continue, so a missed/ungraspable item still falls through
     to the guards below and the episode always terminates.
  2. **No-progress guard.** If the SAME target (perceived-cloud centroid
     within ``_SAME_TARGET_M``) comes back ``_STUCK_LIMIT`` consecutive
     passes, the loop cannot make progress on it (e.g. ungraspable) ->
     ``none`` rather than re-grasping forever. A total-pass budget
     (``_MAX_PASSES``) backstops pathological alternation. State is
     per-episode: keyed by the workflow's trace dir (unique per trial) so a
     long-lived benchmark worker never carries one trial's state into the
     next — a bare-PID key made every post-first trial on a worker exit on
     its first pass.
  3. **Perception.** Otherwise route on the paired ``perceive_item`` verdict
     (a pairwise-tournament detect with a container-excluding
     ``object_description``): a real item was found -> ``found``; nothing
     distinct from the container remains -> ``none``.

Env telemetry: when the connector is a simulator, ``sim.check_success`` is
logged for diagnostics only. ``GAP_DECIDE_TRUST_ENV=1`` opts into using its
``task_completed`` as an exit — for benchmark setups where the env's task IS
the instruction — but nothing requires it.

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

# Stop a cyclic loop that can no longer make progress: if the SAME target is
# re-perceived this many consecutive passes (no delivery removed it, nothing
# moved), exit cleanly rather than looping.
_STUCK_LIMIT = 3
# Two perceived-cloud centroids closer than this (m, XY) are the same target.
_SAME_TARGET_M = 0.03
# Absolute pass budget — backstop for pathological alternation between two
# ungraspable targets (each pass is a full perceive+grasp+transport cycle,
# so 30 passes is far beyond any legitimate episode).
_MAX_PASSES = 30

# Per-pass VLM completion check (ported from the grocery-packing benchmark's
# route_next_object). A confident YES is the loop's primary, unprivileged
# stop signal; NO/None fall through to the guards + perception verdict.
_PACKED_PROMPT = (
    "A robot is packing grocery items from a table into a basket. Looking at "
    "this image of the table and the basket, have ALL the grocery items been "
    "placed INSIDE the basket, leaving the table surface empty except for the "
    "basket itself (and the robot)? Answer YES only if there is NO grocery "
    "item left resting on the table outside the basket."
)


def _exterior_rgb(ctx: NodeContext) -> np.ndarray | None:
    """The agentview (exterior) RGB frame — the wrist eye-in-hand view is too
    close to judge the whole tabletop. Returns None if no camera is available."""
    obs = ctx.tool("robot.get_observation")
    cams = obs.get("cameras") if isinstance(obs, dict) else None
    cams = cams or []
    if isinstance(cams, dict):
        cams = list(cams.values())
    ext = [c for c in cams if "eye_in_hand" not in (c.get("name") or "")]
    cam = next(iter(ext or cams), None)
    if cam is None or cam.get("rgb") is None:
        return None
    return np.asarray(cam["rgb"])


def _vlm_all_packed(ctx: NodeContext) -> bool | None:
    """Ask the VLM whether every object is in the basket. Returns True/False,
    or None when the check could not run (no camera, or VLM error/missing
    creds) so the caller falls back to the guard + perception exits."""
    try:
        rgb = _exterior_rgb(ctx)
        if rgb is None:
            logger.warning("[decide_next_item] VLM all-packed check skipped: no camera frame")
            return None
        resp = ctx.tool("vlm.query_yes_no", prompt=_PACKED_PROMPT, image=rgb)
        ans = bool(resp.get("answer"))
        logger.info("[decide_next_item] VLM all-packed? answer=%s text=%r",
                    ans, str(resp.get("text"))[:200])
        return ans
    except Exception as exc:  # noqa: BLE001  (auth/network/no-bundle -> fall through)
        logger.warning("[decide_next_item] VLM all-packed check failed: %s", exc)
        return None


def _progress_path(ctx: NodeContext | None) -> str:
    """Per-episode scratch file for the no-progress counter.

    Scripts are re-imported on every node execution, so cross-iteration
    state must live outside the module. Key it by the workflow's trace dir
    (unique per trial/run) — NOT just the PID: benchmark workers execute
    many trials in one process, and a PID-keyed file carries the previous
    trial's state into the next trial's first pass.
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


def _no_progress(target_xy: tuple[float, float] | None,
                 ctx: NodeContext | None = None) -> bool:
    """UNPRIVILEGED stuck detector, keyed on the perceived target itself.

    A successful delivery removes the item, so the next pass perceives a
    DIFFERENT target; re-perceiving the same centroid means the last
    grasp+transport cycle changed nothing. Fires after ``_STUCK_LIMIT``
    consecutive same-target passes, or unconditionally past ``_MAX_PASSES``.
    """
    if target_xy is None:
        return False
    path = _progress_path(ctx)
    px, py, consec, total = 1e9, 1e9, 0.0, 0.0
    try:
        with open(path) as f:
            px, py, consec, total = (float(x) for x in f.read().split())
    except Exception:
        pass
    same = abs(target_xy[0] - px) < _SAME_TARGET_M and \
        abs(target_xy[1] - py) < _SAME_TARGET_M
    consec = consec + 1 if same else 0
    total += 1
    try:
        with open(path, "w") as f:
            f.write(f"{target_xy[0]} {target_xy[1]} {consec} {total}")
    except Exception:
        pass
    if total > _MAX_PASSES:
        logger.info("[decide_next_item] pass budget exceeded (%d) -> none", _MAX_PASSES)
        return True
    if consec >= _STUCK_LIMIT:
        logger.info(
            "[decide_next_item] same target (%.3f, %.3f) for %d consecutive "
            "passes -> none", target_xy[0], target_xy[1], _STUCK_LIMIT)
        return True
    return False


def run(
    ctx: NodeContext,
    found: bool,
    cloud: dict,
    container_obb: dict | None = None,
) -> dict:
    # 1: VLM completion check — the primary, unprivileged stop signal.
    if _vlm_all_packed(ctx) is True:
        logger.info("[decide_next_item] VLM reports all items packed -> none")
        return {"route": "none"}

    # Env telemetry (log-only unless GAP_DECIDE_TRUST_ENV=1 opts in).
    try:
        sc = ctx.tool("sim.check_success")
        cr = sc.get("completion_rate")
        logger.info(
            "[decide_next_item] completion_rate=%s task_completed=%s perceived_item=%s",
            f"{cr:.3f}" if isinstance(cr, (int, float)) else cr,
            sc.get("task_completed"), bool(found),
        )
        if os.environ.get("GAP_DECIDE_TRUST_ENV", "").lower() in ("1", "true", "yes") \
                and sc.get("task_completed"):
            logger.info("[decide_next_item] env task_completed (trusted via "
                        "GAP_DECIDE_TRUST_ENV) -> none")
            return {"route": "none"}
    except Exception:
        pass  # non-sim connector (real robot) — telemetry only

    # 3 (order: cheap verdicts first): perception verdict.
    if not found:
        logger.info("[decide_next_item] no grocery item perceived -> none (loop exit)")
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        logger.info("[decide_next_item] empty item cloud -> none (loop exit)")
        return {"route": "none"}

    # 2: unprivileged no-progress guard on the perceived target itself.
    centroid = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
    if _no_progress(centroid, ctx):
        return {"route": "none"}

    logger.info("[decide_next_item] grocery item perceived (%d pts) -> found", pts.shape[0])
    return {"route": "found"}
