"""Long-running skill that drives the SAM3 tracker from observation_stream.

Initializes a tracker session on the first frame read from the observation
stream, then polls the stream at ``update_hz`` and advances the session via
``sam3.tracker_update`` until the workflow signals termination. Always calls
``sam3.tracker_close`` in a ``finally`` block (when ``close_on_exit`` is
True, the default) so tracker state is freed even on exception /
cancellation.

Class-based (:class:`gap.skills.Skill`) so the tracker session is genuine
instance state: with ``close_on_exit=False`` a later visit to the same
workflow state *resumes* the session — ``sam3.tracker_init`` runs exactly
once per session — instead of re-seeding from scratch. The runtime
instantiates one instance per workflow execution and discards it in the
executor's ``finally`` block.

Streaming contract: each update tick publishes a snapshot
``{mask, box, confidence, object_present, n_updates}`` via ``ctx.publish``
(the bundle declares ``gap.streaming: true``), so parallel siblings can read
the latest tracked state through a stream slot.

Cooperative cancellation: the loop calls ``ctx.cancel_token.raise_if_set()``
every tick, so a sibling parallel branch (e.g. a first-success policy) can
pre-empt the tracker.

The module also exposes the loop as a flat tool —
``tracking-objects.track`` — which runs one fresh, self-closing tracker
session per call (the class-in-tools.py wiring makes the same class the
bundle's stateful callable).
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from gap.skills import Skill, tool
from gap.types import BoundingBox2D, Mask


class Output(TypedDict):
    final_mask: Mask | None
    final_box: BoundingBox2D | None
    final_confidence: float
    object_present: bool
    n_updates: int


class TrackObject(Skill):
    """Stateful SAM3 tracker loop — init once, update per tick, close on exit."""

    def __init__(self) -> None:
        # Tracker-session state. Persist across run() visits within one
        # workflow execution (the executor keeps one instance per skill).
        self._tracker_id: str = ""
        self._last_mask: Mask | None = None
        self._last_box: BoundingBox2D | None = None
        self._last_conf: float = 0.0
        self._present: bool = False
        self._n_updates: int = 0

    # ------------------------------------------------------------------

    def run(
        self,
        ctx,
        observation_stream: Any,
        target_prompt: str,
        camera_index: int = 0,
        update_hz: float = 5.0,
        max_updates: int = 200,
        allow_lost_frames: int = 10,
        close_on_exit: bool = True,
    ) -> Output:
        frame = self._frame(observation_stream, camera_index)

        if not self._tracker_id:
            init_resp = ctx.tool(
                "sam3.tracker_init",
                image=frame,
                text=target_prompt,
                object_name=target_prompt,
            )
            self._tracker_id = init_resp["tracker_id"]
            self._last_mask = init_resp["initial_mask"]
            self._last_box = init_resp["initial_box"]
            self._last_conf = init_resp["score"]
            self._present = init_resp["object_present"]

            if not self._present:
                # Seed detection failed — return immediately with
                # object_present=False (the tool already closed its session;
                # tracker_id is empty).
                self._tracker_id = ""
                return {
                    "final_mask": self._last_mask,
                    "final_box": self._last_box,
                    "final_confidence": self._last_conf,
                    "object_present": False,
                    "n_updates": 0,
                }

        lost_streak = 0 if self._present else 1
        period_s = 1.0 / max(update_hz, 1e-3)

        try:
            for _ in range(max_updates):
                # Cooperative cancellation: bail out if a sibling parallel
                # branch has signaled (e.g. supervisor returned,
                # first_success policy).
                cancel_token = getattr(ctx, "cancel_token", None)
                if cancel_token is not None:
                    cancel_token.raise_if_set()
                start = time.monotonic()
                frame = self._frame(observation_stream, camera_index)
                upd = ctx.tool(
                    "sam3.tracker_update",
                    tracker_id=self._tracker_id,
                    image=frame,
                )
                self._n_updates += 1
                if upd["object_present"]:
                    self._last_mask = upd["mask"]
                    self._last_box = upd["box"]
                    self._last_conf = upd["confidence"]
                    self._present = True
                    lost_streak = 0
                else:
                    self._present = False
                    lost_streak += 1

                self._publish_snapshot(ctx, upd)

                if not upd["object_present"] and lost_streak >= allow_lost_frames:
                    break

                elapsed = time.monotonic() - start
                sleep_for = period_s - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            return {
                "final_mask": self._last_mask,
                "final_box": self._last_box,
                "final_confidence": self._last_conf,
                "object_present": self._present,
                "n_updates": self._n_updates,
            }
        finally:
            if close_on_exit:
                self.close(ctx)

    # ------------------------------------------------------------------

    def close(self, ctx) -> None:
        """Free the tracker session (idempotent, best-effort)."""
        tracker_id, self._tracker_id = self._tracker_id, ""
        if not tracker_id:
            return
        try:
            ctx.tool("sam3.tracker_close", tracker_id=tracker_id)
        except Exception:
            # tracker_close is best-effort; the session is reaped by TTL.
            pass

    # ------------------------------------------------------------------

    @staticmethod
    def _frame(observation_stream: Any, camera_index: int):
        obs = observation_stream.latest()
        cameras = obs["cameras"]
        if camera_index < 0 or camera_index >= len(cameras):
            raise RuntimeError(
                f"camera_index={camera_index} out of range "
                f"(have {len(cameras)} cameras)"
            )
        return cameras[camera_index]["rgb"]

    def _publish_snapshot(self, ctx, upd: dict) -> None:
        """Publish the per-tick tracker snapshot (streaming contract)."""
        snapshot = {
            "mask": upd["mask"],
            "box": upd["box"],
            "confidence": upd["confidence"],
            "object_present": upd["object_present"],
            "n_updates": self._n_updates,
        }
        try:
            ctx.publish(snapshot)
        except RuntimeError:
            # Invoked from a non-streaming node (e.g. the flat tool form) —
            # the snapshot is informational, so skip publishing.
            pass


@tool(
    name="tracking-objects.track",
    summary="Track a described object across the observation stream with SAM3; returns the final mask, box and confidence.",
    tags=("perception",),
)
def track(
    ctx,
    observation_stream: Any,
    target_prompt: str,
    camera_index: int = 0,
    update_hz: float = 5.0,
    max_updates: int = 200,
    allow_lost_frames: int = 10,
) -> Output:
    """One-shot tool form of the tracker loop.

    Runs one fresh tracker session (init → update loop → close) and returns
    the final tracked state. For session reuse across multiple workflow
    state visits use the skill form with ``close_on_exit=False``.
    """
    return TrackObject().run(
        ctx,
        observation_stream=observation_stream,
        target_prompt=target_prompt,
        camera_index=camera_index,
        update_hz=update_hz,
        max_updates=max_updates,
        allow_lost_frames=allow_lost_frames,
        close_on_exit=True,
    )
