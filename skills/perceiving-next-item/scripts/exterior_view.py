"""Keep only the exterior (agentview) camera frame, dropping the wrist.

The DINO+VLM perceive script fuses all supplied views; the Franka eye-in-hand is
mounted at an angle and, fused with the agentview, bloats the OBB by merging
neighbouring objects' points (measured: 19 cm vs the true ~5 cm footprint). The
single exterior view localizes objects to ~1 cm (vs the benchmark ground truth),
so we perceive from it alone.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import CameraFrame


class Output(TypedDict):
    cameras: list[CameraFrame]


def run(ctx: NodeContext, cameras: list[CameraFrame]) -> Output:
    ext = [c for c in cameras if "eye_in_hand" not in (c.get("name") or "")]
    return {"cameras": ext or list(cameras)}
