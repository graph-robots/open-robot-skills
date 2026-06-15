"""openpi π0.5 LIBERO checkpoint as a closed-loop VLA-policy skill.

Thin per-checkpoint skill over the shared policy loop
(:class:`gap.runtime.policy_skill.PolicyLoopSkill`). The model identity is
the bundle: ``preset = "pi05-libero"`` resolves the openpi π0.5 LIBERO
server through the executor's ``PolicyExecutor`` (the launcher boots the
matching preset automatically). The closed-loop body and the load-bearing
LIBERO observation encoding live in :mod:`gap.runtime.policy`.

Like the other class-based stateful skills, this module exposes the loop
both as the bundle's callable (the ``Pi05Libero`` class) and as a flat tool
``pi05-libero.run`` for callers that invoke it as a single unit.
"""

from __future__ import annotations

from typing import Any

from gap.runtime.policy_skill import PolicyLoopOutput, PolicyLoopSkill
from gap_core.skills import Param, SkillMeta
from gap_core.tools import tool

_PARAMS = {
    "observation_stream": Param(
        'Graph-scoped observation stream ({"$ref": "in.observation_stream"}).'
    ),
    "prompt": Param("Task instruction passed verbatim to the policy."),
    "termination_prompt": Param(
        "Optional VLM yes/no prompt; when non-empty the loop calls "
        "vlm.query_yes_no every term_period windows and exits with "
        "status='completed_by_vlm' on `yes`."
    ),
    "max_windows": Param("Hard ceiling on closed-loop iterations. Default 20."),
    "replan_every": Param(
        "Action-chunk rows consumed per window before replanning. Default 5 "
        "(the π-series LIBERO replan cadence)."
    ),
    "term_period": Param(
        "VLM termination check cadence (every N windows). Default 2."
    ),
    "arm_id": Param("Index into observation arms. Default 0."),
    "vlm_camera": Param(
        "Index into observation cameras for VLM termination screenshots. "
        "Default 0."
    ),
    "settle_steps": Param(
        "Dummy zero-actions sent before the first window so freshly-spawned "
        "objects settle. Default 10."
    ),
    "gripper_cycle_termination": Param(
        "When True the loop exits with status='gripper_cycle' once the policy "
        "completes one open->close->open grasp/release cycle (one item picked "
        "and released). Additive to max_windows and termination_prompt. "
        "Default False."
    ),
}

_OUTPUTS = {
    "status": "'gripper_cycle', 'completed_by_vlm', or 'max_windows'.",
    "num_windows": "How many windows were executed before exit.",
    "num_steps": "Total number of action rows applied across all windows.",
}


class Pi05Libero(PolicyLoopSkill):
    """openpi π0.5 LIBERO checkpoint (Franka Panda, LIBERO pick-and-place)."""

    preset = "pi05-libero"

    meta = SkillMeta(
        description=(
            "Run the openpi π0.5 LIBERO checkpoint (pi05_libero) in closed "
            "loop for a Franka Panda pick-and-place segment. Reads the "
            "graph-scoped observation_stream every window; terminates on a "
            "gripper cycle, a VLM yes/no check, or max_windows. The policy "
            "server is the bundle's own preset (no policy_id)."
        ),
        params=dict(_PARAMS),
        outputs=dict(_OUTPUTS),
    )


@tool(
    name="pi05-libero.run",
    summary=(
        "Run the π0.5 LIBERO checkpoint in closed loop "
        "(replan/execute/terminate) against its own preset server."
    ),
    tags=("policy",),
)
def run_policy(
    ctx,
    observation_stream: Any,
    prompt: str,
    termination_prompt: str = "",
    max_windows: int = 20,
    replan_every: int = 5,
    term_period: int = 2,
    arm_id: int = 0,
    vlm_camera: int = 0,
    settle_steps: int = 10,
    gripper_cycle_termination: bool = False,
) -> PolicyLoopOutput:
    """One-shot tool form: instantiate the skill and run one policy segment.

    Resolves the websocket client through ``ctx.policy_executor`` exactly
    like the class form (the executor's per-preset client cache makes a
    fresh skill instance per call cheap).
    """
    return Pi05Libero().run(
        ctx,
        observation_stream=observation_stream,
        prompt=prompt,
        termination_prompt=termination_prompt,
        max_windows=max_windows,
        replan_every=replan_every,
        term_period=term_period,
        arm_id=arm_id,
        vlm_camera=vlm_camera,
        settle_steps=settle_steps,
        gripper_cycle_termination=gripper_cycle_termination,
    )
