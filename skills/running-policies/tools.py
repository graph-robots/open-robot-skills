"""Run a learned VLA policy as a long-running, stateful skill.

This is the canonical entry point for VLA policies in graphs that use the
graph-scoped ``observation_stream``: the closed-loop
replan/execute/terminate body is
:func:`gap.runtime.policy.run_policy_loop`. Two properties matter:

1. Observations come from ``observation_stream.latest()`` rather than a
   fresh ``ctx.tool("robot.get_observation")`` per window.
2. The skill is class-based so the cached websocket client and any future
   per-policy state survives across multiple invocations of the same skill
   state within one workflow execution (connections are cached per
   ``policy_id`` — see :class:`gap.runtime.policy.PolicyExecutor`).

The module also exposes the loop as a flat tool — ``running-policies.run``
— for callers that invoke it as a single unit (class-in-tools.py wiring
makes the same class the bundle's stateful callable).
"""

from __future__ import annotations

from typing import Any, TypedDict

from gap.runtime.policy import run_policy_loop
from gap.skills import Param, Skill, SkillMeta, tool


class Output(TypedDict):
    status: str
    num_windows: int
    num_steps: int


class RunPolicy(Skill):
    """Stateful skill that wraps the VLA policy closed-loop body."""

    meta = SkillMeta(
        description=(
            "Run a learned VLA policy in closed loop until VLM termination "
            "fires, a gripper cycle completes, or max_windows is reached. "
            "Reads observations from the graph-scoped observation_stream "
            "every window. The policy's websocket connection is cached "
            "(via the executor's PolicyExecutor) and reused across "
            "invocations within one workflow."
        ),
        params={
            "observation_stream": Param(
                "Graph-scoped observation stream "
                "({\"$ref\": \"in.observation_stream\"})."
            ),
            "policy_id": Param(
                "Registered policy id (must appear under the task config's "
                "`policies:` block / PolicyManager registry)."
            ),
            "prompt": Param(
                "Task instruction passed verbatim to the policy."
            ),
            "termination_prompt": Param(
                "Optional VLM yes/no prompt; when non-empty the loop calls "
                "vlm.query_yes_no every term_period windows and exits "
                "with status='completed_by_vlm' on `yes`."
            ),
            "max_windows": Param(
                "Hard ceiling on closed-loop iterations. Default 20."
            ),
            "replan_every": Param(
                "Action-chunk length consumed per window before replanning. "
                "Default 5 (the pi-series LIBERO replan cadence)."
            ),
            "term_period": Param(
                "VLM termination check cadence (every N windows). Default 2."
            ),
            "arm_id": Param("Index into observation arms. Default 0."),
            "vlm_camera": Param(
                "Index into observation cameras for VLM termination "
                "screenshots. Default 0."
            ),
            "settle_steps": Param(
                "Dummy zero-actions sent before the first window so "
                "freshly-spawned objects settle. Default 10."
            ),
            "gripper_cycle_termination": Param(
                "Opt-in stage-termination mode for multi-item loops. "
                "When True the loop watches the gripper column the "
                "policy emits each window and exits with "
                "status='gripper_cycle' once it completes one "
                "open->close->open grasp/release cycle (one item picked "
                "and placed). Additive to max_windows and "
                "termination_prompt. Default False."
            ),
        },
        outputs={
            "status": (
                "'completed_by_vlm' (VLM termination fired), "
                "'gripper_cycle' (grasp/release cycle completed), "
                "or 'max_windows'."
            ),
            "num_windows": "How many windows were executed before exit.",
            "num_steps": (
                "Total number of action rows applied across all windows."
            ),
        },
    )

    def __init__(self) -> None:
        # Lazy reference to the executor's PolicyManager-backed
        # PolicyExecutor. Set on the first run() call (per workflow).
        self._policy_executor: Any | None = None

    def run(
        self,
        ctx,
        observation_stream: Any,
        policy_id: str,
        prompt: str,
        termination_prompt: str = "",
        max_windows: int = 20,
        replan_every: int = 5,
        term_period: int = 2,
        arm_id: int = 0,
        vlm_camera: int = 0,
        settle_steps: int = 10,
        gripper_cycle_termination: bool = False,
    ) -> Output:
        client = self._client_for(ctx, policy_id)
        result = run_policy_loop(
            ctx,
            client=client,
            policy_id=policy_id,
            prompt=prompt,
            termination_prompt=termination_prompt,
            max_windows=max_windows,
            replan_every=replan_every,
            term_period=term_period,
            arm_id=arm_id,
            vlm_camera=vlm_camera,
            settle_steps=settle_steps,
            gripper_cycle_termination=gripper_cycle_termination,
            obs_provider=lambda: observation_stream.latest(),
        )
        return {
            "status": result["status"],
            "num_windows": result["num_windows"],
            "num_steps": result["num_steps"],
        }

    # ------------------------------------------------------------------

    def _client_for(self, ctx, policy_id: str) -> Any:
        """Return a cached WebsocketClientPolicy for ``policy_id``.

        Looks up the executor's :class:`gap.runtime.policy.PolicyExecutor`
        (which owns the PolicyManager-backed client cache) the first time
        it is needed and keeps the reference on the instance.
        """
        if self._policy_executor is None:
            self._policy_executor = _find_policy_executor(ctx)

        if self._policy_executor is not None:
            return self._policy_executor.client_for(policy_id)

        raise RuntimeError(
            f"running-policies skill: no PolicyExecutor available to "
            f"resolve policy_id={policy_id!r}; the launcher must construct "
            f"a PolicyManager and PolicyExecutor before running a workflow "
            f"that uses running-policies."
        )


def _find_policy_executor(ctx) -> Any | None:
    """Best-effort lookup of the PolicyExecutor via the NodeContext.

    The runtime sets ``ctx.policy_executor`` when a tool/script node is
    dispatched; returns None when no executor was threaded through (e.g.
    running this skill outside the standard launcher).
    """
    return getattr(ctx, "policy_executor", None)


@tool(
    name="running-policies.run",
    summary="Run a VLA policy in closed loop (replan/execute/terminate) against a registered policy server.",
    tags=("policy",),
)
def run_policy(
    ctx,
    observation_stream: Any,
    policy_id: str,
    prompt: str,
    termination_prompt: str = "",
    max_windows: int = 20,
    replan_every: int = 5,
    term_period: int = 2,
    arm_id: int = 0,
    vlm_camera: int = 0,
    settle_steps: int = 10,
    gripper_cycle_termination: bool = False,
) -> Output:
    """One-shot tool form of the policy loop.

    Resolves the websocket client through ``ctx.policy_executor`` exactly
    like the skill form (the executor's per-policy client cache makes a
    fresh skill instance per call cheap).
    """
    return RunPolicy().run(
        ctx,
        observation_stream=observation_stream,
        policy_id=policy_id,
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
