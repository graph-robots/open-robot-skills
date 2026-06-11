---
name: running-policies
description: Run a learned VLA policy in closed loop until a termination
  signal fires or max_windows is reached. Reads observations from the
  graph-scoped observation_stream every window, sends them to an openpi
  websocket policy server, and forwards the action chunks to the robot.
  The policy's websocket connection is cached on the skill instance and
  reused across invocations within one workflow. Use when some segment of
  the manipulation is delegated to a learned policy (e.g. "pick up the
  object via policy, place via skills").
compatibility: requires gap>=0.1
metadata: {category: policy, tags: [policy, vla, long-running, class-based]}
gap:
  allowed_tools:
    - sim.apply_policy_action
    - vlm.query_yes_no
    - robot.get_observation
    - robot.execute_trajectory
  tools:
    - running-policies.run: Run a VLA policy in closed loop (replan/execute/terminate) against a registered policy server.
---

# running-policies

Class-based stateful skill that drives a learned VLA policy through its
closed-loop replan/execute/terminate body
(:func:`gap.runtime.policy.run_policy_loop`). The websocket client to the
policy server is resolved through the executor's
:class:`gap.runtime.policy.PolicyExecutor` (one cached connection per
`policy_id`), so reconnect-per-call doesn't dominate latency. Used by
workflows that include a VLA-policy node.

## Install

The websocket client comes from the `openpi-client` package:

```bash
uv sync --extra running-policies   # (pip: pip install -e "open-robot-skills[running-policies]")
```

## Configuring policies

A policy node names a registered `policy_id`. Policies are registered in
the task config's `policies:` block, consumed by
`gap.runtime.policy_manager.PolicyManager`. Three entry styles:

- **External server** — `url: ws://host:port` (you run the server).
- **Managed server** — `start_cmd: "... --port {port}"` (+ optional
  `env:`); the manager spawns and tears down the subprocess.
- **Preset** — `preset: <name>`; expanded by
  `gap.runtime.policy_presets.resolve_policies` into a known-good managed
  entry. Shipped presets:
  - `pi05-libero` — openpi reference recipe for the LIBERO π0.5
    checkpoint (`s3://openpi-assets/checkpoints/pi05_libero`, served via
    `serve_policy.py`; checkpoint downloads on first run).
  - `molmoact-libero` — the MolmoAct LIBERO checkpoint
    (`hf://allenai/MolmoAct-7B-D-LIBERO-0812`) behind a vLLM-style serve
    script speaking the openpi websocket protocol.

  Both presets reference `$GAP_OPENPI_DIR` (your openpi / MolmoAct
  checkout), expanded by the shell at spawn time.

The one-command path is:

```bash
gap policy serve pi05-libero    # downloads the checkpoint if missing, spawns the server
```

and a task.yaml entry of `policies: {my_policy: {preset: pi05-libero}}`.

## When to use

- Workflows where some segment of the manipulation is delegated to a
  learned policy (e.g. "pick up the object via policy, place via skills").
- Wrapped under a `parallel` state with `join_policy: first_success` and
  a supervisor branch so the policy can be cooperatively cancelled (the
  loop checks `ctx.cancel_token` every window).

## Lifecycle

`run` is invoked per state visit. The first visit resolves the websocket
client (via `ctx.policy_executor`, which caches one connection per
`policy_id`) and runs the closed-loop body. Subsequent visits within the
same workflow execution reuse the cached client. The graph executor's
`finally` block discards the instance at workflow exit.

## Termination knobs

Three additive terminators; the loop exits on whichever fires first:

- **`max_windows`** (default 20) — the hard backstop; exits with
  `status: "max_windows"`.
- **VLM termination** — set `termination_prompt` to a yes/no question;
  every `term_period` windows (default 2) the loop calls
  `vlm.query_yes_no` on the `vlm_camera` frame and exits with
  `status: "completed_by_vlm"` on `yes`. Leave the prompt empty to skip
  all VLM calls.
- **Gripper-cycle termination** — see below; exits with
  `status: "gripper_cycle"`.

## Gripper-cycle termination

Set `gripper_cycle_termination: true` to end the stage as soon as the
policy completes one **open → close → open** gripper cycle — i.e. one
item picked up and released. The loop watches the commanded gripper
column the VLA emits each window (LIBERO convention: `-1` open, `+1`
closed), debounced so a momentary failed grasp does not fire (the close
must hold ≥ 3 windows), and exits with `status: "gripper_cycle"`.

This is the per-item terminator for long-horizon clean-all-items loops
(e.g. grocery packing): the workflow loops `running-policies` back to a
re-perception stage, and the gripper cycle is what hands control back
after each pick-and-place. It is additive to `max_windows` (the hard
backstop) and the VLM `termination_prompt` (left empty when the gripper
cycle is the intended signal — it needs no VLM calls).

## Action forwarding

Each window the loop encodes the latest observation into the openpi obs
dict, asks the policy server for an action chunk, and forwards the first
`replan_every` rows (default 5 — the π-series LIBERO replan cadence) to
the env via `sim.apply_policy_action`. The action layout is the policy
checkpoint's native space; no embodiment translation happens in the loop.
Connectors without a VLA passthrough don't register
`sim.apply_policy_action` — the call fails loudly rather than driving the
robot with the wrong action space. Before the first window the loop sends
`settle_steps` (default 10) zero-EE-delta / gripper-open dummy actions so
freshly spawned objects settle before the policy sees its first frame.
