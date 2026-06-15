---
name: pi05-libero
description: Run the openpi π0.5 LIBERO checkpoint (pi05_libero) as a
  closed-loop VLA policy for the dexterous pick-and-place segment of a task.
  Drives a Franka Panda in the LIBERO/robosuite OSC_POSE action space from
  agentview + wrist cameras; the policy server is the bundle's own preset
  (no policy_id). Reads the graph-scoped observation_stream each window and
  terminates on a gripper open→close→open cycle, a VLM yes/no check, or
  max_windows. Use when a pick/place (or pick-and-drop-in-container) segment
  on tabletop rigid LIBERO objects is delegated to a learned policy — best
  steered (perceive + hover above the target) first. NOT for
  deformables/cloth folding, articulated objects, non-Franka embodiments, or
  tasks outside the LIBERO pick-place distribution.
compatibility: requires gap>=0.1
metadata: {category: policy, tags: [policy, vla, libero, pi, gpu, long-running, class-based]}
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "server.py", "policy:checkpoint",
              "--policy.config=pi05_libero",
              "--policy.dir=s3://openpi-assets/checkpoints/pi05_libero",
              "--port", "{port}"]
    protocol: websocket
    requires_gpu: true
    weights_uri: s3://openpi-assets/checkpoints/pi05_libero
  allowed_tools:
    - sim.apply_policy_action
    - vlm.query_yes_no
    - robot.get_observation
    - robot.execute_trajectory
  exit_conditions:
    gripper_cycle: One grasp/release (open→close→open) cycle completed — one item picked and released.
    completed_by_vlm: The VLM termination prompt answered yes.
    max_windows: The hard window cap was reached without a completion signal (the policy ran but did not declare itself done).
    failed: The policy loop errored (server/inference/execution failure). This is the subgraph's on_error exit.
  tools:
    - pi05-libero.run: Run the π0.5 LIBERO checkpoint in closed loop (replan/execute/terminate) against its own preset server.
---

# pi05-libero

Closed-loop VLA-policy skill backed by **one model checkpoint**: openpi's
π0.5 LIBERO checkpoint (`pi05_libero`). The skill *is* the model — it owns
its serving preset (`pi05-libero`), so a policy node names this skill, not a
free-floating `policy_id`. The closed-loop replan/execute/terminate body and
the load-bearing LIBERO observation encoding live in
`gap.runtime.policy.run_policy_loop`; the websocket client is resolved (and
cached per preset) through the executor's `PolicyExecutor`.

## Capability

- **Embodiment:** Franka Panda (LIBERO/robosuite), OSC_POSE delta action
  space `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`. No embodiment translation
  happens in the loop — the checkpoint's native action space is forwarded to
  `sim.apply_policy_action`.
- **Tasks:** the LIBERO pick-and-place distribution — pick a tabletop rigid
  object, optionally place/drop it in a container. Works best *steered*:
  perceive the target and hover the end-effector above it (preserving the
  current rotation) before handing over, so the policy starts in-distribution.
- **Not for:** deformables / cloth folding, articulated objects, non-LIBERO
  embodiments, or tasks the checkpoint never saw. If the task is outside this
  envelope, pick a different skill or report a missing capability — do not
  delegate it here and hope.

## Serving

The bundle ships its own `server.py` (vendored from openpi `scripts/serve_policy.py`)
and declares `openpi` as a git dep in its own `pyproject.toml` — so this
bundle is **self-contained**: no `$GAP_OPENPI_DIR` clone, no shared venv.
First-run setup is `gap skills install pi05-libero`, which `uv sync`s the
bundle's `.venv/` with all model deps. The launcher then spawns the server
via `uv run --project policies/pi05-libero -- python server.py ...` (so the
bundle's own venv activates automatically) and downloads the checkpoint from
`s3://openpi-assets/checkpoints/pi05_libero` on first run.

Run it yourself with `gap policy serve pi05-libero`. A `policies:` config
entry named `pi05-libero` overrides the recipe (e.g. an external `url:`).

## Termination & exits

The loop exits on whichever fires first — a commanded gripper
open→close→open cycle (`gripper_cycle`, the per-item terminator for
clean-all loops; set `gripper_cycle_termination: true`), a non-empty
`termination_prompt` answered yes by the VLM (`completed_by_vlm`), or the
`max_windows` backstop. These are the subgraph's success exits; the failure
exit is `failed` (the loop raised). **Whether the task actually succeeded is
a checkpoint, not an exit** — attach a postcondition that checks the world
(e.g. the object is in the container), never an exit value like "folded".
