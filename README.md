<div align="center">

# open-robot-skills

**What your robot can do, one directory at a time.**

A curated, contributable library of manipulation **skills** and model-backed
**tool bundles** for [GaP — graph as policy](https://github.com/graph-robots/graph-as-policy), in the
[Anthropic Agent Skills](https://agentskills.io/specification) format.
The LLM pipeline composes these bundles into executable robot graphs;
every bundle is one directory, one `SKILL.md`, one PR.

[![Skills](https://img.shields.io/badge/skills-10-blue.svg)](#skills--what-the-robot-can-do)
[![Tools](https://img.shields.io/badge/tool%20bundles-7-orange.svg)](#tools--what-the-robot-can-compute)
[![Format: Agent Skills](https://img.shields.io/badge/format-Agent%20Skills-purple.svg)](https://agentskills.io/specification)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing-a-bundle)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

Clone this repo **next to the graph-as-policy checkout** (or set `GAP_SKILLS_PATH`) and
every GaP command — `gap run`, `gap generate`, `gap benchmark` — discovers
it automatically; no flags, no registration.

New to GaP? Start with [the libero_quickstart example](https://github.com/graph-robots/graph-as-policy/tree/main/examples/libero_quickstart)
or the [15-minute tour](https://github.com/graph-robots/graph-as-policy/blob/main/docs/quickstart.md),
then come back here when you want to add a capability.

## Contents

- [Skills — what the robot can do](#skills--what-the-robot-can-do)
- [Tools — what the robot can compute](#tools--what-the-robot-can-compute)
- [Install](#install)
- [Verify a checkout](#verify-a-checkout)
- [Contributing a bundle](#contributing-a-bundle)
- [Use with Claude Code](#use-with-claude-code)

## Skills — what the robot can do

Manipulation strategies. A skill owns subgraphs in generated workflows: its
`SKILL.md` body is the guidance the subgraph agent reads, its
`scripts/` are the canonical recipes the graph executes, and its declared
`exit_conditions` are what downstream routing keys on. Closed-loop VLA
policies (`pi05-libero`, `molmoact-libero`) now live under `policies/`
with `kind=policy` — they share the bundle format but are surfaced to
graphs through the policy plane, not the skill registry.

| Bundle | Description | Tools | Extra |
|---|---|---|---|
| [grasping-direct-ik](skills/grasping-direct-ik/) | Direct IK align-then-descend grasping. | — | `open-robot-skills[grasping-direct-ik]` |
| [grasping-short-axis](skills/grasping-short-axis/) | Deterministic short-axis-aligned grasp with CuRobo. | — | `open-robot-skills[grasping-short-axis]` |
| [grasping-with-planner](skills/grasping-with-planner/) | Collision-aware grasping using cuRobo trajectory planning over a per-observation collision world. | — | `open-robot-skills[grasping-with-planner]` |
| [molmoact-libero](policies/molmoact-libero/) | Run the MolmoAct LIBERO checkpoint as a closed-loop VLA policy for the LIBERO Franka pick-and-place segment (the MolmoAct alternative to `pi05-libero`). Owns its serving preset — no `policy_id`. | `molmoact-libero.run` | `uv run gap skills install molmoact-libero` |
| [perceiving-object-parts](skills/perceiving-object-parts/) | Hierarchical perception for subpart targeting. | — | `open-robot-skills[perceiving-object-parts]` |
| [perceiving-objects](skills/perceiving-objects/) | Fast single-path 3D object perception. | — | `open-robot-skills[perceiving-objects]` |
| [perceiving-objects-multiview](skills/perceiving-objects-multiview/) | Robust three-method 3D object perception. | — | `open-robot-skills[perceiving-objects-multiview]` |
| [perceiving-objects-oneshot](skills/perceiving-objects-oneshot/) | Lightweight one-shot 3D object perception. | — | `open-robot-skills[perceiving-objects-oneshot]` |
| [pi05-libero](policies/pi05-libero/) | Run the openpi π0.5 LIBERO checkpoint as a closed-loop VLA policy for the LIBERO Franka pick-and-place segment. Owns its serving preset — no `policy_id`. | `pi05-libero.run` | `uv run gap skills install pi05-libero` |
| [tracking-objects](skills/tracking-objects/) | Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. | `tracking-objects.track` | `open-robot-skills[tracking-objects]` |
| [transporting-objects](skills/transporting-objects/) | Move the currently-held object above a destination container and release. | — | `open-robot-skills[transporting-objects]` |

## Tools — what the robot can compute

Model-backed typed callables — **one bundle per model**, no task strategy.
Tool bundles never own subgraphs; they appear in the flat tool catalog every
subgraph agent (and every skill script) calls through `ctx.tool(...)`. Each
bundle's `SKILL.md` documents its setup: dependencies (declared in the
bundle's own `pyproject.toml`), environment variables, weights, and quirks.

| Bundle | Description | Tools | Extra |
|---|---|---|---|
| [curobo](tools/curobo/) | NVIDIA cuRobo motion planning — collision-free trajectories to grasp goalsets, transport with an attached object, constrained linear moves, single-pose planning, geometric IK, batch grasp feasibility, and joint-trajectory collision validation. | `curobo.batch_grasp_feasibility`, `curobo.plan_directed_linear`, `curobo.plan_grasp_motion`, `curobo.plan_linear`, `curobo.plan_to_grasp_poses`, `curobo.plan_to_pose`, `curobo.plan_with_grasped_object`, `curobo.solve_ik`, `curobo.validate_joint_trajectory_grasped`, `curobo.validate_joint_trajectory_robot` | `open-robot-skills[curobo]` |
| [gemini-er](tools/gemini-er/) | Open-vocabulary 2D object detection via the Gemini Robotics-ER API — one call returns pixel-space bounding boxes with labels and scores for a text query. | `gemini-er.detect` | `open-robot-skills[gemini-er]` |
| [geometry](tools/geometry/) | Pure-math 3D geometry toolbox — back-project masks and depth to point clouds, DBSCAN-filter noise, fit oriented bounding boxes, derive top-down/front grasp poses, and reconstruct collision worlds from RGB-D frames. | `geometry.build_world_config`, `geometry.compute_drop_position`, `geometry.compute_obb`, `geometry.compute_xy_distance`, `geometry.depth_to_point_cloud`, `geometry.exclude_robot_points`, `geometry.filter_and_compute_obb`, `geometry.filter_noise`, `geometry.front_grasp_from_obb`, `geometry.iou`, `geometry.mask_to_world_points`, `geometry.pixel_to_world_point`, `geometry.pose_distance`, `geometry.rotate_quat_z90`, `geometry.select_top_down_grasp`, `geometry.top_down_grasp_candidates`, `geometry.top_down_grasp_from_obb`, `geometry.transform_points` | `open-robot-skills[geometry]` |
| [grounding-dino](tools/grounding-dino/) | Grounding DINO zero-shot object detection — natural-language queries to labeled 2D bounding boxes with confidence scores. | `grounding-dino.detect` | `open-robot-skills[grounding-dino]` |
| [molmo](tools/molmo/) | Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). | `molmo.point_prompt`, `molmo.query`, `molmo.query_yes_no` | `open-robot-skills[molmo]` |
| [sam3](tools/sam3/) | Segment Anything 3 — text-, point-, and box-prompted instance segmentation, plus a stateful streaming video tracker that carries object identity through SAM3's memory bank. | `sam3.segment_box`, `sam3.segment_point`, `sam3.segment_text`, `sam3.tracker_close`, `sam3.tracker_init`, `sam3.tracker_update` | `open-robot-skills[sam3]` |
| [vlm](tools/vlm/) | Free-form and yes/no visual question answering against a hosted vision-language model (OpenRouter API by default; Vertex AI Gemini selectable by config). | `vlm.query`, `vlm.query_yes_no` | `open-robot-skills[vlm]` |

Both tables are generated — regenerate after adding a bundle:

```bash
uv run gap skills table --format markdown --kind skill   # and --kind tool
```

## Install

Tool and policy bundles now own their own `pyproject.toml`; install them
per-bundle with `uv run gap skills install <bundle>` (or `--all`), not via
top-level extras. From the GaP checkout:

```bash
cd ../graph-as-policy
uv sync                                       # engine + sim baseline (LIBERO included)
uv run gap skills install --all               # every bundle in this registry
uv run gap skills install sam3 grounding-dino geometry   # …or pick a subset
CUDA_HOME=/usr/local/cuda uv run gap skills install curobo  # CuRobo compiles CUDA at install
uv run gap skills check --download           # verify bundles + prefetch model weights
```

`uv.lock` pins the exact environment of the acceptance benchmark run. Two
deliberate knobs keep it solvable (documented in `pyproject.toml`): sam3's
over-strict `numpy==1.26` metadata pin is relaxed via
`override-dependencies`, and `nvidia-curobo` builds unisolated against the
environment's torch.

## Verify a checkout

```bash
uv run gap skills check             # per-bundle PASS/WARN/FAIL; non-zero exit on FAIL
uv run gap skills check --download  # + prefetch model weights (HF_TOKEN for gated repos)
uv run gap skills table             # the catalog above (--format markdown|json)
uv run gap check                    # capability report: which bundles can run HERE
uv run gap tools list               # the flat tool catalog with live schemas
```

`gap skills check` runs the engine-side format validation (frontmatter
shape per bundle kind, referenced resource paths, `allowed_tools`
resolution, declared type names, one extra per bundle) plus an import
probe that maps missing dependencies to their install line — the same
install-verification step the GaP quickstart runs. Missing weights are a
WARN, not a FAIL — nothing installs behind your back.

`gap check` answers the operational question instead: per bundle, are the
deps importable, the declared `gap.requires:` met (GPU, env vars), the
weights cached — and which skills are therefore runnable right now, with
a fix hint per failure.

This repo is one **skill registry** — the canonical example. GaP merges
any number of them by precedence (your lab's fork can shadow individual
bundles here): `gap registry init` scaffolds a new one, `gap registry add
<name> <path>` layers it on top, `gap registry list` shows the active
set. See the engine's
[docs/skills.md](https://github.com/graph-robots/graph-as-policy/blob/main/docs/skills.md).

## Contributing a bundle

**One bundle = one directory = one PR.** Scaffold it:

```bash
uv run gap skills new my-skill --kind skill    # or --kind tool
```

That creates the layout (`SKILL.md` + `scripts/` or `tools.py`) **and a
unit-test skeleton** (`tests/test_my_skill.py`); then:

1. **Write the `SKILL.md`.** Spec frontmatter (`name` == dirname,
   third-person `description` ending in a "Use when…" sentence — it is the
   coordinator's entire view of your bundle), GaP extensions under the
   `gap:` key (`allowed_tools`, `exit_conditions`, `canonical_scripts`, …).
2. **Declare operational requirements** — a `gap.requires:` block
   (`{gpu: true, env: [MY_API_KEY], env_any: [...], weights: true}`;
   `requires: {}` when it needs nothing — mandatory for tool bundles, the
   test suite enforces it). `gap check` derives runnability from this.
   Bundles that download weights may add a filesystem-only
   `weights_cached() -> bool | None` next to `prefetch()` in `tools.py`.
3. **Declare dependencies once** — add one extra named after your bundle in
   `pyproject.toml` (empty list if it has none) and run `uv lock`.
4. **Lazy-load models.** Importing your `tools.py` must not import
   torch/transformers; load weights on first call (the test suite enforces
   this).
5. **Test it without hardware** using `gap.testing` (`FakeContext`,
   `make_test_observation`) — flesh out the scaffolded test;
   `uv run gap skills test my-skill` (or plain `uv run pytest tests -q`)
   must stay green on a workstation without a robot or accelerator;
   model-touching smokes go behind the `gpu` marker.
6. **Check yourself before the PR:**

   ```bash
   uv run gap skills check && uv run gap skills test my-skill && uv run pytest tests -q
   ```

The full authoring guide — bundle anatomy, the skill-facing `ctx` API,
exit-condition design, streaming skills — is in
[gap/docs/skills.md](https://github.com/graph-robots/graph-as-policy/blob/main/docs/skills.md).

## Use with Claude Code

[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json) indexes
every bundle, so this checkout doubles as a Claude Code plugin marketplace:
the same `SKILL.md` files that drive GaP's graph generation are loadable as
agent skills.

To drive **GaP itself** from Claude Code — search these registries, check
capabilities, run/generate graphs, author new tested bundles — install the
engine's agent skill from the GaP repo (it also re-exports this registry's
bundles, so one marketplace covers both):

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap
```

## License

MIT for this repo's code and docs; model weights and the pinned upstream
packages (SAM3, Grounding DINO, cuRobo, …) keep their own licenses — see
[gap/NOTICE.md](https://github.com/graph-robots/graph-as-policy/blob/main/NOTICE.md) for the attribution table.
