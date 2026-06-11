<div align="center">

# open-robot-skills

**What your robot can do, one directory at a time.**

A curated, contributable library of manipulation **skills** and model-backed
**tool bundles** for [gap — graph as policy](https://github.com/graph-robots/graph-as-policy), in the
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
every gap command — `gap run`, `gap generate`, `gap benchmark` — discovers
it automatically; no flags, no registration.

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
`exit_conditions` are what downstream routing keys on.

| Bundle | Description | Tools | Extra |
|---|---|---|---|
| [grasping-direct-ik](skills/grasping-direct-ik/) | Direct IK align-then-descend grasping. | — | `open-robot-skills[grasping-direct-ik]` |
| [grasping-short-axis](skills/grasping-short-axis/) | Deterministic short-axis-aligned grasp with CuRobo. | — | `open-robot-skills[grasping-short-axis]` |
| [grasping-with-planner](skills/grasping-with-planner/) | Collision-aware grasping using cuRobo trajectory planning over a per-observation collision world. | — | `open-robot-skills[grasping-with-planner]` |
| [perceiving-object-parts](skills/perceiving-object-parts/) | Hierarchical perception for subpart targeting. | — | `open-robot-skills[perceiving-object-parts]` |
| [perceiving-objects](skills/perceiving-objects/) | Fast single-path 3D object perception. | — | `open-robot-skills[perceiving-objects]` |
| [perceiving-objects-multiview](skills/perceiving-objects-multiview/) | Robust three-method 3D object perception. | — | `open-robot-skills[perceiving-objects-multiview]` |
| [perceiving-objects-oneshot](skills/perceiving-objects-oneshot/) | Lightweight one-shot 3D object perception. | — | `open-robot-skills[perceiving-objects-oneshot]` |
| [running-policies](skills/running-policies/) | Run a learned VLA policy in closed loop until a termination signal fires or max_windows is reached. | `running-policies.run` | `open-robot-skills[running-policies]` |
| [tracking-objects](skills/tracking-objects/) | Long-running skill that drives the SAM3 tracker from the graph-scoped observation stream. | `tracking-objects.track` | `open-robot-skills[tracking-objects]` |
| [transporting-objects](skills/transporting-objects/) | Move the currently-held object above a destination container and release. | — | `open-robot-skills[transporting-objects]` |

## Tools — what the robot can compute

Model-backed typed callables — **one bundle per model**, no task strategy.
Tool bundles never own subgraphs; they appear in the flat tool catalog every
subgraph agent (and every skill script) calls through `ctx.tool(...)`. Each
bundle's `SKILL.md` documents its setup: dependencies (one extra per
bundle), environment variables, weights, and quirks.

| Bundle | Description | Tools | Extra |
|---|---|---|---|
| [curobo](tools/curobo/) | NVIDIA cuRobo motion planning — collision-free trajectories to grasp goalsets, transport with an attached object, constrained linear moves, single-pose planning, geometric IK, batch grasp feasibility, and joint-trajectory collision validation. | `curobo.batch_grasp_feasibility`, `curobo.plan_directed_linear`, `curobo.plan_grasp_motion`, `curobo.plan_linear`, `curobo.plan_to_grasp_poses`, `curobo.plan_to_pose`, `curobo.plan_with_grasped_object`, `curobo.solve_ik`, `curobo.validate_joint_trajectory_grasped`, `curobo.validate_joint_trajectory_robot` | `open-robot-skills[curobo]` |
| [gemini-er](tools/gemini-er/) | Open-vocabulary 2D object detection via the Gemini Robotics-ER API — one call returns pixel-space bounding boxes with labels and scores for a text query. | `gemini-er.detect` | `open-robot-skills[gemini-er]` |
| [geometry](tools/geometry/) | Pure-math 3D geometry toolbox — back-project masks and depth to point clouds, DBSCAN-filter noise, fit oriented bounding boxes, derive top-down/front grasp poses, and reconstruct collision worlds from RGB-D frames. | `geometry.build_world_config`, `geometry.compute_drop_position`, `geometry.compute_obb`, `geometry.compute_xy_distance`, `geometry.depth_to_point_cloud`, `geometry.exclude_robot_points`, `geometry.filter_and_compute_obb`, `geometry.filter_noise`, `geometry.front_grasp_from_obb`, `geometry.iou`, `geometry.mask_to_world_points`, `geometry.pixel_to_world_point`, `geometry.pose_distance`, `geometry.rotate_quat_z90`, `geometry.select_top_down_grasp`, `geometry.top_down_grasp_candidates`, `geometry.top_down_grasp_from_obb`, `geometry.transform_points` | `open-robot-skills[geometry]` |
| [grounding-dino](tools/grounding-dino/) | Grounding DINO zero-shot object detection — natural-language queries to labeled 2D bounding boxes with confidence scores. | `grounding-dino.detect` | `open-robot-skills[grounding-dino]` |
| [molmo](tools/molmo/) | Visual pointing and Q&A via the Molmo VLM served from a self-hosted vLLM endpoint (OpenAI-compatible API). | `molmo.point_prompt`, `molmo.query`, `molmo.query_yes_no` | `open-robot-skills[molmo]` |
| [sam3](tools/sam3/) | Segment Anything 3 — text-, point-, and box-prompted instance segmentation, plus a stateful streaming video tracker that carries object identity through SAM3's memory bank. | `sam3.segment_box`, `sam3.segment_point`, `sam3.segment_text`, `sam3.tracker_close`, `sam3.tracker_init`, `sam3.tracker_update` | `open-robot-skills[sam3]` |
| [vlm](tools/vlm/) | Free-form and yes/no visual question answering against a hosted vision-language model (Anthropic API by default; OpenAI-compatible endpoints and Vertex AI selectable by config). | `vlm.query`, `vlm.query_yes_no` | `open-robot-skills[vlm]` |

Both tables are generated — regenerate after adding a bundle:

```bash
uv run gap skills table --format markdown --kind skill   # and --kind tool
```

## Install

Dependencies are declared **per bundle** as a pip extra of this repo (extra
name == bundle name), so installs are declarative and one resolver run
surfaces cross-bundle conflicts. The common case is the curated sets from
the gap checkout:

```bash
cd ../graph-as-policy
uv sync --extra quickstart   # sam3 + grounding-dino + geometry (+ engine + sim)
uv sync --extra grocery      # + curobo (the acceptance-benchmark set; needs CUDA_HOME)
uv sync --extra all          # everything
uv run gap skills check --download    # verify bundles + prefetch model weights
```

Working on this repo standalone, sync it directly (the engine resolves from
the sibling `../graph-as-policy` checkout):

```bash
uv sync --extra sam3                                  # any single bundle…
CUDA_HOME=/usr/local/cuda uv sync --extra curobo      # …CuRobo compiles CUDA at install
uv sync --extra all
```

`uv.lock` pins the exact environment of the acceptance benchmark run. Two
deliberate knobs keep it solvable (documented in `pyproject.toml`): sam3's
over-strict `numpy==1.26` metadata pin is relaxed via
`override-dependencies`, and `nvidia-curobo` builds unisolated against the
environment's torch. pip works too: `pip install -e ../graph-as-policy -e ".[<extra>]"`
(add `--no-build-isolation` for curobo).

## Verify a checkout

```bash
uv run gap skills check             # per-bundle PASS/WARN/FAIL; non-zero exit on FAIL
uv run gap skills check --download  # + prefetch model weights (HF_TOKEN for gated repos)
uv run gap skills table             # the catalog above (--format markdown|json)
```

`check` runs the engine-side format validation (frontmatter shape per
bundle kind, referenced resource paths, `allowed_tools` resolution,
declared type names, one extra per bundle) plus an import probe that maps
missing dependencies to their install line. Missing weights are a WARN,
not a FAIL — nothing installs behind your back.

## Contributing a bundle

**One bundle = one directory = one PR.** Scaffold it:

```bash
uv run gap skills new my-skill --kind skill    # or --kind tool
```

That creates the layout (`SKILL.md` + `scripts/` or `tools.py`); then:

1. **Write the `SKILL.md`.** Spec frontmatter (`name` == dirname,
   third-person `description` ending in a "Use when…" sentence — it is the
   coordinator's entire view of your bundle), gap extensions under the
   `gap:` key (`allowed_tools`, `exit_conditions`, `canonical_scripts`, …).
2. **Declare dependencies once** — add one extra named after your bundle in
   `pyproject.toml` (empty list if it has none) and run `uv lock`.
3. **Lazy-load models.** Importing your `tools.py` must not import
   torch/transformers; load weights on first call (the test suite enforces
   this).
4. **Test it CPU-only** with `gap.testing` (`FakeContext`,
   `make_test_observation`) — `uv run pytest tests -q` must stay green
   without a GPU; model-touching smokes go behind the `gpu` marker.
5. **Check yourself before the PR:**

   ```bash
   uv run gap skills check && uv run pytest tests -q
   ```

The full authoring guide — bundle anatomy, the skill-facing `ctx` API,
exit-condition design, streaming skills — is in
[gap/docs/skills.md](https://github.com/graph-robots/graph-as-policy/blob/main/docs/skills.md).

## Use with Claude Code

[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json) indexes
every bundle, so this checkout doubles as a Claude Code plugin marketplace:
the same `SKILL.md` files that drive gap's graph generation are loadable as
agent skills.

## License

MIT for this repo's code and docs; model weights and the pinned upstream
packages (SAM3, Grounding DINO, cuRobo, …) keep their own licenses — see
[gap/NOTICE.md](https://github.com/graph-robots/graph-as-policy/blob/main/NOTICE.md) for the attribution table.
