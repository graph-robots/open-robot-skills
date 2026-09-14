"""Cross-cutting contract every ``kind: skill`` bundle in this checkout keeps.

``gap skills check`` validates one bundle's frontmatter against the loader's
rules. These tests hold the lines the checker does not, or only warns on:

- every canonical script's ``run()`` is typed: ``ctx`` first, an annotation on
  every other parameter, a return annotation, and — ratcheted — a TypedDict
  return so the registry exposes an output schema;
- ``gap.requires.connector`` names only what nothing here provides: not a
  generic connector tool and not any bundle's ``gap.tools`` (the checker
  warns; here it is an error);
- the ``sim-only`` tag is present exactly when an allowed tool is a
  ``sim.*`` / ``cable.*`` name;
- no script under ``skills/*/scripts`` refers to the tree the bundles were
  promoted from (``robosimstudio``, ``tasks/``, ``/home/``).
"""

from __future__ import annotations

import inspect
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

SIM_PREFIXES = ("sim.", "cable.")
FORBIDDEN = ("robosimstudio", "tasks/", "/home/")

#: ``bundle::script`` whose ``run()`` returns a bare ``dict`` rather than a
#: TypedDict, so the registry sees no output schema for it. A shrinking
#: ratchet: give one a TypedDict ``Output`` and delete its line; never add one.
UNTYPED_RETURN_RATCHET = frozenset(
    {
        "perceiving-next-item::decide_next_item",
    }
)


def _skill_bundles(skills_registry):
    return skills_registry.list_skills(kind="skill")


def _scripts(skills_registry):
    for info in _skill_bundles(skills_registry):
        for name, script in info.canonical_scripts.items():
            yield f"{info.name}::{name}", script


def test_canonical_scripts_have_typed_run_signatures(skills_registry):
    problems = []
    for label, script in _scripts(skills_registry):
        run = getattr(script.module, "run", None)
        if run is None or not callable(run):
            problems.append(f"{label}: no run()")
            continue
        sig = inspect.signature(run)
        params = list(sig.parameters.values())
        if not params or params[0].name != "ctx":
            problems.append(f"{label}: run()'s first parameter must be ctx")
        for p in params[1:]:
            if p.annotation is inspect.Parameter.empty:
                problems.append(f"{label}: parameter {p.name!r} is unannotated")
        if sig.return_annotation is inspect.Signature.empty:
            problems.append(f"{label}: run() has no return annotation")
    assert not problems, "\n".join(problems)


def test_canonical_scripts_expose_an_output_schema(skills_registry):
    untyped = {label for label, script in _scripts(skills_registry) if not script.schema.outputs}
    new = sorted(untyped - UNTYPED_RETURN_RATCHET)
    assert not new, (
        f"run() returns without a TypedDict, so the registry exposes no outputs for: {new}"
    )


def test_connector_requirements_are_disjoint_from_provided_tools(skills_registry):
    from gap.skills.validate import connector_tool_names

    provided = set(connector_tool_names())
    for info in skills_registry.list_skills():
        provided.update(info.meta.tools)
    stale = {}
    for info in _skill_bundles(skills_registry):
        req = info.meta.requires
        if req is None:
            continue
        overlap = sorted(set(req.connector) & provided)
        if overlap:
            stale[info.name] = overlap
    assert not stale, (
        "requires.connector names tools a generic connector or a bundle in this "
        f"checkout already provides: {stale}"
    )


def test_sim_only_tag_tracks_sim_and_cable_tools(skills_registry):
    wrong = {}
    for info in _skill_bundles(skills_registry):
        reads_sim = sorted(t for t in info.meta.allowed_tools if t.startswith(SIM_PREFIXES))
        tagged = "sim-only" in info.meta.tags
        if bool(reads_sim) != tagged:
            wrong[info.name] = (
                f"allows {reads_sim} but is not tagged sim-only"
                if reads_sim
                else "tagged sim-only but allows no sim.*/cable.* tool"
            )
    assert not wrong, wrong


def test_scripts_do_not_reference_the_source_tree():
    hits = []
    for path in sorted(SKILLS_DIR.glob("*/scripts/**/*.py")):
        lowered = path.read_text(encoding="utf-8").lower()
        for needle in FORBIDDEN:
            if needle in lowered:
                hits.append(f"{path.relative_to(SKILLS_DIR.parent)}: {needle!r}")
    assert not hits, "\n".join(hits)
