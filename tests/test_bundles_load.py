"""Loader-level checks for the four tool bundles + repo-wide format gate.

- the checkout exposes exactly the expected tool bundles;
- every name declared under ``gap.tools:`` in a bundle's SKILL.md has a
  matching ``@tool`` registration from that bundle's tools.py (and nothing
  undeclared leaks out);
- schema extraction works on every tool (no empty-schema fallback);
- importing the bundles is lazy: no torch / transformers / curobo / sam3
  in ``sys.modules`` after the loader has imported every tools.py;
- every bundle in the repo passes the engine-side format validation that
  ``gap skills check`` runs (:mod:`gap.skills.validate` — the rules live in
  the gap engine so third-party checkouts get the same checker).
"""

from __future__ import annotations

import sys
from pathlib import Path

SKILLS_ROOT = Path(__file__).resolve().parents[1]

TOOL_BUNDLES = ("curobo", "geometry", "grounding-dino", "sam3")

#: Heavy ML packages that must only load inside tool function bodies.
_FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "curobo", "sam3"}


def test_four_tool_bundles_discovered(skills_registry):
    names = {info.name for info in skills_registry.list_skills(kind="tool")}
    assert set(TOOL_BUNDLES) <= names
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        assert info.kind == "tool"
        assert info.namespace == "tools"
        assert info.tools_module is not None, f"{bundle}: tools.py did not import"
        assert info.meta.tools, f"{bundle}: SKILL.md declares no gap.tools"


def test_lazy_import_no_heavy_deps_in_sys_modules(skills_registry):
    """Importing every bundle's tools.py must not pull GPU/ML deps."""
    loaded_roots = {name.split(".")[0] for name in sys.modules}
    leaked = _FORBIDDEN_TOP_LEVEL & loaded_roots
    assert not leaked, f"bundle import pulled heavy deps eagerly: {sorted(leaked)}"


def test_declared_tools_match_registered_tools(skills_registry, tool_registry):
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        declared = set(info.meta.tools)

        # Names follow the <bundle>.<func> convention.
        for name in declared:
            assert name.startswith(f"{bundle}."), (
                f"{bundle}: declared tool {name!r} not namespaced under the bundle"
            )

        registered = {
            name
            for name, desc in tool_registry.runtime_tools().items()
            if desc.metadata.get("module") == f"gap_skills.tools.{bundle}.tools"
        }
        assert declared == registered, (
            f"{bundle}: SKILL.md gap.tools and @tool registrations disagree; "
            f"declared-only={sorted(declared - registered)}, "
            f"registered-only={sorted(registered - declared)}"
        )


def test_schema_extraction_works_on_every_tool(skills_registry, tool_registry):
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        for name in info.meta.tools:
            desc = tool_registry.get(name)
            assert desc.summary, f"{name}: empty summary"
            # Every bundle tool takes at least one input; an empty inputs dict
            # means extract_schema fell back to the bare UnitSchema.
            assert desc.schema.inputs, f"{name}: schema extraction fell back/empty"
            for field in desc.schema.inputs.values():
                assert field.type_str, f"{name}: untyped input {field.name}"


def test_repo_passes_engine_format_validation():
    """Every bundle passes the `gap skills check` format rules: frontmatter
    shape per kind, referenced resources on disk, allowed_tools resolvable
    against connector + declared bundle tools, declared type names, and the
    one-pip-extra-per-bundle convention (extra name == bundle name)."""
    from gap.skills.validate import load_checkout_extras, validate_checkout

    reports = validate_checkout(SKILLS_ROOT)
    assert len(reports) >= 17, "bundle discovery regressed"

    failures = {
        r.name: [str(i) for i in r.errors] for r in reports if r.errors
    }
    assert not failures, f"format validation errors: {failures}"

    # The extras convention is a warning in the engine checker (third-party
    # checkouts may not have a pyproject); in THIS repo it is a hard rule
    # for `kind: skill` bundles only. Tool and policy bundles are exempted:
    # they live under tools/ or policies/ with their own pyproject.toml +
    # .venv/ (gap-managed via `gap skills install <name>`), so the root
    # pyproject does NOT list them as extras — the bundle's pyproject is
    # the single source of truth for its deps.
    extras = load_checkout_extras(SKILLS_ROOT)
    assert extras is not None
    missing = [
        r.name for r in reports
        if r.kind == "skill" and r.name not in extras
    ]
    assert not missing, (
        f"skill bundles without a pip extra (extra name == bundle name): {missing}"
    )


def test_tool_tags_classify_bundles(skills_registry, tool_registry):
    perception_bundles = {"sam3", "grounding-dino"}
    for bundle in TOOL_BUNDLES:
        info = skills_registry.get(bundle)
        for name in info.meta.tools:
            tags = set(tool_registry.get(name).tags)
            if bundle in perception_bundles:
                assert "perception" in tags, f"{name}: missing perception tag"
            elif bundle == "curobo":
                assert "planning" in tags, f"{name}: missing planning tag"
            else:  # geometry is judged per tool
                assert tags & {"perception", "planning"}, f"{name}: untagged"
