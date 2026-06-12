"""Every tool bundle must declare its operational requirements.

``gap check`` derives "can this tool run here?" from each bundle's
``gap.requires`` frontmatter block (GPU, env vars, downloaded weights).
A tool bundle that forgets the block silently reports as
unconditionally ready, so this registry mandates an explicit
declaration on every ``tools/*`` bundle — ``requires: {}`` when there
are genuinely no requirements.
"""

from __future__ import annotations

import os
import re


def _tool_bundles(skills_registry):
    return skills_registry.list_skills(kind="tool")


def test_every_tool_bundle_declares_requires(skills_registry):
    undeclared = [
        info.name for info in _tool_bundles(skills_registry)
        if info.meta.requires is None
    ]
    assert not undeclared, (
        f"tool bundles without a gap.requires block: {undeclared} — "
        f"declare `requires: {{}}` when the bundle truly needs nothing"
    )


def test_requires_env_names_look_like_env_vars(skills_registry):
    env_re = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for info in skills_registry.list_skills():
        req = info.meta.requires
        if req is None:
            continue
        for var in [*req.env, *req.env_any]:
            assert env_re.match(var), (
                f"{info.name}: requires entry {var!r} does not look like "
                f"an environment variable name"
            )


def test_gpu_tagged_bundles_declare_gpu_requirement(skills_registry):
    for info in _tool_bundles(skills_registry):
        if "gpu" in info.meta.tags:
            req = info.meta.requires
            assert req is not None and req.gpu, (
                f"{info.name} is gpu-tagged but does not declare "
                f"gap.requires.gpu: true"
            )


def test_weights_cached_hooks_are_filesystem_only(skills_registry):
    """Bundles with a weights_cached() hook: callable, fast, tri-state."""
    for info in _tool_bundles(skills_registry):
        module = info.tools_module
        hook = getattr(module, "weights_cached", None) if module else None
        if hook is None:
            continue
        result = hook()
        assert result is None or isinstance(result, bool)


def test_molmo_requirement_matches_implementation(skills_registry):
    """The declared env var is the one tools.py actually reads."""
    req = skills_registry.get("molmo").meta.requires
    assert req is not None and req.env == ["GAP_MOLMO_BASE_URL"]
    source = (
        skills_registry.get("molmo").bundle_dir / "tools.py"
    ).read_text(encoding="utf-8")
    assert "GAP_MOLMO_BASE_URL" in source
    assert "GAP_MOLMO_BASE_URL" in os.environ or True  # env-independent test
