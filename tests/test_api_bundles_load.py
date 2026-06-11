"""Loader tests for the API-backed (zero-GPU) tool bundles: vlm, gemini-er, molmo.

Covers loader discovery, @tool-name agreement with the SKILL.md ``gap.tools``
frontmatter, schema extraction, and lazy-import hygiene (importing the
bundles must not pull in google-genai).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from gap.skills import load_skills
from gap.tools import ToolRegistry
from gap.tools._registry import _PENDING_TOOLS

ROOT = Path(__file__).resolve().parents[1]

API_BUNDLES = ["gemini-er", "molmo", "vlm"]

EXPECTED_TOOLS: dict[str, list[str]] = {
    "vlm": ["vlm.query", "vlm.query_yes_no"],
    "gemini-er": ["gemini-er.detect"],
    "molmo": ["molmo.point_prompt", "molmo.query", "molmo.query_yes_no"],
}

EXPECTED_INPUTS: dict[str, list[str]] = {
    "vlm.query": ["prompt", "image", "images", "provider", "model"],
    "vlm.query_yes_no": ["prompt", "image", "images", "provider", "model"],
    "gemini-er.detect": ["image", "query", "model"],
    "molmo.point_prompt": ["image", "query"],
    "molmo.query": ["prompt", "image"],
    "molmo.query_yes_no": ["prompt", "image"],
}

EXPECTED_OUTPUTS: dict[str, list[str]] = {
    "vlm.query": ["text"],
    "vlm.query_yes_no": ["answer", "text"],
    "gemini-er.detect": ["detections"],
    "molmo.point_prompt": ["pixel_x", "pixel_y", "found"],
    "molmo.query": ["text"],
    "molmo.query_yes_no": ["answer", "text"],
}

ALL_TOOL_NAMES = {n for names in EXPECTED_TOOLS.values() for n in names}


#: Heavy modules the API bundles must never import (zero-GPU contract).
HEAVY_MODULES = ("google.genai", "torch", "grpc", "transformers")


@pytest.fixture(scope="module")
def loaded():
    """A fresh (registry, tool_registry, heavy_added) triple for the bundles.

    Forces a re-import of each bundle's ``tools.py`` so the @tool decorators
    re-append to the pending list even if another test module already loaded
    the bundles (module imports are cached process-wide). ``heavy_added`` is
    the set of heavy modules the re-import pulled into ``sys.modules``.
    """
    for bundle in API_BUNDLES:
        sys.modules.pop(f"gap_skills.tools.{bundle}.tools", None)
    # Drop any stale pending entries for our names so the fresh import's
    # registrations don't collide.
    _PENDING_TOOLS[:] = [e for e in _PENDING_TOOLS if e["name"] not in ALL_TOOL_NAMES]

    heavy_before = {m for m in HEAVY_MODULES if m in sys.modules}
    registry = load_skills(ROOT, only=API_BUNDLES)
    heavy_added = {m for m in HEAVY_MODULES if m in sys.modules} - heavy_before

    tool_registry = ToolRegistry()
    tool_registry.discover_pending()
    return registry, tool_registry, heavy_added


# ---------------------------------------------------------------------------
# Loader discovery
# ---------------------------------------------------------------------------


def test_loader_discovers_api_bundles(loaded):
    registry, _, _ = loaded
    assert sorted(i.name for i in registry.list_skills()) == sorted(API_BUNDLES)
    for bundle in API_BUNDLES:
        info = registry.get(bundle)
        assert info.kind == "tool"
        assert info.namespace == "tools"
        assert info.tools_module is not None, f"{bundle} tools.py did not import"
        assert info.meta.category == "perception"


def test_skill_md_tools_frontmatter_matches_registrations(loaded):
    registry, tool_registry, _ = loaded
    for bundle, expected in EXPECTED_TOOLS.items():
        meta_tools = registry.get(bundle).meta.tools
        # SKILL.md `gap.tools` is the documented catalog…
        assert sorted(meta_tools) == sorted(expected)
        # …and every documented tool has a non-empty summary that matches the
        # @tool registration in tools.py.
        for name in expected:
            assert name in tool_registry, f"{name} not registered via @tool"
            descriptor = tool_registry.get(name)
            assert meta_tools[name] == descriptor.summary
            assert descriptor.summary


# ---------------------------------------------------------------------------
# gap.tools registration: names, tags, scope, schema extraction
# ---------------------------------------------------------------------------


def test_descriptors_have_perception_tag_and_runtime_scope(loaded):
    _, tool_registry, _ = loaded
    for name in ALL_TOOL_NAMES:
        descriptor = tool_registry.get(name)
        assert descriptor.tags == ("perception",)
        assert descriptor.scope == "runtime"
        assert descriptor.transport == "python"


def test_schema_extraction_inputs_and_outputs(loaded):
    _, tool_registry, _ = loaded
    for name, expected_inputs in EXPECTED_INPUTS.items():
        schema = tool_registry.get(name).schema
        assert list(schema.inputs) == expected_inputs, name
        assert list(schema.outputs) == EXPECTED_OUTPUTS[name], name

    # Spot-check typing detail: image params are optional where the proto had
    # an optional image, required for the localization tools.
    assert tool_registry.get("vlm.query").schema.inputs["image"].required is False
    assert tool_registry.get("molmo.query").schema.inputs["image"].required is False
    assert tool_registry.get("molmo.point_prompt").schema.inputs["image"].required is True
    assert tool_registry.get("gemini-er.detect").schema.inputs["image"].required is True
    assert tool_registry.get("gemini-er.detect").schema.inputs["query"].required is True


def test_tools_dispatchable_through_registry(loaded):
    """The registry can invoke a bundle tool end-to-end (cheap path only)."""
    _, tool_registry, _ = loaded
    # molmo with no base URL raises the configured ToolError through dispatch —
    # proves the adapter wiring without any network.
    import os

    from gap.errors import ToolError
    saved = os.environ.pop("GAP_MOLMO_BASE_URL", None)
    try:
        import numpy as np
        with pytest.raises(ToolError, match="GAP_MOLMO_BASE_URL"):
            tool_registry.invoke(
                "molmo.point_prompt",
                image=np.zeros((2, 2, 3), dtype=np.uint8),
                query="cup",
            )
    finally:
        if saved is not None:
            os.environ["GAP_MOLMO_BASE_URL"] = saved


# ---------------------------------------------------------------------------
# Lazy-import hygiene
# ---------------------------------------------------------------------------


def test_bundle_import_pulls_no_heavy_modules(loaded):
    """Importing the bundles is lazy: no google-genai (must stay a call-time
    import), and — zero-GPU contract — no torch / grpc / transformers.

    The fixture measured exactly what the bundles' (re-)import added to
    ``sys.modules``, so this holds regardless of what other test modules in
    the session may have imported.
    """
    _, _, heavy_added = loaded
    assert heavy_added == set()
