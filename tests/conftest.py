"""Shared fixtures: load the open-robot-skills checkout once per session.

``load_skills`` imports each bundle's ``tools.py`` through the synthetic
``gap_skills.tools.<bundle>.tools`` package, which fires the ``@tool``
decorators into ``gap.tools._registry._PENDING_TOOLS``. The decorator queue
is drained exactly once (module imports are idempotent), so both fixtures
are session-scoped and shared by every test module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The open-robot-skills checkout root (parent of tests/).
SKILLS_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def skills_registry():
    from gap.skills import load_skills

    return load_skills(SKILLS_ROOT)


@pytest.fixture(scope="session")
def tool_registry(skills_registry):
    """ToolRegistry with the bundles' pending @tool registrations drained."""
    from gap_core.tools import ToolRegistry

    reg = ToolRegistry()
    reg.discover_pending()
    return reg
