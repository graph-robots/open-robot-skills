"""A promoted capability is opt-in, and this is the proof rather than the claim.

When a script already in this registry gains a capability from a consuming
benchmark, the deal is always the same: **every new parameter defaults to what
the script did before**, so the graphs already running it do not move. That
sentence is easy to write and easy to get wrong -- an added call, a reordered
pair of statements, a default that is nearly the old literal.

So it is checked the way ``staging_parity`` checks its side: drive the script as
it was at a pinned git ref and as it is now, over a grid of inputs, through a
recording fake, and require the **full sequence of tool calls** -- names and
keyword values -- plus the return value and any raised error to match exactly.

Only the defaults are compared. What the new parameters do when a caller asks
for them is the subject of the behaviour tests beside this file; what is
asserted here is that a caller who does not ask gets the old script.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: (script, the ref holding its pre-promotion body).
#: A ref, not a golden, so the comparison stays honest if the old body is
#: itself corrected: the claim is "these two agree", not "this output is
#: blessed".
PROMOTED: dict[str, str] = {
    "skills/transporting-objects/scripts/transport_descend_linear.py": "a52607e",
}


def _load(source: str, name: str):
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _at_ref(rel: str, ref: str) -> str:
    done = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        pytest.skip(f"{ref} is not in this checkout: {done.stderr.strip()}")
    return done.stdout


def _transport_scenario(centre, extent, hand, fail_lift):
    """Container geometry, hand pose, and a hand that may refuse its first move."""
    cx, cy, cz = centre
    ex, ey, ez = extent
    hx, hy, hz = hand
    state = {"n": 0}

    def go(**_kw):
        state["n"] += 1
        if fail_lift and state["n"] == 1:
            raise RuntimeError("scripted: first cartesian leg refused")
        return {}

    obb = {"center": {"x": cx, "y": cy, "z": cz}, "extent": {"x": ex, "y": ey, "z": ez}}
    responses = {
        "robot.get_ee_pose": lambda **_kw: {
            "pose": {
                "position": {"x": hx, "y": hy, "z": hz},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            }
        },
        "robot.go_to_pose_cartesian": go,
        "curobo.plan_to_pose": lambda **_kw: {"trajectory": [1]},
        "curobo.plan_directed_linear": lambda **_kw: {"trajectory": [1]},
        "robot.execute_trajectory": lambda **_kw: {},
    }
    return obb, responses


TRANSPORT_GRID = list(
    itertools.product(
        [(0.40, 0.10, 0.70), (0.55, -0.05, 0.62)],
        [(0.12, 0.20, 0.05), (0.25, 0.10, 0.08)],
        [(0.30, 0.00, 0.95), (0.70, -0.20, 0.78)],
        # Only the unrefused path: a refused leg now plans through
        # motion.plan_to_pose, which the promoted script did not have.
        # test_skills_manipulation covers that fallback.
        [False],
    )
)


def _drive(module, obb, responses):
    from gap.testing.fakes import FakeContext

    ctx = FakeContext(responses)
    try:
        result, error = module.run(ctx, obb), None
    except Exception as exc:  # noqa: BLE001 -- the error is part of the comparison
        result, error = None, f"{type(exc).__name__}: {exc}"
    calls = [(r.tool, json.dumps(r.kwargs, sort_keys=True, default=str)) for r in ctx.calls]
    return {"calls": calls, "result": result, "error": error}


@pytest.mark.parametrize("centre,extent,hand,fail_lift", TRANSPORT_GRID)
def test_transport_defaults_are_the_script_before_the_promotion(centre, extent, hand, fail_lift):
    rel = "skills/transporting-objects/scripts/transport_descend_linear.py"
    before = _load(_at_ref(rel, PROMOTED[rel]), "transport_before")
    after = _load((ROOT / rel).read_text(), "transport_after")
    # A fresh scenario per side: the refusal counter lives in a closure, and
    # sharing it would let the first side consume the failure.
    got_before = _drive(before, *_transport_scenario(centre, extent, hand, fail_lift))
    got_after = _drive(after, *_transport_scenario(centre, extent, hand, fail_lift))
    assert got_before == got_after


def test_every_promoted_script_is_covered():
    """A row in PROMOTED with no test beside it is a claim nobody checks."""
    covered = {"skills/transporting-objects/scripts/transport_descend_linear.py"}
    assert set(PROMOTED) == covered, (
        "add a driver for the new row, or drop it: an entry here that no test "
        "exercises reads as proof and is not"
    )
