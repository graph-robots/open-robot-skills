"""Deterministic relation validation and candidate assignment."""

from __future__ import annotations

from itertools import permutations
from typing import Any, TypedDict

import numpy as np


class Output(TypedDict):
    status: str
    pairs: list[dict[str, Any]]


def _center(candidate: dict[str, Any]) -> np.ndarray:
    center = candidate["obb"]["center"]
    return np.array([center["x"], center["y"], center["z"]], dtype=float)


def _pair(source: dict[str, Any], destination: dict[str, Any], confidence: float) -> dict[str, Any]:
    sid, did = str(source["id"]), str(destination["id"])
    return {"pair_id": f"pair:{sid}->{did}", "source_id": sid,
            "destination_id": did, "source": source, "destination": destination,
            "confidence": float(confidence)}


def run(ctx, sources: list[dict[str, Any]], destinations: list[dict[str, Any]],
        relation: dict[str, Any], confidence: float = 1.0) -> Output:
    del ctx
    if not sources:
        return {"status": "finished", "pairs": []}
    if not destinations:
        return {"status": "not_found", "pairs": []}
    op = str(relation.get("operator", ""))
    if op == "shared_destination":
        if len(destinations) != 1:
            return {"status": "cardinality_mismatch", "pairs": []}
        return {"status": "paired", "pairs": [_pair(s, destinations[0], confidence) for s in sources]}
    if len(sources) != len(destinations):
        return {"status": "cardinality_mismatch", "pairs": []}
    if op == "corresponding_order":
        axis = np.asarray(relation.get("axis", [0, 1, 0]), dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return {"status": "ambiguous", "pairs": []}
        axis /= norm
        direction = 1.0 if float(relation.get("direction", 1)) >= 0 else -1.0
        ss = sorted(sources, key=lambda x: direction * float(_center(x) @ axis))
        dd = sorted(destinations, key=lambda x: direction * float(_center(x) @ axis))
        return {"status": "paired", "pairs": [_pair(s, d, confidence) for s, d in zip(ss, dd)]}
    if op == "nearest":
        best = min(permutations(destinations), key=lambda order: sum(
            float(np.linalg.norm(_center(s) - _center(d))) for s, d in zip(sources, order)))
        return {"status": "paired", "pairs": [_pair(s, d, confidence) for s, d in zip(sources, best)]}
    ids = relation.get("pairs") or []
    source_by_id = {str(x["id"]): x for x in sources}
    destination_by_id = {str(x["id"]): x for x in destinations}
    if len(ids) != len(sources):
        return {"status": "ambiguous", "pairs": []}
    try:
        pairs = [_pair(source_by_id[str(x["source_id"])],
                       destination_by_id[str(x["destination_id"])], confidence) for x in ids]
    except KeyError:
        return {"status": "ambiguous", "pairs": []}
    if len({x["source_id"] for x in pairs}) != len(pairs) or len({x["destination_id"] for x in pairs}) != len(pairs):
        return {"status": "ambiguous", "pairs": []}
    return {"status": "paired", "pairs": pairs}
