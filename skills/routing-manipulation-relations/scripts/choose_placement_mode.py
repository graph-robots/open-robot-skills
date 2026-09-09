"""Route a typed manipulation relation to a declared execution branch."""

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    mode: str


_DEFAULT_ROUTES = (
    {
        "mode": "fixture",
        "relations": ("loop_over_shaft", "feature_to_fixture"),
    },
    {
        "mode": "drop",
        "relations": (
            "shaft_into_aperture",
            "tip_through_aperture",
            "insert_through",
            "feature_into_container",
        ),
    },
)


def run(ctx: NodeContext, relation: str,
        relation_routes: list[dict[str, Any]] | None = None) -> Output:
    del ctx
    routes = relation_routes or list(_DEFAULT_ROUTES)
    matches = [str(route["mode"]) for route in routes
               if relation in {str(item) for item in route.get("relations", [])}]
    if len(matches) != 1:
        qualifier = "unsupported" if not matches else "ambiguous"
        raise ValueError(f"{qualifier} manipulation relation {relation!r}")
    return {"mode": matches[0]}
