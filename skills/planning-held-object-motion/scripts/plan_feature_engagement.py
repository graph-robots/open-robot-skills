"""Canonical relation-to-engagement planner from planning-held-object-motion."""
from typing import Any, TypedDict
from gap import NodeContext

class Output(TypedDict):
    placement_plan: dict[str, Any]

def run(ctx: NodeContext, approach_pose: dict[str, Any], engaged_pose: dict[str, Any],
        mate_pose: dict[str, Any], relation: str, world_config: dict[str, Any],
        attached_object: dict[str, Any]) -> Output:
    del ctx
    supported = {"loop_over_shaft", "shaft_into_aperture", "tip_through_aperture",
                 "insert_through", "feature_to_fixture"}
    if relation not in supported:
        raise ValueError(f"unsupported feature relation {relation!r}")
    waypoints = [{"pose": approach_pose, "mode": "planned_joint"}]
    if relation == "loop_over_shaft":
        # The engaged pose is on the board side of the hook tip.  It is a
        # crossing motion, not a seating motion: stopping on first contact can
        # leave the loop balanced against the distal tip without enclosing the
        # shaft.  Track this short segment explicitly, then use contact-aware
        # motion only while lowering the already-crossed loop onto the shaft.
        waypoints.append({"pose": engaged_pose, "mode": "cartesian_cross"})
        waypoints.append({"pose": mate_pose, "mode": "contact_seat"})
    elif relation == "feature_to_fixture":
        waypoints.append({"pose": engaged_pose, "mode": "contact_seat"})
        waypoints.append({"pose": mate_pose, "mode": "contact_seat"})
    else:
        waypoints.append({"pose": engaged_pose, "mode": "planned_linear",
                          "allow_goal_contact": True})
        waypoints.append({"pose": mate_pose, "mode": "planned_linear",
                          "allow_goal_contact": True})
    return {"placement_plan": {"relation": relation, "waypoints": waypoints,
                               "world_config": world_config,
                               "attached_object": attached_object}}
