"""Release at the mate, then retreat away from the fixture without sweeping it.

**The planned retract (opt-in).** ``retract_planner=True`` asks the planner for
the retreat as an orientation-locked ``motion.plan_linear`` line (start contact
allowed: the open fingers still touch the released object) over a ladder of
distances, longest first, executes the first that plans with
``robot.execute_trajectory``, and only when every rung is refused sends the
Cartesian servo to the retreat it always used. Measured on the
``sharps_disposal`` benchmark (graph ``gap_perception_v2``, ``tray_clutter_t11``, sweep s2):
the hand ends an insertion low over the box with the shoulder near its limit --
joint 2 0.096 rad from its upper limit -- and the servo gave up 102 mm short of
a 160 mm rise, aborting the episode with the syringe already released. The
ladder the graph ran from sweep s2 through s8 is (0.16, 0.08). Off by default:
a planner call is a recorded tool call, and graphs that did not ask keep the
servo.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext

#: Minimum retreat when the caller gives no ``retract_m``.
_DEFAULT_RETRACT_M = 0.08
#: Extra clearance past the farthest attached sphere so the opened gripper
#: leaves the released object's envelope.
_CLEARANCE_MARGIN_M = 0.01
#: Relations where the held feature went *into* the fixture: retreating along
#: the fixture axis would push further in, so the retreat is reversed.
_INSERTED_RELATIONS = frozenset({"shaft_into_aperture", "tip_through_aperture", "insert_through"})
#: How much of the retreat distance is also taken straight up.
_VERTICAL_RETREAT_FRACTION = 0.5
#: Planned-retract distances, longest first, and the start-contact margin the
#: graph planned them with. Only read under ``retract_planner``.
_RETRACT_LADDER_M = (0.16, 0.08)
_RETRACT_CONTACT_MARGIN_M = 0.008


class Output(TypedDict):
    released: bool
    retreat_pose: dict[str, Any]
    release_report: dict[str, Any]


def run(
    ctx: NodeContext,
    final_pose: dict[str, Any],
    retract_m: float = 0.0,
    retreat_axis: dict[str, float] | None = None,
    attached_object: dict[str, Any] | None = None,
    relation: str = "loop_over_shaft",
    open_settle_steps: int = 80,
    settle_steps: int = 120,
    arm_id: int | None = None,
    contact_profile: dict[str, Any] | None = None,
    retract_planner: bool = False,
    retract_ladder_m: list[float] | None = None,
    retract_contact_margin_m: float = _RETRACT_CONTACT_MARGIN_M,
) -> Output:
    """Open, back off along the fixture axis, and let the object come to rest.

    ``contact_profile`` is a table for a graph that carries one profile per
    object kind rather than one node per kind -- the shape the tool-hanging
    level-2 graph uses, where a wrench and a pair of scissors want different
    settle times off the same node. **Every key is optional and every absent
    key falls through to the named argument above it**, so a caller that passes
    no profile gets exactly the behaviour this script had before the table
    existed. That is not a courtesy: eight shipped subgraphs bind this node
    without one, and ``test_promotion_is_behaviour_preserving`` compares the
    tool calls a script makes, so a default that moved would show up as a
    behaviour change in a graph nobody touched.

    ``arm_id`` follows the same rule one step further. It is threaded as an
    ABSENT keyword rather than as ``arm_id=0`` when nothing asks for an arm,
    because ``ctx.tool("robot.open_gripper", arm_id=0)`` and
    ``ctx.tool("robot.open_gripper")`` are two different recorded calls, and
    both the replay gate and ``tools/graph_ab.py`` compare names *and* keyword
    values. Single-arm graphs therefore replay byte-identically while a
    bimanual one passes a hand.

    ``retract_planner`` plans the retreat instead (see the module docstring).
    Each rung of ``retract_ladder_m`` (unset: 0.16 then 0.08 m) replaces
    ``retract_m`` as the asked distance and keeps every other rule -- the
    attached-extent floor, the axis and its reversal, the vertical lift -- so
    a rung never retreats less than the servo would. The rung distances are
    planned with ``contact_margin=retract_contact_margin_m``. When no rung
    plans (a refusal, an empty trajectory, or a planner that raises), the
    servo retreat below runs unchanged. ``release_report`` then
    also carries ``retreat_mode`` (``planned_linear`` or ``cartesian``).
    """
    profile = contact_profile or {}
    held = attached_object or {}
    arm = arm_id if arm_id is not None else held.get("arm_id")
    on_arm: dict[str, Any] = {} if arm is None else {"arm_id": int(arm)}

    release_steps = int(profile.get("release_settle_steps", open_settle_steps))
    ctx.tool("robot.open_gripper", settle_steps=release_steps, **on_arm)
    start = {
        "position": dict(final_pose["position"]),
        "rotation": dict(final_pose["rotation"]),
    }
    axis = np.array([0.0, 0.0, 1.0])
    inserted = frozenset(profile.get("reverse_retreat_relations", _INSERTED_RELATIONS))
    if retreat_axis is not None:
        axis = np.array([retreat_axis[k] for k in ("x", "y", "z")], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        if relation in inserted:
            axis = -axis
    spheres = list(held.get("spheres") or [])
    extent = max(
        (
            float(np.linalg.norm([s["center"][k] for k in ("x", "y", "z")]))
            + float(s.get("radius", 0.0))
            for s in spheres
        ),
        default=0.0,
    )
    clearance = float(profile.get("object_clearance_m", _CLEARANCE_MARGIN_M))

    def retreat_by(asked: float) -> tuple[dict[str, Any], float]:
        minimum = asked if asked > 0.0 else _DEFAULT_RETRACT_M
        distance = max(minimum, extent + clearance)
        pose = {"position": dict(start["position"]), "rotation": dict(start["rotation"])}
        for key, component in zip(("x", "y", "z"), axis, strict=True):
            pose["position"][key] = float(
                float(pose["position"][key]) + distance * float(component)
            )
        if retreat_axis is not None:
            # Lift while backing off along the fixture axis so the open fingers
            # clear the released object instead of dragging it along the fixture.
            rise = float(profile.get("vertical_retreat_fraction", _VERTICAL_RETREAT_FRACTION))
            pose["position"]["z"] = float(pose["position"]["z"]) + rise * distance
        return pose, distance

    retreat: dict[str, Any] = start
    distance = 0.0
    planned_mode: str | None = None
    if retract_planner:
        planned_mode = "cartesian"
        rungs = _RETRACT_LADDER_M if retract_ladder_m is None else retract_ladder_m
        tried: list[float] = []
        for rung in rungs:
            pose, rung_distance = retreat_by(float(rung))
            if rung_distance in tried:
                continue  # the extent floor collapsed two rungs onto one line
            tried.append(rung_distance)
            try:
                plan = ctx.tool(
                    "motion.plan_linear", end=pose, orientation="lock", allow_start_contact=True,
                    contact_margin=float(retract_contact_margin_m), **on_arm,
                )
            except Exception as error:  # noqa: BLE001 -- a raise is a refusal; the servo remains
                print(f"[release_and_retract] planned retract of {rung_distance:.3f} m failed: {error}", flush=True)
                continue
            trajectory = (plan or {}).get("trajectory") if (plan or {}).get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                ctx.tool("robot.execute_trajectory", trajectory=trajectory, **on_arm)
                retreat, distance, planned_mode = pose, rung_distance, "planned_linear"
                break
    if planned_mode != "planned_linear":
        retreat, distance = retreat_by(float(profile.get("retract_m", retract_m)))
        ctx.tool("robot.go_to_pose_cartesian", pose=retreat, **on_arm)
    # Let the released object come to rest before the graph reports success:
    # a freshly released object can still be swinging on its fixture.
    wait_steps = int(profile.get("post_release_wait_steps", settle_steps))
    ctx.tool("robot.wait_steps", steps=wait_steps)
    report: dict[str, Any] = {
        "relation": relation,
        "arm_id": int(arm) if arm is not None else 0,
        "release_settle_steps": release_steps,
        "retreat_distance_m": distance,
        "post_release_wait_steps": wait_steps,
    }
    if planned_mode is not None:
        report["retreat_mode"] = planned_mode
    return {"released": True, "retreat_pose": retreat, "release_report": report}
