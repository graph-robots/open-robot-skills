"""Plan lift, minimum reorientation, and orientation-locked fixture transit.

Four strategies, the first two unchanged from before the profile existed:

- ``clearance_first`` (default): escape along the support normal, rotate in
  place to the least wrist turn that a planner check accepts, then transit to
  the staging pose in front of the fixture.
- ``direct``: escape, then one planned joint leg to a caller-supplied
  ``approach_pose``.
- ``direct_cartesian``: escape, then a Cartesian transition to
  ``approach_pose`` with the collision world and attachment left out -- for a
  pickup near a reach boundary where a planned leg keeps refusing a start
  state the executor can simply servo out of.
- ``carry_then_turn``: escape, carry to the staging position still in the
  pick orientation (a planned joint leg), then turn in place there (a planned
  joint leg flagged ``hold_position``). The turn waypoint carries a
  ``turn_search`` block, which ``executing-held-object-motion`` uses to choose
  the carry and the wrist yaw together by planning them, insertion strokes
  and retract included. Needs a connector whose ``motion.plan_to_pose``
  accepts ``hold_position`` and ``seed_joints``.

Why ``carry_then_turn`` exists (the ``sharps_disposal`` benchmark,
``gap_perception_v2``). The reference scripted policy for that suite carries
the syringe to the box in its pick orientation and only turns it upright once
parked next to the box; rotating upright over the tray and then carrying it
vertical was the opposite order. ``hold_position`` keeps the solver from
re-resolving the whole arm into an unrelated elbow configuration for what is a
pure in-place turn: without it one joint landed 35 deg from the equivalent
solve and 4/4 fresh runs lost the insertion. The smallest-turn yaw this
planner picks is only checked with the lenient ``motion.plan_joint``; on
``tray_clutter_t03`` it left the insertion strokes 2-6 mrad from a joint limit
and cuRobo refused them, while yaws +110..+160 deg had 0.4-0.7 rad of margin.
So the yaw is re-chosen live by the executor, and this pose is its fallback
geometry. The transit leg is a free joint goal rather than an axis-locked
line: five ``motion.plan_linear`` transit runs all executed a 390-570 mm climb.

A graph that carries one profile per object kind passes ``motion_profile``
(or a ``motion_profiles`` table keyed by ``target_kind``). The profile's
``strategy`` picks one of the three (``staged_alignment`` is accepted as the
older name for ``clearance_first``); its other keys are OPTIONAL and, when
absent, leave the plan exactly as the strategy alone would emit it -- in
particular ``speed_scale``, ``use_world``, ``max_attempts`` and
``cartesian_fallback`` appear on a waypoint only when the profile names them,
because the executor only forwards what a waypoint carries, and a graph that
never asked for a speed must not start passing one. ``arm_id`` is likewise
threaded only when set.

``accept_unchecked_symmetry`` (default off) keeps a plan when every symmetry
candidate fails the ``motion.plan_joint`` check, taking the smallest turn
unchecked instead of raising. It is meant for ``carry_then_turn``, whose turn
the executor re-plans anyway: sweep s5's ``t02`` aborted here when the lenient
IK member rejected all eight headings. Left off, no feasible candidate still
raises and routes to ``blocked``.
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


_STRATEGY_ALIASES = {"staged_alignment": "clearance_first"}
_STRATEGIES = {"clearance_first", "direct", "direct_cartesian", "carry_then_turn"}

#: The insertion ``plan_linear_engagement`` flies with ``stroke_depths_m``:
#: 20 mm before the fixture plane, then 10 mm phases to it -- signed positions
#: of the held feature along the fixture axis from its centre. The executor's
#: turn search probes these strokes from each candidate's end joints.
_TURN_SEARCH_DEPTHS_M = (-0.020, -0.010, 0.0)


def _vec(value: dict[str, Any]) -> np.ndarray:
    return np.array([float(value[key]) for key in ("x", "y", "z")], dtype=float)


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])])


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {
        "position": dict(zip(("x", "y", "z"), map(float, position), strict=True)),
        "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
    }


def _resolve_profile(
    motion_profile: dict[str, Any] | None,
    target_kind: str,
    motion_profiles: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if motion_profile is not None:
        return dict(motion_profile)
    if not target_kind and not motion_profiles:
        return {}
    # Compatibility adapter for already-materialised workflows; new graphs
    # pass the selected ``motion_profile`` directly.
    profiles = {str(item["source_kind"]): item for item in (motion_profiles or [])}
    if target_kind not in profiles:
        raise ValueError(f"no held-object motion profile declared for profile key {target_kind!r}")
    return dict(profiles[target_kind])


def _leg(pose: dict[str, Any], mode: str, profile: dict[str, Any], *, speed_key: str,
         **fixed: Any) -> dict[str, Any]:
    """One waypoint: the fixed keys always, the profile-driven keys only when named."""
    leg: dict[str, Any] = {"pose": pose, "mode": mode, "cartesian": mode == "contact_transition"}
    leg.update(fixed)
    if speed_key in profile:
        leg["speed_scale"] = float(profile[speed_key])
    return leg


def run(
    ctx: NodeContext,
    held_feature_in_tcp: dict[str, Any],
    fixture_feature: dict[str, Any],
    relation: str,
    world_config: dict[str, Any],
    attached_object: dict[str, Any],
    approach_pose: dict[str, Any] | None = None,
    strategy: str = "clearance_first",
    support_normal: dict[str, float] | None = None,
    approach_clearance_m: float = 0.08,
    escape_m: float = 0.06,
    stage_outward_m: float = 0.0,
    arm_id: int | None = None,
    target_kind: str = "",
    motion_profile: dict[str, Any] | None = None,
    motion_profiles: list[dict[str, Any]] | None = None,
    accept_unchecked_symmetry: bool = False,
    turn_search: bool = True,
    turn_search_depths_m: list[float] | None = None,
) -> Output:
    profile = _resolve_profile(motion_profile, target_kind, motion_profiles)
    strategy = str(profile.get("strategy", strategy))
    strategy = _STRATEGY_ALIASES.get(strategy, strategy)
    if strategy not in _STRATEGIES:
        raise ValueError(f"unsupported carry strategy {strategy!r}")
    if arm_id is None:
        carried = attached_object.get("arm_id") if isinstance(attached_object, dict) else None
        arm_id = None if carried is None else int(carried)
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}
    plan_head: dict[str, Any] = dict(on_arm)  # the plan names its arm only when one was named
    normal = _vec(support_normal or {"x": 0.0, "y": 0.0, "z": 1.0})
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)

    def optional(keys: dict[str, str]) -> dict[str, Any]:
        """``{waypoint key: profile key}`` -> the ones the profile names, typed."""
        out: dict[str, Any] = {}
        for waypoint_key, profile_key in keys.items():
            if profile_key in profile:
                value = profile[profile_key]
                out[waypoint_key] = int(value) if waypoint_key == "max_attempts" else bool(value)
        return out

    if strategy in {"direct", "direct_cartesian"}:
        if approach_pose is None:
            raise ValueError("direct carry requires an approach pose")
        ee = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
        current_position = _vec(ee["position"])
        current_rotation = _rotation(ee)
        # Move the newly attached object out of residual support/target depth
        # before the planner validates its start state. The initial move
        # preserves orientation and is deliberately a contact transition; its
        # clearance is governed by the held body's thickness at the hand, not
        # by spheres at the far ends of a long axis. The executor plans the
        # single joint leg to the approach pose from that lifted state.
        escape = float(profile.get("escape_distance_m", escape_m))
        escape_pose = _pose(current_position + escape * normal, current_rotation)
        if strategy == "direct":
            transit = {
                "pose": approach_pose, "mode": "planned_joint", "cartesian": False,
                "allow_start_contact": True, "max_attempts": 3,
            }
            time_scale = float(profile.get("time_scale", 2.0))
        else:
            # Do not demand a large wrist rotation at the pickup position: near
            # a workspace boundary it can be outside the arm's orientation
            # range even though the destination is reachable. Move up and in
            # and rotate in one smooth Cartesian transition to the already
            # validated approach pose; residual RGB-D fragments and the
            # conservative attachment fit are omitted for this monotone
            # free-space segment.
            transit = _leg(
                approach_pose, "contact_transition", profile, speed_key="transit_speed_scale",
                use_attachment=bool(profile.get("use_attachment", False)),
                use_world=bool(profile.get("use_world", False)),
                max_attempts=int(profile.get("max_attempts", 3)),
                cartesian_fallback=bool(profile.get("cartesian_fallback", True)),
            )
            time_scale = float(profile.get("time_scale", 1.25))
        return {
            "reorientation_plan": {
                **plan_head,
                "time_scale": time_scale,
                "waypoints": [
                    _leg(escape_pose, "contact_transition", profile, speed_key="lift_speed_scale"),
                    transit,
                ],
                "world_config": world_config,
                "attached_object": attached_object,
            }
        }

    if relation not in {
        "shaft_into_aperture",
        "tip_through_aperture",
        "insert_through",
        "loop_over_shaft",
        "feature_to_fixture",
    }:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis_value = fixture_feature.get("axis")
    if not axis_value:
        raise ValueError("fixture feature has no directed axis")
    fixture_pose = fixture_feature["pose"]
    axis = _vec(axis_value)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)

    ee = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    current_rotation = _rotation(ee)
    current_position = _vec(ee["position"])
    feature_rotation = _rotation(held_feature_in_tcp)
    feature_offset = _vec(held_feature_in_tcp["position"])
    current_axis = current_rotation.apply(feature_rotation.apply([0.0, 0.0, 1.0]))
    current_axis /= max(float(np.linalg.norm(current_axis)), 1.0e-12)
    correction, _ = Rotation.align_vectors([axis], [current_axis])
    aligned = correction * current_rotation

    fixture_center = _vec(fixture_pose["position"])
    clearance = max(0.02, float(approach_clearance_m))
    # A shaft axis points from its base toward its free end, so a loop stages
    # farther along that axis. An aperture axis points into the opening, so an
    # inserting tip stages on the opposite side.
    direction = 1.0 if relation == "loop_over_shaft" else -1.0
    feature_target = fixture_center + direction * clearance * axis
    # ``horizontal_staging_offset_m`` is the profile's name for the same
    # thing; the offset is taken in the support plane, which for a level bench
    # is the horizontal plane.
    outward_m = float(profile.get("horizontal_staging_offset_m", stage_outward_m))
    if outward_m > 0.0:
        # A recessed opening is often observed most strongly at its far rim.
        # Stage over the interior instead of directly above that depth return:
        # shift the staging target within the support plane, away from the
        # fixture toward the current hand.
        outward = current_position - fixture_center
        outward -= normal * float(outward @ normal)
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > 1.0e-6:
            feature_target = feature_target + outward_m * outward / outward_norm
    validate = bool(profile.get("validate_symmetry_with_planner", True))
    candidates = []
    unchecked = []
    for angle in np.deg2rad([0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0]):
        rotation = Rotation.from_rotvec(axis * float(angle)) * aligned
        hand_position = feature_target - rotation.apply(feature_offset)
        candidate = _pose(hand_position, rotation)
        # Geodesic TCP rotation is the stable, robot-independent score. It
        # prevents symmetry from producing an arbitrary large wrist turn.
        turn = float((rotation * current_rotation.inv()).magnitude())
        unchecked.append((turn, abs(float(angle)), candidate, rotation))
        if not validate:
            # A Cartesian-only profile deliberately never invokes the planner,
            # not even as a candidate filter; the Cartesian executor reports a
            # genuinely unreachable pose.
            candidates.append((turn, abs(float(angle)), candidate, rotation))
            continue
        try:
            check = ctx.tool("motion.plan_joint", pose=candidate, orientation="lock", **on_arm)
        except Exception:
            continue
        if not check.get("planned"):
            continue
        if float(check.get("position_error_m", 0.0)) > 0.006:
            continue
        if float(check.get("rotation_error_rad", 0.0)) > np.deg2rad(4.0):
            continue
        candidates.append((turn, abs(float(angle)), candidate, rotation))
    if not candidates:
        if not bool(profile.get("accept_unchecked_symmetry", accept_unchecked_symmetry)):
            raise RuntimeError("no feasible orientation aligns the held feature with the fixture")
        candidates = unchecked
    _, _, transit_pose, goal_rotation = min(candidates, key=lambda item: item[:2])

    def sphere_extent(sphere: dict[str, Any]) -> float:
        center = sphere.get("center") or {}
        offset = (
            np.asarray(center, dtype=float).reshape(3)
            if isinstance(center, (list, tuple))
            else np.array([float(center.get(k, 0.0)) for k in ("x", "y", "z")])
        )
        return float(np.linalg.norm(offset)) + float(sphere.get("radius", 0.0))

    extents = [sphere_extent(s) for s in attached_object.get("spheres", [])]
    escape_distance = max(0.04, max(extents, default=0.025) + 0.015)
    escape_position = current_position + escape_distance * normal
    rotate_pose = _pose(escape_position, goal_rotation)
    escape_leg = _leg(_pose(escape_position, current_rotation), "contact_transition", profile,
                      speed_key="lift_speed_scale")
    # A profile may ask for the in-place rotation, or the transit, to be a
    # Cartesian transition instead of a planned leg when the local volume is
    # known to be clear; unasked, both are planned as they always were.
    reorient_mode = "contact_transition" if bool(profile.get("cartesian_reorient", False)) else "planned_joint"
    transit_mode = "contact_transition" if bool(profile.get("cartesian_transit", False)) else "planned_joint"
    clearance_legs = [
        _leg(rotate_pose, reorient_mode, profile, speed_key="reorient_speed_scale",
             use_attachment=bool(profile.get("reorient_use_attachment", False)),
             **optional({"use_world": "reorient_use_world", "max_attempts": "max_attempts",
                         "cartesian_fallback": "cartesian_fallback"})),
        _leg(transit_pose, transit_mode, profile, speed_key="transit_speed_scale",
             **optional({"use_attachment": "transit_use_attachment", "use_world": "transit_use_world",
                         "max_attempts": "max_attempts", "cartesian_fallback": "transit_cartesian_fallback"})),
    ]
    if strategy == "carry_then_turn":
        # Carry first, at the pick orientation, to the same hand position the
        # turn ends at: the turn is then a same-place rotation, not a second
        # lateral move disguised as one.
        carry_pose = _pose(_vec(transit_pose["position"]), current_rotation)
        turn_leg = _leg(transit_pose, "planned_joint", profile, speed_key="reorient_speed_scale",
                        hold_position=True,
                        **optional({"use_attachment": "reorient_use_attachment", "use_world": "reorient_use_world",
                                    "max_attempts": "max_attempts", "cartesian_fallback": "cartesian_fallback"}))
        if bool(profile.get("turn_search", turn_search)):
            depths = profile.get("turn_search_depths_m", turn_search_depths_m)
            turn_leg["turn_search"] = {
                "fixture_center": [float(v) for v in fixture_center],
                "axis": [float(v) for v in axis],
                "feature_offset": [float(v) for v in feature_offset],
                "feature_rotation": dict(held_feature_in_tcp["rotation"]),
                "transit_clearance_m": clearance,
                # Where the feature sits at the turn, with this planner's own
                # staging sign and outward shift; the search stages there.
                "transit_target": [float(v) for v in feature_target],
                "insert_depths_m": [float(d) for d in (_TURN_SEARCH_DEPTHS_M if depths is None else depths)],
            }
        # A connector without ``hold_position`` / ``seed_joints`` refuses the turn
        # at the call, before the arm has left the escape point; the executor then
        # flies these instead -- exactly the legs ``clearance_first`` emits here.
        turn_leg["clearance_first_fallback"] = clearance_legs
        return {
            "reorientation_plan": {
                **plan_head,
                "time_scale": float(profile.get("time_scale", 2.0)),
                "waypoints": [
                    escape_leg,
                    _leg(carry_pose, "planned_joint", profile, speed_key="transit_speed_scale",
                         **optional({"use_attachment": "transit_use_attachment", "use_world": "transit_use_world",
                                     "max_attempts": "max_attempts",
                                     "cartesian_fallback": "transit_cartesian_fallback"})),
                    turn_leg,
                ],
                "world_config": world_config,
                "attached_object": attached_object,
            }
        }
    return {
        "reorientation_plan": {
            **plan_head,
            "time_scale": float(profile.get("time_scale", 2.0)),
            "waypoints": [escape_leg, *clearance_legs],
            "world_config": world_config,
            "attached_object": attached_object,
        }
    }
