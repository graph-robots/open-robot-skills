"""Canonical clearance-first carry planner from planning-held-object-motion."""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation


class Output(TypedDict):
    reorientation_plan: dict[str, Any]


def _rotation(pose: dict[str, Any]) -> Rotation:
    q = pose["rotation"]
    return Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]])


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {"position": dict(zip(("x", "y", "z"), map(float, position))),
            "rotation": {"w": float(q[3]), "x": float(q[0]),
                         "y": float(q[1]), "z": float(q[2])}}


def _vec(value: dict[str, float]) -> np.ndarray:
    return np.array([value[key] for key in ("x", "y", "z")], dtype=np.float64)


def run(ctx: NodeContext, held_feature_in_tcp: dict[str, Any],
        fixture_feature: dict[str, Any], relation: str,
        world_config: dict[str, Any], attached_object: dict[str, Any],
        approach_pose: dict[str, Any] | None = None,
        target_kind: str = "",
        support_normal: dict[str, float] | None = None,
        approach_clearance_m: float = 0.08,
        arm_id: int | None = None,
        motion_profile: dict[str, Any] | None = None,
        motion_profiles: list[dict[str, Any]] | None = None) -> Output:
    if arm_id is None:
        arm_id = int(attached_object.get("arm_id", 0))
    if motion_profile is not None:
        profile = dict(motion_profile)
    elif not target_kind and not motion_profiles:
        profile = {"strategy": "staged_alignment"}
    else:
        # Compatibility adapter for already-materialized workflows. New graphs
        # should pass the selected ``motion_profile`` directly.
        profiles = {str(item["source_kind"]): item for item in (motion_profiles or [])}
        if target_kind not in profiles:
            raise ValueError(
                f"no held-object motion profile declared for profile key {target_kind!r}"
            )
        profile = dict(profiles[target_kind])
    strategy = str(profile["strategy"])
    if strategy == "direct_cartesian":
        if approach_pose is None:
            raise ValueError("direct carry requires an approach pose")
        ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
        current_position = _vec(ee["position"])
        current_rotation = _rotation(ee)
        normal = _vec(support_normal or {"x": 0.0, "y": 0.0, "z": 1.0})
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        # Move the newly attached object out of residual support depth
        # before asking CuRobo to validate its start state.  The initial move
        # preserves orientation and is deliberately a contact transition.
        # Keep the profile-tested 60 mm default Cartesian lift: a larger
        # straight lift can cross an arm's local reach boundary. Robustness comes from
        # splitting wrist rotation and fixture transit into separate plans,
        # rather than increasing the lift height.
        escape_distance = float(profile.get("escape_distance_m", 0.06))
        escape_pose = _pose(current_position + escape_distance * normal, current_rotation)
        return {"reorientation_plan": {
            "arm_id": int(arm_id),
            "time_scale": float(profile.get("time_scale", 1.25)),
            "waypoints": [
                {"pose": escape_pose, "mode": "contact_transition", "cartesian": True,
                 "speed_scale": float(profile.get("lift_speed_scale", 1.0))},
                # Do not demand a large wrist rotation at this exact pickup
                # position: near workspace boundaries it can be outside the
                # arm's orientation range even though the destination is
                # reachable. Move upward/inward and rotate in one smooth joint
                # trajectory to the already validated approach pose. Residual
                # RGB-D robot/support fragments and a conservative attachment
                # sphere fit are omitted for this monotone free-space segment;
                # insertion still uses the scene-aware local alignment below.
                {"pose": approach_pose,
                 "mode": "contact_transition",
                 "cartesian": True,
                 "speed_scale": float(profile.get("transit_speed_scale", 1.0)),
                 "use_attachment": bool(profile.get("use_attachment", False)),
                 "use_world": bool(profile.get("use_world", False)),
                 "max_attempts": int(profile.get("max_attempts", 3)),
                 "cartesian_fallback": bool(profile.get("cartesian_fallback", True))},
            ],
            "world_config": world_config,
            "attached_object": attached_object,
        }}
    if strategy != "staged_alignment":
        raise ValueError(f"unsupported held-object motion strategy {strategy!r}")
    if relation not in {"shaft_into_aperture", "tip_through_aperture", "insert_through",
                        "loop_over_shaft", "feature_to_fixture"}:
        raise ValueError(f"unsupported feature relation {relation!r}")
    axis_value = fixture_feature.get("axis")
    if not axis_value:
        raise ValueError("fixture feature has no directed axis")
    axis = _vec(axis_value); axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    normal = _vec(support_normal or {"x": 0.0, "y": 0.0, "z": 1.0})
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)

    ee = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
    current_rotation = _rotation(ee)
    feature_rotation = _rotation(held_feature_in_tcp)
    feature_offset = _vec(held_feature_in_tcp["position"])
    current_axis = current_rotation.apply(feature_rotation.apply([0.0, 0.0, 1.0]))
    current_axis /= max(float(np.linalg.norm(current_axis)), 1.0e-12)
    correction, _ = Rotation.align_vectors([axis], [current_axis])
    aligned = correction * current_rotation

    fixture_center = _vec(fixture_feature["pose"]["position"])
    current_position = _vec(ee["position"])
    clearance = max(0.02, float(approach_clearance_m))
    # A shaft axis points from its base toward its free end, so a loop stages
    # farther along that axis.  An aperture axis points into the opening, so an
    # inserting tip stages on the opposite side.
    direction = 1.0 if relation == "loop_over_shaft" else -1.0
    feature_target = fixture_center + direction * clearance * axis
    staging_offset = float(profile.get("horizontal_staging_offset_m", 0.0))
    if staging_offset > 0.0:
        # A top-opening box mounted on a wall is often observed most strongly
        # at its rear rim. Stage over the interior instead of directly above
        # that wall-adjacent depth return. The outward direction is inferred
        # from the aperture toward the current hand in the horizontal plane.
        outward = current_position - fixture_center
        outward[2] = 0.0
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > 1.0e-6:
            feature_target += staging_offset * outward / outward_norm
    candidates = []
    for angle in np.deg2rad([0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0]):
        rotation = Rotation.from_rotvec(axis * float(angle)) * aligned
        candidate = _pose(feature_target - rotation.apply(feature_offset), rotation)
        turn = float((rotation * current_rotation.inv()).magnitude())
        if not bool(profile.get("validate_symmetry_with_planner", True)):
            # Cartesian-only experiments deliberately avoid invoking CuRobo,
            # including as a candidate-pose filter. The Cartesian executor
            # remains responsible for reporting a genuinely unreachable pose.
            candidates.append((turn, abs(float(angle)), candidate, rotation))
            continue
        try:
            check = ctx.tool("motion.plan_joint", pose=candidate, orientation="lock", arm_id=int(arm_id))
        except Exception:
            continue
        if (not check.get("planned") or float(check.get("position_error_m", 0.0)) > 0.006
                or float(check.get("rotation_error_rad", 0.0)) > np.deg2rad(4.0)):
            continue
        candidates.append((turn, abs(float(angle)), candidate, rotation))
    if not candidates:
        raise RuntimeError("no feasible orientation aligns the held feature with the fixture")
    _, _, transit_pose, goal_rotation = min(candidates, key=lambda item: item[:2])

    def sphere_extent(sphere: dict[str, Any]) -> float:
        center = sphere.get("center") or {}
        offset = (np.asarray(center, dtype=float).reshape(3) if isinstance(center, (list, tuple))
                  else np.array([float(center.get(k, 0.0)) for k in ("x", "y", "z")]))
        return float(np.linalg.norm(offset)) + float(sphere.get("radius", 0.0))

    escape_distance = max(0.04, max((sphere_extent(s) for s in attached_object.get("spheres", [])),
                                    default=0.025) + 0.015)
    escape_position = current_position + escape_distance * normal
    return {"reorientation_plan": {
        "arm_id": int(arm_id),
        # The selected profile controls free-space timing independently from
        # insertion/contact timing and is reusable across object categories.
        "time_scale": float(profile.get("time_scale", 2.0)),
        "waypoints": [
            {"pose": _pose(escape_position, current_rotation),
             "mode": "contact_transition", "cartesian": True,
             "speed_scale": float(profile.get("lift_speed_scale", 1.0))},
            {"pose": _pose(escape_position, goal_rotation),
             # A profile may request Cartesian-first reorientation after the
             # vertical escape when the local volume is known to be clear.
             "mode": "contact_transition" if bool(profile.get("cartesian_reorient", False)) else "planned_joint",
             "cartesian": bool(profile.get("cartesian_reorient", False)),
             "speed_scale": float(profile.get("reorient_speed_scale", 1.0)),
             # At this lifted pose the tool has already cleared its support.
             # Sparse RGB-D arm/table fragments can make an otherwise valid
             # in-place wrist rotation appear colliding. Match the proven
             # declared carry behavior for this free-space orientation change;
             # the fixture transit below restores the full collision world.
             "use_attachment": bool(profile.get("reorient_use_attachment", False)),
             "use_world": bool(profile.get("reorient_use_world", False)),
             "max_attempts": int(profile.get("max_attempts", 3)),
             "cartesian_fallback": bool(profile.get("cartesian_fallback", True))},
            {"pose": transit_pose,
             "mode": "contact_transition" if bool(profile.get("cartesian_transit", False)) else "planned_joint",
             "cartesian": bool(profile.get("cartesian_transit", False)),
             "speed_scale": float(profile.get("transit_speed_scale", 1.0)),
             # At the free-space boundary, sparse RGB-D fixture fragments and
             # conservative attached spheres repeatedly report a false goal
             # collision. Use the profile-declared carry strategy here; the
             # subsequent pre-approach/insertion stages restore the complete
             # scene and attachment checks can resume before fixture clearance.
             "use_attachment": bool(profile.get("transit_use_attachment", True)),
             "use_world": bool(profile.get("transit_use_world", True)),
             "max_attempts": int(profile.get("max_attempts", 3)),
             "cartesian_fallback": bool(profile.get("cartesian_fallback", False))},
        ],
        "world_config": world_config,
        "attached_object": attached_object,
    }}
