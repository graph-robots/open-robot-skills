"""Canonical computing-feature-mating-poses implementation from Open Robot Skills."""

from typing import Any, TypedDict

from gap import NodeContext


class Output(TypedDict):
    mate_pose: dict[str, Any]
    approach_pose: dict[str, Any]
    engaged_pose: dict[str, Any]
    approach_axis: dict[str, float]
    seating_distance: float
    minimum_clearance: float


def _move_loop_poses_to_shaft_interior(
    result: dict[str, Any],
    held_feature_in_tcp: dict[str, Any],
    fixture_feature: dict[str, Any],
    axis: dict[str, float],
    held_radius: float,
    fixture_radius: float,
) -> None:
    """Ensure loop poses reach, but do not overshoot, shaft-interior depths."""
    usable_length = max(0.0, float(fixture_feature.get("usable_length", 0.0)))
    if usable_length <= 0.0:
        return

    # Cross far enough that the loop surrounds the shaft, then place the final
    # mate well inside the observed segment while retaining clearance from its
    # mounting surface. All distances derive from functional geometry.
    crossing_depth = max(0.006, 2.0 * fixture_radius, 0.20 * held_radius)
    base_margin = max(0.005, 2.0 * fixture_radius)
    maximum_depth = max(0.0, usable_length - base_margin)
    # Place the loop well inside the observed usable segment while retaining
    # clearance from the mounting surface. The requested depth is enforced
    # exactly below, so this displacement is not applied twice.
    mate_depth = min(maximum_depth, max(crossing_depth, 0.40 * usable_length))
    engaged_depth = mate_depth

    held_position = held_feature_in_tcp.get("position", {})
    local_point = [float(held_position.get(key, 0.0)) for key in ("x", "y", "z")]
    fixture_position = fixture_feature["pose"]["position"]
    tip = [float(fixture_position[key]) for key in ("x", "y", "z")]
    outward_axis = [float(axis[key]) for key in ("x", "y", "z")]
    profile = dict(fixture_feature.get("mating_profile") or {})
    settling_axis = [float(v) for v in profile.get("settling_axis", [0.0, 0.0, 1.0])]

    def feature_world_and_position(pose_name: str):
        pose = result[pose_name]
        rotation = pose["rotation"]
        w, x, y, z = (float(rotation[key]) for key in ("w", "x", "y", "z"))
        qx = 2.0 * (y * local_point[2] - z * local_point[1])
        qy = 2.0 * (z * local_point[0] - x * local_point[2])
        qz = 2.0 * (x * local_point[1] - y * local_point[0])
        rotated = [
            local_point[0] + w * qx + (y * qz - z * qy),
            local_point[1] + w * qy + (z * qx - x * qz),
            local_point[2] + w * qz + (x * qy - y * qx),
        ]
        position = pose["position"]
        feature_world = [
            float(position[key]) + rotated[index]
            for index, key in enumerate(("x", "y", "z"))
        ]
        return feature_world, position

    # The loop must cross the distal tip with the shaft centered in its
    # opening. Crossing at the inner rim gives effectively zero clearance and
    # lets contact push the held object away before engagement completes.
    for pose_name in ("approach_pose", "engaged_pose"):
        feature_world, position = feature_world_and_position(pose_name)
        relative = [feature_world[index] - tip[index] for index in range(3)]
        axial = sum(relative[index] * outward_axis[index] for index in range(3))
        transverse = [
            relative[index] - axial * outward_axis[index] for index in range(3)
        ]
        for index, key in enumerate(("x", "y", "z")):
            position[key] = float(position[key]) - transverse[index]
        crossing_offsets = profile.get("crossing_offsets_m", {})
        compensation = float(crossing_offsets.get(pose_name, 0.0))
        for index, key in enumerate(("x", "y", "z")):
            position[key] = float(position[key]) + compensation * settling_axis[index]

    # Once inside, lower the loop until the shaft approaches its inner rim,
    # retaining a small geometric clearance for the local contact motion.
    feature_world, position = feature_world_and_position("mate_pose")
    relative = [feature_world[index] - tip[index] for index in range(3)]
    axial = sum(relative[index] * outward_axis[index] for index in range(3))
    transverse = [relative[index] - axial * outward_axis[index] for index in range(3)]
    transverse_norm = sum(value * value for value in transverse) ** 0.5
    default_seated_offset = max(0.0, held_radius - fixture_radius - 0.001)
    seated_offset = float(profile.get("seated_radial_offset_m", default_seated_offset))
    if transverse_norm > 1e-9:
        desired = [value * seated_offset / transverse_norm for value in transverse]
        for index, key in enumerate(("x", "y", "z")):
            position[key] = float(position[key]) + desired[index] - transverse[index]
    mate_offset = float(profile.get("mate_settling_offset_m", 0.0))
    for index, key in enumerate(("x", "y", "z")):
        position[key] = float(position[key]) + mate_offset * settling_axis[index]

    for pose_name, requested_depth in (
        ("engaged_pose", engaged_depth),
        ("mate_pose", mate_depth),
    ):
        feature_world, position = feature_world_and_position(pose_name)
        current_depth = -sum(
            (feature_world[index] - tip[index]) * outward_axis[index]
            for index in range(3)
        )
        depth_correction = requested_depth - current_depth
        for key in ("x", "y", "z"):
            position[key] = float(position[key]) - depth_correction * float(axis[key])


def run(
    ctx: NodeContext,
    held_feature_in_tcp: dict[str, Any],
    fixture_feature: dict[str, Any],
    relation: str,
    attached_object: dict[str, Any],
    arm_id: int | None = None,
) -> Output:
    if arm_id is None:
        arm_id = int(attached_object.get("arm_id", 0))
    axis = fixture_feature.get("axis")
    if not axis:
        raise ValueError("fixture feature has no directed axis")
    held_radius = float(held_feature_in_tcp.get("radius_inner", 0.0))
    fixture_radius = float(fixture_feature.get("radius_outer", 0.0))
    dimensions = {}
    if relation == "loop_over_shaft" and max(held_radius, fixture_radius) > 0.0:
        dimensions = {
            "approach_margin": 3.0 * max(held_radius, fixture_radius),
            "seating_margin": float(fixture_feature.get("seating_margin", 0.0)),
        }
    elif relation == "loop_over_shaft":
        dimensions = {
            "seating_margin": float(fixture_feature.get("seating_margin", 0.0)),
        }
    reference_pose = ctx.tool("robot.get_ee_pose", arm_id=int(arm_id))["pose"]
    result = ctx.tool(
        "geometry.compute_feature_mate",
        held_feature_in_tcp=held_feature_in_tcp,
        fixture_pose=fixture_feature["pose"],
        fixture_axis=axis,
        relation=relation,
        held_radius=held_radius,
        fixture_radius=fixture_radius,
        reference_tcp_pose=reference_pose,
        attached_object=attached_object,
        **dimensions,
    )
    if relation == "loop_over_shaft":
        _move_loop_poses_to_shaft_interior(
            result, held_feature_in_tcp, fixture_feature, axis,
            held_radius, fixture_radius
        )
    if relation in {"shaft_into_aperture", "tip_through_aperture", "insert_through"}:
        # The fixture axis points into an aperture. Keep the approach on the
        # free side of the mate even when a backend reports the axial offset
        # using the protruding-shaft convention.
        mate = result["mate_pose"]["position"]
        approach = result["approach_pose"]["position"]
        for key in ("x", "y", "z"):
            approach[key] = float(mate[key]) + (float(mate[key]) - float(approach[key]))
        depth = float(fixture_feature.get("insertion_depth", 0.0))
        for pose_name in ("mate_pose", "engaged_pose"):
            position = result[pose_name]["position"]
            for key in ("x", "y", "z"):
                position[key] = float(position[key]) + depth * float(axis[key])
    return result
