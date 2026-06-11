"""Compute a safe drop pose and approach pose above a container OBB.

Given a container's oriented bounding box, produce:
- ``drop_position`` — Vec3 at the container's XY center, slightly above its top
- ``drop_pose`` — Se3Pose at the drop position with a top-down gripper orientation
- ``approach_pose`` — Se3Pose well above the container for a safe lateral approach

All rotations use the canonical top-down quaternion (w=0, x=1, y=0, z=0), which
points the gripper straight down regardless of the container's orientation —
except when ``ee_pose_at_grasp`` is supplied, in which case the grasp-time
wrist YAW is preserved (see ``_yaw_only_topdown``).
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox, Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3
    drop_pose: Se3Pose
    approach_pose: Se3Pose


# Canonical top-down gripper orientation (z-axis pointing down in world).
_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def _yaw_only_topdown(ee_pose: Se3Pose) -> Quaternion:
    """Return a top-down quaternion that carries only the world-z yaw of
    ``ee_pose``'s rotation. Used to align the drop wrist with whatever
    yaw the grasp acquired (so the planner doesn't unspool that yaw
    mid-transport), while keeping pitch/roll strictly top-down so the
    held object always lands flat regardless of how the grasp was
    angled.

    Decomposition: yaw is extracted as the world-z rotation of the
    gripper's local x-axis after applying the grasp rotation. We then
    compose ``R_z(yaw) · R_x(pi)`` as the new orientation — equivalent
    to the canonical ``_DOWN`` rotated by ``yaw`` around world z.
    """
    import math

    from scipy.spatial.transform import Rotation as _R

    q = ee_pose["rotation"]
    # gap Quaternion is wxyz; scipy expects (x, y, z, w).
    R_grasp = _R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    # World-z component of the rotated x-axis encodes the yaw of the
    # gripper relative to its top-down reference frame.
    rx = R_grasp[:, 0]
    yaw = math.atan2(float(rx[1]), float(rx[0]))
    # R_x(pi) flips Z so the gripper points down; then R_z(yaw)
    # spins around world z. Order: world_yaw · top_down.
    R_yaw = _R.from_euler("z", yaw)
    R_topdown = _R.from_euler("x", math.pi)
    R_final = (R_yaw * R_topdown).as_quat()  # (x, y, z, w)
    return {
        "x": float(R_final[0]), "y": float(R_final[1]),
        "z": float(R_final[2]), "w": float(R_final[3]),
    }


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    container_interior_obb: OrientedBoundingBox | None = None,
    ee_pose_at_grasp: Se3Pose | None = None,
    drop_clearance: float = 0.05,
    approach_height: float = 0.20,
    held_obb: OrientedBoundingBox | None = None,
    panda_hand_to_tcp: float = 0.1029,
) -> Output:
    # Container-top reference (always defined). Used as the descent
    # target for the no-held-geometry fallback so the gripper releases
    # *just above* the rim — the legacy contract callers without
    # ``held_obb`` still rely on.
    container_top = container_obb["center"]["z"] + container_obb["extent"]["z"]

    # Target the interior placement zone when available; otherwise fall
    # back to the exterior top.
    if container_interior_obb is not None:
        zone_center = container_interior_obb["center"]
        zone_floor = zone_center["z"] - container_interior_obb["extent"]["z"]
        zone_ceiling = zone_center["z"] + container_interior_obb["extent"]["z"]
    else:
        zone_center = container_obb["center"]
        zone_floor = container_top  # exterior top
        zone_ceiling = zone_floor + 0.10  # arbitrary headroom for fallback path

    # Measure the at-grasp EE height LIVE. This node runs right after the
    # grasp subgraph closes the gripper and before any lift, so the
    # current EE pose IS the at-grasp pose the invariant below requires.
    # Generated graphs typically wire ``ee_pose_at_grasp`` from an
    # ``observe`` node that runs at the pre-grasp HOVER (between approach
    # and plan) — recorded traces show that observation ~0.12-0.20 m above
    # the true at-grasp height, which inflates ``ee_to_obj_z`` and made
    # the release happen ~30 cm above the rim (items bounced out of the
    # basket or rolled away on landing). The wired value is kept for yaw
    # preservation and as a fallback when the live read fails.
    ee_z_at_grasp: float | None = None
    try:
        live = ctx.tool("robot.get_ee_pose")
        ee_z_at_grasp = float(live["pose"]["position"]["z"])
    except Exception:
        if ee_pose_at_grasp is not None:
            ee_z_at_grasp = float(ee_pose_at_grasp["position"]["z"])

    if held_obb is not None and ee_z_at_grasp is not None:
        # LIBERO's ``In(obj, contain_region)`` predicate checks that the
        # object's CENTER is inside the contain_region 3D AABB. Target a
        # held-object Z that:
        #   (a) keeps the held object's BOTTOM clear of the basket walls
        #       (modelled as a 2 cm shallow lip at the basket floor) so
        #       cuRobo's collision check passes during the descent,
        #   (b) keeps the held object's CENTER strictly inside the zone
        #       (so In() fires after release), and
        #   (c) gives a TCP target that's well within the Franka's
        #       reachable volume — short objects need a higher held
        #       center than the wall-clearance floor would suggest,
        #       otherwise the end-leg IK has too few feasible joint
        #       configurations.
        margin = max(0.03, drop_clearance)  # 3 cm above wall top
        desired_obj_z = zone_floor + margin + held_obb["extent"]["z"]
        # Hard ceiling: never push the held object's center past the zone
        # top — the In() predicate would fail.
        if desired_obj_z > zone_ceiling - 0.001:
            desired_obj_z = zone_ceiling - 0.001
        # Soft floor: if the zone is so shallow that the bottom margin
        # forces the center past the ceiling, give up on the margin.
        if desired_obj_z < zone_center["z"]:
            desired_obj_z = max(
                desired_obj_z, zone_floor + 0.001 + held_obb["extent"]["z"]
            )
        # The held cuboid is rigidly attached to the EE link. With a
        # top-down grip, the world Z difference between EE and the
        # held-object center is preserved across the trajectory:
        #   ee_z_at_drop - held_z_at_drop == ee_z_at_grasp - obj_z_at_grasp
        # ``held_obb["center"]["z"]`` IS ``obj_z_at_grasp`` because the OBB
        # was captured before the gripper closed; ``ee_z_at_grasp`` is the
        # live measurement taken above.
        ee_to_obj_z = ee_z_at_grasp - held_obb["center"]["z"]
        ee_z_at_drop = desired_obj_z + ee_to_obj_z
        tcp_z = ee_z_at_drop - panda_hand_to_tcp
    elif held_obb is not None:
        # Fallback: assume cuRobo's primary grasp_z_offset (0.04). Less
        # accurate but doesn't require ee_pose_at_grasp.
        desired_obj_z = max(
            zone_center["z"],
            zone_floor + drop_clearance + held_obb["extent"]["z"],
        )
        tcp_z = desired_obj_z + held_obb["extent"]["z"] - 0.04
    else:
        # No held geometry: place TCP just above the container top by
        # ``drop_clearance``. This matches the pre-redesign contract so
        # legacy callers (static workflows that don't pass ``held_obb``)
        # keep their behavior — the ``descend_release`` script
        # go_to_pose's *to* this height before opening the gripper, so it
        # must be a real release height near the rim, not a
        # high-headroom waypoint.
        tcp_z = container_top + drop_clearance

    # Preserve only the grasp-time wrist YAW (z-axis rotation) — keep
    # pitch/roll top-down. Without this the drop uses a fixed yaw=0
    # top-down quat (``_DOWN``), forcing curobo to unspool the wrist
    # yaw the gripper acquired to grip a non-axis-aligned subpart (e.g.
    # a horizontal frypan handle); the planner swings the arm through
    # a redundant-joint reconfiguration to do it — visible as a
    # circular elbow motion between pick and place. Preserving the FULL
    # grasp rotation (yaw + pitch + roll) is too aggressive: it carries
    # any grasp-angle tilt into the drop, so a slanted grasp ends up
    # releasing the held object on edge. Yaw-only keeps pitch/roll at
    # top-down so the held object lands flat regardless of how the
    # grasp was angled.
    drop_rotation = (
        _yaw_only_topdown(ee_pose_at_grasp) if ee_pose_at_grasp is not None else _DOWN
    )
    drop_position: Vec3 = {
        "x": zone_center["x"], "y": zone_center["y"], "z": tcp_z,
    }
    drop_pose: Se3Pose = {"position": drop_position, "rotation": drop_rotation}
    approach_pose: Se3Pose = {
        "position": {
            "x": zone_center["x"],
            "y": zone_center["y"],
            "z": tcp_z + approach_height,
        },
        "rotation": drop_rotation,
    }
    return {
        "drop_position": drop_position,
        "drop_pose": drop_pose,
        "approach_pose": approach_pose,
    }
