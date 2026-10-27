"""Straight-Z placement INTO a walled container: lift (Z) → XY over the
container → LINEAR descend to inside the walls.

The generalized form of the reference packing recipe, for any container with
an interior (basket, bin, box, tote). Lift/XY are straight-line cartesian
(``robot.go_to_pose_cartesian``); the descend uses cuRobo's AXIS-CONSTRAINED
linear move (``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``,
``orientation_mode="LOCK"``) — a guaranteed straight vertical drop. The descend
lowers the TCP to ``container_top − place_offset`` (with ``place_offset`` small
or negative to sit just above the rim so the held object clears the walls on the
way down and drops in) so the object is placed IN the container, not from a
lateral swing. For placement ONTO a surface or into a described sub-region/zone,
use the ``compute_drop_pose → waypoint_move → descend_release_linear`` path
instead — this script is the walled-container fast path.

Heights are derived, not resting literals: with ``lift_z`` / ``transport_z``
left at their ``-1.0`` sentinels both come from the perceived container's rim
and the wrist's current height, so the same script places on a table at any
height. Every cartesian leg is checked against ``robot.get_ee_pose`` afterwards
(a servo that stops 3 cm short is a leg that did not happen), and the lateral
leg searches the container's reachable interior when its centre is out of
reach.
"""

import logging
import math
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Quaternion, Vec3

logger = logging.getLogger(__name__)


class Output(TypedDict):
    place_position: Vec3


def _aligned_rotation(rotation: Quaternion, extent: Vec3) -> Quaternion | None:
    """Turn the held object's long axis to lie along the container's long one.

    **This is not a wrist-yaw search.** A sweep that probes a ladder of yaws
    with a planner, looking for one it will accept, is a measured dead end: the
    planner refuses almost everything and is not a feasibility oracle, and such
    a sweep spent 24 probes per give-up and rescued nothing. This computes ONE
    angle from geometry, asks no planner, and is about whether the object FITS
    rather than whether the arm can get there.

    Both facts it needs are already to hand. A grasp frame built with local Y
    along the object's fitted linear feature means the rotation the hand is
    already carrying names the object's long axis -- column 1 of its matrix --
    so nothing extra has to be wired in. The container's own long axis is
    whichever of ``extent.x``/``extent.y`` is bigger.

    Without this the release yaw is inherited from wherever the object happened
    to be lying when it was picked up, which in a scattered bin is arbitrary. A
    compartment 180 x 230 mm inside will not take a 200 mm tool crosswise: at
    the wrong yaw it comes down across a divider rather than into a cell, and
    since a placement predicate scores the body's centre against the region,
    which way it then topples decides the outcome. Nothing raises when that
    happens, which is why it is worth spending a rotation to avoid.

    POST-multiplied, never pre-multiplied. Pre-multiplying by Rz spins a tilted
    approach around a cone and moves where the tool points; post-multiplying
    turns the tool about its own approach axis and leaves the approach exactly
    where it was.

    Returns ``None`` when no turn is wanted -- the long axis is already nearer
    the container's long axis than its short one, or points so close to
    vertical that its heading is not meaningful. ``None`` means "keep the grasp
    rotation", which is what the caller does unconditionally without this.
    """
    import numpy as np  # noqa: PLC0415 -- kept out of module scope, as this bundle requires
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    matrix = Rotation.from_quat(
        [float(rotation["x"]), float(rotation["y"]), float(rotation["z"]), float(rotation["w"])]
    ).as_matrix()
    axis = matrix[:, 1]
    flat = np.array([axis[0], axis[1]])
    if float(np.linalg.norm(flat)) < 0.15:
        return None
    have = math.atan2(float(flat[1]), float(flat[0]))
    want = math.pi / 2.0 if float(extent["y"]) >= float(extent["x"]) else 0.0
    # A long axis is a line, not an arrow: 180 degrees apart is the same lie.
    delta = (want - have + math.pi / 2.0) % math.pi - math.pi / 2.0
    if abs(delta) <= math.pi / 4.0:
        return None
    # Which way the post-multiplication turns the world heading depends on the
    # handedness of the grasp frame, and a downward approach makes that basis
    # left-handed about world +z. Rather than reason it out, build both and keep
    # whichever actually lands on the container's long axis.
    best, best_error = None, math.inf
    for signed in (delta, -delta):
        turned = matrix @ Rotation.from_euler("z", signed).as_matrix()
        heading = math.atan2(float(turned[1, 1]), float(turned[0, 1]))
        error = abs((want - heading + math.pi / 2.0) % math.pi - math.pi / 2.0)
        if error < best_error:
            best, best_error = turned, error
    quaternion = Rotation.from_matrix(best).as_quat()  # xyzw
    return {
        "w": float(quaternion[3]),
        "x": float(quaternion[0]),
        "y": float(quaternion[1]),
        "z": float(quaternion[2]),
    }


def _held_object(
    target_obb: OrientedBoundingBox | None, surface_inset_m: float
) -> tuple[float, float, float]:
    """``(half_length, half_width, hang)`` of the object in the jaws, in metres.

    The box carries HALF-extents in its own frame, so sorting the three gives
    half-length, half-width and half-thickness for anything resting flat.

    ``hang`` is how far the object reaches BELOW the tool centre. A grasp
    computed as ``feature_center + approach * surface_inset`` on a cloud that is
    the object's top surface puts the tool centre ``surface_inset`` below that
    surface, and the rest of the object's thickness hangs under it. That is the
    number every clearance over a rim or a divider is really about.

    All zeros when no box is supplied, which is the tool-centre rule this
    script used before it could know any of this.
    """
    if not target_obb:
        return 0.0, 0.0, 0.0
    extent = target_obb["extent"]
    half = sorted((float(extent["x"]), float(extent["y"]), float(extent["z"])), reverse=True)
    return half[0], half[1], max(0.0, 2.0 * half[2] - surface_inset_m)


def _cartesian(ctx: NodeContext, x: float, y: float, z: float, rotation: Quaternion) -> None:
    pose = {"position": {"x": float(x), "y": float(y), "z": float(z)}, "rotation": rotation}
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    except Exception as exc:  # noqa: BLE001
        # A Cartesian servo can hit a joint-limit basin even when another arm
        # configuration reaches the same clear-space pose. Plan the joint-space
        # reconfiguration through the robot's own planner: it owns the arm model
        # and base frame, which a raw curobo.plan_to_pose call would assume.
        logger.warning(
            "[transport_move] straight-line leg to (%.3f, %.3f, %.3f) failed "
            "(%s); falling back to motion.plan_to_pose.",
            x,
            y,
            z,
            exc,
        )
        plan = ctx.tool("motion.plan_to_pose", pose=pose)
        if not plan.get("planned") or not plan.get("trajectory"):
            raise RuntimeError("The robot could not plan the clear-space transport leg") from exc
        ctx.tool(
            "robot.execute_trajectory", trajectory=plan["trajectory"], max_steps_per_waypoint=60
        )
    reached = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    error = math.sqrt(
        (float(reached["x"]) - float(x)) ** 2
        + (float(reached["y"]) - float(y)) ** 2
        + (float(reached["z"]) - float(z)) ** 2
    )
    if error > 0.03:
        raise RuntimeError(
            f"transport leg did not reach its requested pose: error={error:.3f} m, "
            f"requested=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"reached=({reached['x']:.3f}, {reached['y']:.3f}, {reached['z']:.3f})"
        )


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    # FINGERTIP-frame heights. Distance is from_z - target_z, NOT
    # get_ee_pose().z - target_z: on a hand whose reported link sits above the
    # fingertip, the latter overshoots the descent by the TCP offset and rams
    # the object into the container floor.
    dist = float(from_z) - float(target_z)
    if dist <= 0.002:
        return
    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    res = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=js,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=dist,
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if res.get("success") and res.get("trajectory"):
        ctx.tool(
            "robot.execute_trajectory", trajectory=res["trajectory"], max_steps_per_waypoint=60
        )
    else:
        ee_pose = ctx.tool("robot.get_ee_pose")["pose"]
        ee = ee_pose["position"]
        _cartesian(ctx, ee["x"], ee["y"], target_z, ee_pose["rotation"])


def _plan_reachable_interior(
    ctx: NodeContext,
    center: Vec3,
    extent: Vec3,
    z: float,
    preferred_x: float,
    preferred_y: float,
    rotation: Quaternion,
    margin_x: float = 0.015,
    margin_y: float = 0.015,
    nearest_first: bool = False,
) -> tuple[float, float]:
    """Move to a reachable point inside the perceived destination footprint.

    ``margin_x``/``margin_y`` are how far inside the footprint a candidate sits.
    They are per-axis so a caller that knows the size of the thing in the jaws
    can keep the OBJECT inside the walls rather than the tool centre; left at
    their defaults they are the single symmetric inset this used before.

    ``nearest_first`` orders the four corner fallbacks by distance from the
    current hand position. The preferred point is always tried first either
    way: biasing the *preferred* point toward the hand biases release toward a
    divider, which can look like a misplacement.

    **Do not replace the servo with a planner probe.** ``_cartesian`` tries the
    Cartesian servo first and only calls the planner when that raises. The
    planner says "unreachable" about poses the servo reaches all day, so it
    cannot be used as a feasibility oracle, and every attempt to be cleverer
    here has failed on that fact -- a wrist-yaw sweep, a ladder of lower carry
    heights, and probing every candidate from one fixed start state all rescued
    nothing, and the last of them made things worse by spending the servo
    attempt that had been doing the work.
    """
    xr = max(0.0, float(extent["x"]) - margin_x)
    yr = max(0.0, float(extent["y"]) - margin_y)
    cx, cy = float(center["x"]), float(center["y"])
    candidates = [
        (preferred_x, preferred_y),
        (cx - xr, cy - yr),
        (cx - xr, cy + yr),
        (cx + xr, cy - yr),
        (cx + xr, cy + yr),
        (cx, cy),
    ]
    seen: set[tuple[float, float]] = set()
    unique: list[tuple[float, float]] = []
    for x, y in candidates:
        key = (round(x, 5), round(y, 5))
        if key in seen:
            continue
        seen.add(key)
        unique.append((x, y))
    if nearest_first and len(unique) > 1:
        here = ctx.tool("robot.get_ee_pose")["pose"]["position"]
        hx, hy = float(here["x"]), float(here["y"])
        unique = unique[:1] + sorted(unique[1:], key=lambda p: math.hypot(p[0] - hx, p[1] - hy))
    for x, y in unique:
        try:
            _cartesian(ctx, x, y, z, rotation)
            return x, y
        except Exception:  # noqa: BLE001
            logger.debug("destination candidate (%.3f, %.3f) unreachable", x, y)
    raise RuntimeError("CuRobo found no reachable point inside the destination OBB")


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = -1.0,
    lift_z: float = -1.0,
    place_offset: float = 0.06,
    target_obb: OrientedBoundingBox | None = None,
    surface_inset_m: float = 0.012,
    cell_clearance_m: float = 0.010,
    carry_cap_m: float = 0.12,
    align_to_container: bool = False,
    level_lift: bool = False,
    nearest_first_fallbacks: bool = False,
) -> Output:
    """Place a held object into a walled container.

    Everything the caller does not ask for is the tool-centre behaviour this
    script has always had. Four capabilities are opt-in, each measured on a
    compartment-sorting tier where the tool centre was not a good enough proxy
    for the object:

    ``target_obb``
        The box of the thing in the jaws. Given one, the carry height clears
        the object's UNDERSIDE rather than the tool centre (``surface_inset_m``
        says how far it hangs), and the interior margins keep the OBJECT inside
        the walls (``cell_clearance_m`` is the gap left beside it). Left
        ``None``, hang is zero and the margins collapse to the symmetric inset,
        which is the old rule exactly.
    ``carry_cap_m``
        Where height stops being free. Horizontal reach is a circle about the
        base whose radius depends on tool height; measured on one cell over
        z = 0.84..1.36 it peaks across z = 0.96..1.00 and falls away above, so
        the extra height an object's hang asks for is capped before it starts
        costing the reach that placement is already short of.
    ``align_to_container``
        Turn the object's long axis onto the container's long axis during the
        lift, so the turn costs no separate motion. See ``_aligned_rotation``.
    ``level_lift``
        Lift to the full carry height before translating instead of to a
        centimetre below it. The lateral leg is a Cartesian servo, so a lift
        that stops short does not stay short -- it turns the crossing into a
        diagonal that climbs while traversing, which is the one stretch where a
        held object rides lowest over the dividers.
    ``nearest_first_fallbacks``
        Order the corner fallbacks by distance from the hand.
    """
    c = container_obb["center"]
    e = container_obb["extent"]
    place_z = float(c["z"]) + float(e["z"]) - float(place_offset)
    current_pose = ctx.tool("robot.get_ee_pose")["pose"]
    cur = current_pose["position"]
    # Preserve the current grasp orientation throughout lift, transfer, and
    # release; do not add a task-irrelevant yaw over a destination box.
    transport_rotation = current_pose["rotation"]
    # Prefer the container centre. Perceived masks can slightly overestimate
    # a compartment footprint; clamping the current hand position into that
    # footprint biases release toward a divider and can look like a label swap.
    # _plan_reachable_interior still tries alternative interior points if this
    # centre pose is genuinely unreachable.
    bx = float(c["x"])
    by = float(c["y"])
    half_length, half_width, hang = _held_object(target_obb, surface_inset_m)
    # Keep the OBJECT inside the container, not the tool centre. When the object
    # has been turned onto the container's long axis, its half-length is spent
    # on whichever of x/y is the longer and its half-width on the other. With no
    # target_obb both are zero and this is the old symmetric 15 mm inset.
    long_is_y = float(e["y"]) >= float(e["x"])
    margin_x = max(0.015, (half_width if long_is_y else half_length) + cell_clearance_m)
    margin_y = max(0.015, (half_length if long_is_y else half_width) + cell_clearance_m)
    # Derive safe heights from perceived geometry unless the graph supplies
    # explicit world-frame heights. A resting literal belongs to one workspace
    # and is below (or far above) the work surface of the next.
    if transport_z < 0.0:
        # The release pose can itself be above the rim (negative place_offset),
        # so clearance must be measured from place_z rather than added twice.
        # Eight centimetres gives a held object a little room over the source
        # container and any dividers on the way across -- plus whatever hangs
        # below the tool centre, so every object gets the same margin instead
        # of thick ones being dragged while thin ones fly clear. Measured at a
        # flat height, that margin ran from 55 mm for one object down to 6 mm
        # for a thick one. Capped at carry_cap_m; see the docstring.
        transport_z = min(place_z + 0.08 + hang, place_z + carry_cap_m)
    if lift_z < 0.0:
        # Do not add clearance blindly above an already-clear grasp pose: that
        # pushes a fixed-down wrist to the upper workspace boundary. Seven
        # centimetres above the release pose clears the rim; otherwise retain
        # the current height so the lateral leg never descends. Under
        # level_lift the cap stays and only the last centimetre of sag goes.
        lift_z = (
            transport_z
            if level_lift
            else max(place_z + 0.07, min(float(cur["z"]), place_z + 0.08))
        )
    # First lift the grasped item straight up so it clears the table before the
    # lateral move (avoids dragging it across the scene). lift_z is kept BELOW
    # transport_z on purpose, but even a MODEST in-place lift can be infeasible
    # at a far-edge grasp XY: the arm grasps low there yet has almost no
    # vertical room, so BOTH the straight-line and the planned solve fail. So
    # the lift is BEST-EFFORT: if it can't be solved we skip it and go straight
    # up-and-over -- the lateral leg lifts the item anyway as it moves toward
    # the central, reachable container. The lift must never abort the place;
    # only the lateral leg and the descend are essential.
    #
    # Turning the object square to the container rides along with the lift, so
    # the alignment costs no separate motion.
    grasp_rotation = transport_rotation
    aligned = _aligned_rotation(grasp_rotation, e) if align_to_container else None
    squared = False
    for attempt in ([aligned, grasp_rotation] if aligned is not None else [grasp_rotation]):
        try:
            _cartesian(ctx, cur["x"], cur["y"], lift_z, attempt)
            squared = attempt is aligned
            transport_rotation = attempt
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[transport_move] in-place lift to z=%.3f infeasible at (%.3f, %.3f) "
                "(%s); skipping lift, going straight up-and-over.",
                lift_z,
                cur["x"],
                cur["y"],
                exc,
            )
    # THE ALIGNMENT NEVER COSTS A PLACE. It is only ever tried FIRST: if no
    # interior point is reachable while holding the object square to the
    # container, the grasp's own rotation gets exactly the candidates it would
    # have got without this, and the worst case is the unaligned place that
    # would have happened anyway.
    rotations = [aligned, grasp_rotation] if squared else [transport_rotation]
    for index, attempt in enumerate(rotations):
        try:
            bx, by = _plan_reachable_interior(
                ctx,
                c,
                e,
                transport_z,
                bx,
                by,
                attempt,
                margin_x,
                margin_y,
                nearest_first_fallbacks,
            )  # up & over the container
            transport_rotation = attempt
            break
        except RuntimeError:
            if index == len(rotations) - 1:
                raise
            logger.warning(
                "[transport_move] no interior point reachable with the object squared to "
                "the container; falling back to the grasp's own rotation.",
            )
    _descend_linear(ctx, place_z, transport_z)  # descend INTO it (constrained Z)
    return {"place_position": {"x": bx, "y": by, "z": place_z}}
