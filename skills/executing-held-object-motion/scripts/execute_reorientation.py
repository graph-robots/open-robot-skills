"""Execute a reorientation plan waypoint by waypoint, collision-aware.

Each waypoint is planned with the held object attached and the caller's
collision world, then streamed; ``contact_transition`` waypoints skip the
planner and take the bounded Cartesian servo, because contact is the point.

Two optional layers ride on top of that and both default OFF, so a graph that
binds only ``reorientation_plan`` gets exactly the calls it always got:

- ``arm_id`` names the hand on a bimanual cell. Unset, every robot call is
  made WITHOUT an ``arm_id`` keyword -- not with ``arm_id=0`` -- because the
  promotion-parity gate compares tool names and keyword values, and a
  single-arm graph must replay byte-identically.
- ``execution_profile`` is a per-object-kind table (``speed_scale``,
  ``cartesian_fallback``); a waypoint's own keys override it. ``speed_scale`` is
  passed to the Cartesian servo only when something asked for one, for the
  same reason. ``cartesian_fallback`` turns a planner failure into a bounded
  Cartesian move instead of an error, and is counted.

The report fields (``waypoint_reports``, ``fallback_count``,
``registration_uncertainty_m``) are additive outputs. The per-waypoint pose
error is read off ``robot.get_ee_pose`` only when ``measure_errors`` is set:
that read is a recorded tool call, and a graph that never asked for it must
not gain one per waypoint. ``arrival_tolerance_m`` / ``arrival_tolerance_deg``
(default off) make that read a check: a waypoint that ends farther than either
raises, which routes to ``blocked``. The sharps disposal graph lifts a
marginal pinch this way (3 mm / 2 deg) and refuses to go on otherwise.

Two more layers are driven by the PLAN, not by parameters, so a plan without
them executes exactly as before:

- ``hold_position`` on a ``planned_joint`` waypoint is forwarded to
  ``motion.plan_to_pose``: the solver keeps the hand where it is and searches
  orientation only, for an in-place turn.
- ``turn_search`` on a waypoint that follows a ``planned_joint`` carry (as
  ``planning-held-object-motion``'s ``carry_then_turn`` emits) chooses the carry
  and the turn's wrist yaw together, by planning, before anything moves -- see
  ``_Search``. Both need a connector whose ``motion.plan_to_pose`` accepts
  ``hold_position`` and ``seed_joints`` and whose ``motion.plan_linear`` accepts
  ``start`` and ``seed_joints`` (the sharps benchmark's runtime does; ``tools/curobo`` does
  not), plus ``robot.describe_arm`` and ``robot.forward_kinematics``.

A connector without them refuses the call itself -- the tool registry raises
``ToolArgumentError`` for an argument a tool does not declare, and ``KeyError``
for a tool it does not register -- and the executor falls back rather than
routing ``blocked`` (see ``_unsupported``; a planner that answers
``planned: false``, or fails for any other reason, is not this):

- the turn search refused before anything moved: the turn waypoint's
  ``clearance_first_fallback`` legs are flown instead (``carry_then_turn``
  embeds the plan ``clearance_first`` would have emitted from the same escape
  point), or, with none, the carry and the turn as plain waypoints;
- ``hold_position`` refused: that turn and any later one are re-planned without
  it, a free turn;
- ``robot.forward_kinematics`` refused after the carry: the chosen turn flies as
  planned, without the drift re-plan.

The report of every waypoint flown that way carries ``unsupported_fallback``
(``clearance_first``, ``plain_turn`` or ``free_turn``); otherwise it is absent.

Why the turn search (the sharps disposal graph, ``gap_perception_v2``,
``tray_clutter_t03``, 2026-09-13): the planner's smallest turn put the hand on
the far side of the aperture from the base. The turn planned, but the insertion
strokes after it came within 2-6 mrad of a joint limit and cuRobo refused them
("IK line switched joint branch"), while an offline IK pretest found yaws at the
same placement with 0.4-0.7 rad of margin. Whether one is reachable depends on
the posture the carry ends in: in sweeps s3/s5 a cramped carry end (t06, t08,
t11) left every in-place yaw at 1-6 mrad. With the search the graph solved 16/30
trays and posted 152/176 syringes (sweep s8).
"""

from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from scipy.spatial.transform import Rotation

#: Per-waypoint step budget handed to ``robot.execute_trajectory``. A carry
#: plan is dense, so a waypoint that has not converged in this many control
#: steps is stuck, not slow; the executor moves on rather than timing out.
_MAX_STEPS_PER_WAYPOINT = 60

#: Turn-search defaults. A ``turn_search`` block may override any of them by key.
_SEARCH_DEFAULTS: dict[str, Any] = {
    # Candidate spacing about the fixture axis. The pretest's comfortable band
    # is about 50 deg wide at 10 deg resolution, so 30 deg always lands in it.
    "yaw_step_deg": 30.0,
    # Past this, more joint margin stops outranking a smaller wrist turn.
    "margin_enough_rad": 0.30,
    # A carry-then-turn-in-place candidate this good ends the search; below it
    # the carried-upright candidates are tried.
    "margin_good_rad": 0.15,
    # Below this best margin the carry-then-free-turn candidates are tried too.
    "margin_retry_rad": 0.05,
    # The rise after release, probed from the last stroke's end, along
    # ``retract_probe_axis`` (world). ``tray_clutter_t11`` chose a yaw whose
    # insertion planned but left the shoulder 0.1 rad from its limit, and the
    # 160 mm retract then failed.
    "retract_probe_m": 0.16,
    "retract_probe_axis": (0.0, 0.0, 1.0),
    # Contact margin of the probed strokes after the first (which allow contact).
    "stroke_contact_margin_m": 0.008,
    # Playback stretch for an in-place turn, slower than the plan's time_scale:
    # a pinched object resists a pivot about the pinch axis only by friction.
    # ``tray_clutter_t24``: a 119 deg turn at 2x swung the syringe 60 deg in
    # the fingers within 30 steps.
    "turn_time_scale": 4.0,
    # Playback stretch for a turn that reconfigures the arm (a free turn or an
    # upright carry). Sweep s7's t10 took a 139 deg free turn at 4x that dropped
    # the hand 116 mm and swung the syringe 70 deg.
    "reconfiguring_turn_time_scale": 6.0,
    # If the executed carry ends farther than this from its planned end joints,
    # the chosen turn is re-planned from the live joints.
    "replan_joint_tolerance_rad": 0.05,
}


class Output(TypedDict):
    final_pose: dict[str, Any]
    waypoint_reports: list[dict[str, Any]]
    fallback_count: int
    registration_uncertainty_m: float


def _time_scale(trajectory: dict[str, Any], scale: float) -> dict[str, Any]:
    """Resample a joint path so execution duration changes, not its geometry."""
    waypoints = list(trajectory.get("waypoints") or [])
    if scale <= 1.0 or len(waypoints) < 2:
        return trajectory
    rows = np.asarray([waypoint["positions"] for waypoint in waypoints], dtype=float)
    count = max(2, int(round((len(rows) - 1) * scale)) + 1)
    source, target = np.linspace(0.0, 1.0, len(rows)), np.linspace(0.0, 1.0, count)
    scaled = np.stack([np.interp(target, source, rows[:, j]) for j in range(rows.shape[1])], axis=1)
    return {"waypoints": [{"positions": row.tolist()} for row in scaled]}


def _pose_errors(ctx: NodeContext, pose: dict[str, Any], on_arm: dict[str, Any]) -> tuple[float, float]:
    """Position [m] and rotation [deg] error between the live TCP and *pose*."""
    actual = ctx.tool("robot.get_ee_pose", **on_arm)["pose"]
    position = float(np.linalg.norm([
        float(actual["position"][key]) - float(pose["position"][key]) for key in ("x", "y", "z")
    ]))
    dot = min(1.0, abs(sum(
        float(actual["rotation"][key]) * float(pose["rotation"][key]) for key in ("w", "x", "y", "z")
    )))
    return position, float(np.degrees(2.0 * np.arccos(dot)))


def _rotation(quat: dict[str, float]) -> Rotation:
    return Rotation.from_quat([float(quat["x"]), float(quat["y"]), float(quat["z"]), float(quat["w"])])


def _pose(position: np.ndarray, rotation: Rotation) -> dict[str, Any]:
    q = rotation.as_quat()
    return {"position": dict(zip(("x", "y", "z"), map(float, position), strict=True)),
            "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])}}


def _path(result: dict[str, Any] | None) -> np.ndarray | None:
    result = result or {}
    trajectory = result.get("trajectory") if result.get("planned") else None
    if not trajectory or not trajectory.get("waypoints"):
        return None
    return np.asarray([w["positions"] for w in trajectory["waypoints"]], dtype=float)


def _trajectory(rows: np.ndarray) -> dict[str, Any]:
    return {"waypoints": [{"positions": row.tolist()} for row in rows]}


def _unsupported(error: BaseException) -> bool:
    """Whether *error* is the connector refusing a call's contract, not the planner failing.

    The tool registry raises ``ToolArgumentError`` ("does not accept
    'hold_position'") for an argument a tool never declared and ``KeyError``
    ("Tool 'robot.forward_kinematics' not found") for a tool it does not
    register; after an RPC hop, or from a plain Python callable, the same two
    read as their messages. Matched by class name and message so the script
    needs no import from the runtime's internals.
    """
    if any(cls.__name__ == "ToolArgumentError" for cls in type(error).__mro__):
        return True
    text = str(error)
    if isinstance(error, KeyError):
        return "not found" in text or "not registered" in text
    return "does not accept" in text or "unexpected keyword argument" in text


class _Search:
    """Candidate carries and turns over the fixture, probed with the planners that fly them.

    Every candidate is planned from the posture it would really start in:

    1. **carry, then turn in place.** One carry plan to the transit position in
       the pick orientation; each yaw's in-place turn (``hold_position``) is
       planned from that carry's end joints (``seed_joints``).
    2. **carried upright.** If no in-place turn keeps ``margin_good_rad``, each
       yaw as one free move from the live joints straight to the upright pose.
    3. **carry, then free turn**, if the best is still under ``margin_retry_rad``.

    From each candidate's end joints the insertion strokes (``insert_depths_m``)
    and the retract are probed with ``motion.plan_linear`` (``start`` +
    ``seed_joints``). The kept candidate has the largest joint-limit margin,
    capped at ``margin_enough_rad``, over every planned waypoint, then the
    smallest turn; its planned trajectories are what get executed.
    """

    def __init__(self, ctx: NodeContext, search: dict[str, Any], carry: dict[str, Any], turn: dict[str, Any],
                 world: Any, attachment: Any, on_arm: dict[str, Any]) -> None:
        self.ctx, self.world, self.on_arm = ctx, world, on_arm
        self.options = {**_SEARCH_DEFAULTS, **{k: v for k, v in search.items() if k in _SEARCH_DEFAULTS}}
        self.carry, self.turn = carry, turn
        self.carry_world = world if carry.get("use_world", True) else None
        self.carry_attachment = attachment if carry.get("use_attachment", True) else None
        self.turn_world = world if turn.get("use_world", True) else None
        self.attachment = attachment if turn.get("use_attachment", True) else None
        limits = ctx.tool("robot.describe_arm", **on_arm)["joint_limits"]
        self.lower = np.asarray(limits["lower"], dtype=float)
        self.upper = np.asarray(limits["upper"], dtype=float)
        self.current = _rotation(ctx.tool("robot.get_ee_pose", **on_arm)["pose"]["rotation"])
        self.center = np.asarray(search["fixture_center"], dtype=float)
        axis = np.asarray(search["axis"], dtype=float)
        self.axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        self.offset = np.asarray(search["feature_offset"], dtype=float)
        held_axis = self.current.apply(_rotation(search["feature_rotation"]).apply([0.0, 0.0, 1.0]))
        correction, _ = Rotation.align_vectors(
            [self.axis], [held_axis / max(float(np.linalg.norm(held_axis)), 1.0e-12)])
        self.aligned = correction * self.current
        clearance = float(search.get("transit_clearance_m", 0.08))
        target = search.get("transit_target")
        self.transit = (np.asarray(target, dtype=float) if target is not None
                        else self.center - clearance * self.axis)
        self.depths = [float(d) for d in search.get("insert_depths_m", (-0.020, -0.010, 0.0))]
        retract_axis = np.asarray(self.options["retract_probe_axis"], dtype=float)
        self.retract = float(self.options["retract_probe_m"]) * retract_axis / max(
            float(np.linalg.norm(retract_axis)), 1.0e-12)

    def margin(self, rows: np.ndarray) -> float:
        rows = rows[:, : len(self.lower)]
        return float(np.minimum(rows - self.lower, self.upper - rows).min())

    def hand(self, feature: np.ndarray, rotation: Rotation) -> dict[str, Any]:
        return _pose(feature - rotation.apply(self.offset), rotation)

    def plan_turn(self, pose: dict[str, Any], hold: bool, start: np.ndarray | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "pose": pose, "world_config": self.turn_world, "attached_object": self.attachment,
            "allow_start_contact": bool(self.turn.get("allow_start_contact", False)),
            "allow_goal_contact": bool(self.turn.get("allow_goal_contact", False)),
            "contact_margin": float(self.turn.get("contact_margin", 0.005)),
            "hold_position": hold,
        }
        if start is not None:
            kwargs["seed_joints"] = [float(v) for v in start]
        return self.ctx.tool("motion.plan_to_pose", **kwargs, **self.on_arm)

    def candidate(self, yaw: float, strategy: str, start: np.ndarray | None, hold: bool,
                  base_margin: float = np.inf) -> dict[str, Any]:
        """Plan one yaw's turn from ``start`` (live joints when None), then probe insertion and retract."""
        rotation = Rotation.from_rotvec(self.axis * np.deg2rad(yaw)) * self.aligned
        transit = self.hand(self.transit, rotation)
        entry: dict[str, Any] = {
            "yaw_deg": float(yaw), "strategy": strategy, "hold": hold, "stage": "turn refused",
            "margin": None, "pose": transit, "start": start,
            "turn_deg": float(np.degrees((rotation * self.current.inv()).magnitude())),
        }
        rows = _path(self.plan_turn(transit, hold, start))
        if rows is None:
            return entry
        entry.update(turn=_trajectory(rows), margin=min(base_margin, self.margin(rows)))
        joints, leg_start = rows[-1], transit
        contact_margin = float(self.options["stroke_contact_margin_m"])
        for index, depth in enumerate(self.depths):
            target = self.hand(self.center + depth * self.axis, rotation)
            contact = index > 0
            leg_rows = _path(self.ctx.tool(
                "motion.plan_linear", start=leg_start, end=target, seed_joints=[float(v) for v in joints],
                orientation="lock", world_config=self.turn_world, attached_object=self.attachment,
                allow_start_contact=contact, allow_goal_contact=contact,
                contact_margin=contact_margin if contact else 0.005, **self.on_arm,
            ))
            if leg_rows is None:
                entry["stage"] = f"insertion leg {index} refused"
                return entry
            entry["margin"] = min(entry["margin"], self.margin(leg_rows))
            joints, leg_start = leg_rows[-1], target
        # The retract after release must also be flyable from where the stroke
        # ends; it carries no attachment because the object has been let go.
        entry["stage"] = "inserted, retract refused"
        lifted = _pose(np.array([float(leg_start["position"][k]) for k in ("x", "y", "z")]) + self.retract,
                       _rotation(leg_start["rotation"]))
        retract_rows = _path(self.ctx.tool(
            "motion.plan_linear", start=leg_start, end=lifted, seed_joints=[float(v) for v in joints],
            orientation="lock", world_config=self.turn_world, allow_start_contact=True,
            contact_margin=contact_margin, **self.on_arm,
        ))
        if retract_rows is not None:
            entry["margin"] = min(entry["margin"], self.margin(retract_rows))
            entry["stage"] = "ok"
        return entry

    def best(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        feasible = ([c for c in candidates if c["stage"] == "ok"]
                    or [c for c in candidates if c["stage"] == "inserted, retract refused"])
        if not feasible:
            return None
        enough = float(self.options["margin_enough_rad"])
        return max(feasible, key=lambda c: (min(c["margin"], enough), -c["turn_deg"]))

    def run(self) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
        """The chosen carry trajectory (None when the candidate carries upright itself), turn, and log."""
        yaws = np.arange(-180.0, 180.0, float(self.options["yaw_step_deg"]))
        candidates: list[dict[str, Any]] = []
        carry_rows = carry_margin = None
        for _ in range(max(1, min(int(self.carry.get("max_attempts", 1)), 3))):
            carry_rows = _path(self.ctx.tool(
                "motion.plan_to_pose", pose=self.carry["pose"], world_config=self.carry_world,
                attached_object=self.carry_attachment,
                allow_start_contact=bool(self.carry.get("allow_start_contact", False)),
                allow_goal_contact=bool(self.carry.get("allow_goal_contact", False)),
                contact_margin=float(self.carry.get("contact_margin", 0.005)), **self.on_arm,
            ))
            if carry_rows is not None:
                break
        if carry_rows is not None:
            carry_margin = self.margin(carry_rows)
            candidates += [self.candidate(y, "turn in place", carry_rows[-1], True, carry_margin) for y in yaws]
        best = self.best(candidates)
        if best is None or best["margin"] < float(self.options["margin_good_rad"]):
            candidates += [self.candidate(y, "carried upright", None, False) for y in yaws]
            best = self.best(candidates)
        if carry_rows is not None and (best is None or best["margin"] < float(self.options["margin_retry_rad"])):
            candidates += [self.candidate(y, "free turn", carry_rows[-1], False, carry_margin) for y in yaws]
            best = self.best(candidates)
        log = [{"yaw_deg": c["yaw_deg"], "strategy": c["strategy"], "stage": c["stage"], "margin_rad": c["margin"]}
               for c in candidates]
        if best is None:
            summary = "; ".join(f"{c['yaw_deg']:+.0f} {c['strategy']}: {c['stage']}" for c in candidates)
            raise RuntimeError(f"no carry and wrist yaw lets the planner turn and insert ({summary})")
        carry_trajectory = None if best["start"] is None else _trajectory(carry_rows)
        return carry_trajectory, best, log


def run(
    ctx: NodeContext,
    reorientation_plan: dict[str, Any],
    arm_id: int | None = None,
    execution_profile: dict[str, Any] | None = None,
    registration_uncertainty_m: float | None = None,
    measure_errors: bool = False,
    arrival_tolerance_m: float | None = None,
    arrival_tolerance_deg: float | None = None,
) -> Output:
    waypoints = list(reorientation_plan.get("waypoints") or [])
    if not waypoints:
        raise ValueError("reorientation_plan must contain at least one waypoint")

    profile = execution_profile or {}
    world = reorientation_plan.get("world_config")
    attachment = reorientation_plan.get("attached_object") or {}
    # An arm is only ever NAMED, never assumed: the plan may carry one from a
    # bimanual graph, and a graph that never mentions arms gets no keyword.
    if arm_id is None:
        carried = reorientation_plan.get("arm_id", attachment.get("arm_id"))
        arm_id = None if carried is None else int(carried)
    on_arm: dict[str, Any] = {} if arm_id is None else {"arm_id": int(arm_id)}

    checking = arrival_tolerance_m is not None or arrival_tolerance_deg is not None

    def errors(pose: dict[str, Any], index: int) -> tuple[float | None, float | None]:
        if not (measure_errors or checking):
            return None, None
        position, rotation = _pose_errors(ctx, pose, on_arm)
        if ((arrival_tolerance_m is not None and position > float(arrival_tolerance_m))
                or (arrival_tolerance_deg is not None and rotation > float(arrival_tolerance_deg))):
            raise RuntimeError(
                f"waypoint {index} ended {position * 1000:.1f} mm / {rotation:.1f} deg from its pose"
            )
        return position, rotation

    if registration_uncertainty_m is None:
        registration_uncertainty_m = float(attachment.get("translation_uncertainty_m", 0.0))

    scale = max(1.0, float(reorientation_plan.get("time_scale", 1.0)))
    reports: list[dict[str, Any]] = []
    fallback_count = 0
    final_pose: dict[str, Any] | None = None

    def servo(pose: dict[str, Any], waypoint: dict[str, Any]) -> None:
        speed = waypoint.get("speed_scale", profile.get("speed_scale"))
        extra = {} if speed is None else {"speed_scale": float(speed)}
        ctx.tool("robot.go_to_pose_cartesian", pose=pose, **on_arm, **extra)

    def execute(trajectory: dict[str, Any], stretch: float) -> None:
        ctx.tool(
            "robot.execute_trajectory",
            trajectory=_time_scale(trajectory, stretch),
            max_steps_per_waypoint=_MAX_STEPS_PER_WAYPOINT,
            **on_arm,
        )

    hold_supported = True  # until the connector refuses hold_position once
    index = 0
    while index < len(waypoints):
        waypoint = waypoints[index]
        final_pose = waypoint["pose"]
        mode = waypoint.get("mode")
        if mode is None:
            mode = "planned_linear" if waypoint.get("cartesian", False) else "planned_joint"
        following = waypoints[index + 1] if index + 1 < len(waypoints) else None
        if mode == "planned_joint" and following is not None and following.get("turn_search"):
            # The carry and the turn after it are chosen together (``_Search``).
            try:
                search = _Search(ctx, following["turn_search"], waypoint, following, world, attachment, on_arm)
                carry_trajectory, best, log = search.run()
            except Exception as error:  # noqa: BLE001 -- only a contract refusal is handled; the rest re-raise
                if not _unsupported(error):
                    raise
                # The search only plans, so nothing has moved since the previous
                # waypoint: replace the carry and the turn, and fly the replacement.
                legs = following.get("clearance_first_fallback")
                label = "clearance_first" if legs else "plain_turn"
                print(f"[execute_reorientation] turn search refused by the connector, flying {label}: {error}",
                      flush=True)
                if not legs:
                    legs = [waypoint, {key: value for key, value in following.items() if key != "turn_search"}]
                waypoints[index:index + 2] = [{**leg, "unsupported_fallback": label} for leg in legs]
                continue
            turn, replanned = best["turn"], False
            if carry_trajectory is not None:
                execute(carry_trajectory, scale)
                position_error, rotation_error = errors(waypoint["pose"], index)
                reports.append({"index": index, "mode": mode, "attempts": 1, "fallback": "none",
                                "position_error_m": position_error, "rotation_error_deg": rotation_error})
                try:
                    live = np.asarray(ctx.tool("robot.forward_kinematics", **on_arm)["joint_config"], dtype=float)
                except Exception as error:  # noqa: BLE001 -- only a contract refusal is handled; the rest re-raise
                    if not _unsupported(error):
                        raise
                    print(f"[execute_reorientation] no joint read-back, turning as planned: {error}", flush=True)
                    live = None
                if live is not None:
                    planned_end = np.asarray(best["start"], dtype=float)
                    drift = float(np.max(np.abs(live[: len(planned_end)] - planned_end)))
                    if drift > float(search.options["replan_joint_tolerance_rad"]):
                        rows = _path(search.plan_turn(best["pose"], best["hold"], None))
                        if rows is not None:
                            turn, replanned = _trajectory(rows), True
            in_place = best["strategy"] == "turn in place"
            stretch = float(search.options["turn_time_scale" if in_place else "reconfiguring_turn_time_scale"])
            execute(turn, max(scale, stretch))
            final_pose = best["pose"]
            position_error, rotation_error = errors(final_pose, index + 1)
            reports.append({"index": index + 1, "mode": following.get("mode", "planned_joint"), "attempts": 1,
                            "fallback": "none", "position_error_m": position_error,
                            "rotation_error_deg": rotation_error,
                            "turn_search": {"yaw_deg": best["yaw_deg"], "strategy": best["strategy"],
                                            "margin_rad": best["margin"], "turn_deg": best["turn_deg"],
                                            "replanned": replanned, "candidates": log}})
            index += 2
            continue
        note = ({"unsupported_fallback": waypoint["unsupported_fallback"]}
                if "unsupported_fallback" in waypoint else {})
        if mode == "contact_transition":
            servo(final_pose, waypoint)
            position_error, rotation_error = errors(final_pose, index)
            reports.append({"index": index, "mode": mode, "attempts": 1, "fallback": "none",
                            "position_error_m": position_error, "rotation_error_deg": rotation_error, **note})
            index += 1
            continue
        if mode not in {"planned_joint", "planned_linear"}:
            raise ValueError(f"unsupported reorientation waypoint mode {mode!r}")
        use_attachment = attachment if waypoint.get("use_attachment", True) else None
        use_world = world if waypoint.get("use_world", True) else None
        attempts = max(1, min(int(waypoint.get("max_attempts", 1)), 3))
        asks_hold = mode == "planned_joint" and "hold_position" in waypoint
        if asks_hold and not hold_supported:
            note["unsupported_fallback"] = "free_turn"
        trajectory = None
        for _ in range(attempts):
            planner_tool = (
                "motion.plan_linear" if mode == "planned_linear" else "motion.plan_to_pose"
            )
            inputs = (
                {"end": final_pose, "orientation": "lock"}
                if mode == "planned_linear"
                else {"pose": final_pose}
            )
            if asks_hold and hold_supported:
                inputs["hold_position"] = bool(waypoint["hold_position"])
            inputs.update(
                world_config=use_world,
                attached_object=use_attachment,
                allow_start_contact=bool(waypoint.get("allow_start_contact", False)),
                allow_goal_contact=bool(waypoint.get("allow_goal_contact", False)),
                contact_margin=float(waypoint.get("contact_margin", 0.005)),
                **on_arm,
            )
            try:
                result = ctx.tool(planner_tool, **inputs)
            except Exception as error:  # noqa: BLE001 -- only a refused hold_position is retried
                if "hold_position" not in inputs or not _unsupported(error):
                    raise
                print(f"[execute_reorientation] waypoint {index}: hold_position refused, turning freely: {error}",
                      flush=True)
                hold_supported = False
                note["unsupported_fallback"] = "free_turn"
                inputs.pop("hold_position")
                result = ctx.tool(planner_tool, **inputs)
            trajectory = result.get("trajectory") if result.get("planned") else None
            if trajectory and trajectory.get("waypoints"):
                break
        if not trajectory or not trajectory.get("waypoints"):
            if bool(waypoint.get("cartesian_fallback", profile.get("cartesian_fallback", False))):
                servo(final_pose, waypoint)
                fallback = "cartesian"
                fallback_count += 1
            else:
                raise RuntimeError(
                    f"collision-aware reorientation failed at waypoint {index} ({mode})"
                )
        else:
            ctx.tool(
                "robot.execute_trajectory",
                trajectory=_time_scale(trajectory, scale),
                max_steps_per_waypoint=_MAX_STEPS_PER_WAYPOINT,
                **on_arm,
            )
            fallback = "none"
        position_error, rotation_error = errors(final_pose, index)
        reports.append({"index": index, "mode": mode, "attempts": attempts, "fallback": fallback,
                        "position_error_m": position_error, "rotation_error_deg": rotation_error, **note})
        index += 1

    assert final_pose is not None
    return {
        "final_pose": final_pose,
        "waypoint_reports": reports,
        "fallback_count": fallback_count,
        "registration_uncertainty_m": float(registration_uncertainty_m),
    }
