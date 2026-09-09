"""CPU-only contracts for the reusable rigid-task skills."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1] / "skills"

def load(skill, script):
    path = ROOT / skill / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"test_{skill}", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class Context:
    def __init__(self, handlers): self.handlers, self.calls = handlers, []
    def tool(self, name, **kwargs):
        self.calls.append((name, kwargs)); handler=self.handlers[name]
        return handler(**kwargs) if callable(handler) else handler

def pose(x=0.0, y=0.0, z=0.0):
    return {"position":{"x":x,"y":y,"z":z},
            "rotation":{"w":1.0,"x":0.0,"y":0.0,"z":0.0}}


def test_relation_router_is_object_independent_and_profile_driven():
    module=load("routing-manipulation-relations", "choose_placement_mode.py")
    ctx=Context({})
    assert module.run(ctx, "loop_over_shaft") == {"mode": "fixture"}
    assert module.run(ctx, "shaft_into_aperture") == {"mode": "drop"}
    custom=[{"mode":"assembly", "relations":["surface_on_surface"]}]
    assert module.run(ctx, "surface_on_surface", custom) == {"mode":"assembly"}

def test_compute_mate_delegates_typed_relation():
    module=load("computing-feature-mating-poses","compute_mate.py")
    expected={"mate_pose":pose(),"approach_pose":pose(z=0.1),"approach_axis":{"x":0,"y":0,"z":1},
              "seating_distance":0.01,"minimum_clearance":0.004}
    ctx=Context({"robot.get_ee_pose":{"pose":pose()},
                 "geometry.compute_feature_mate":expected})
    fixture={"pose":pose(),"axis":{"x":0.0,"y":0.0,"z":1.0},"radius_outer":0.003}
    out=module.run(ctx,pose(),fixture,"loop_over_shaft",{"frame":"tcp","spheres":[]})
    assert out==expected
    assert ctx.calls[1][1]["relation"]=="loop_over_shaft"
    assert ctx.calls[1][1]["reference_tcp_pose"]==pose()

def test_compute_mate_uses_declared_compliance_profile_not_object_identity():
    module=load("computing-feature-mating-poses","compute_mate.py")
    result={"mate_pose":pose(),"approach_pose":pose(x=.08),"engaged_pose":pose(),
            "approach_axis":{"x":-1,"y":0,"z":0},"seating_distance":.01,
            "minimum_clearance":.004}
    ctx=Context({"robot.get_ee_pose":{"pose":pose()},
                 "geometry.compute_feature_mate":result})
    fixture={"pose":pose(),"axis":{"x":-1.0,"y":0.0,"z":0.0},
             "radius_outer":.003,"usable_length":.06,
             "mating_profile":{"settling_axis":[0,0,1],
                 "crossing_offsets_m":{"approach_pose":.008,"engaged_pose":.018},
                 "seated_radial_offset_m":0.0,"mate_settling_offset_m":.010}}
    held={"position":{"x":0,"y":0,"z":0},
          "rotation":{"w":1,"x":0,"y":0,"z":0},"radius_inner":.013}
    out=module.run(ctx,held,fixture,"loop_over_shaft",{"frame":"tcp","spheres":[]})
    assert abs(out["approach_pose"]["position"]["z"]-.008)<1e-9
    assert abs(out["engaged_pose"]["position"]["z"]-.018)<1e-9
    assert abs(out["mate_pose"]["position"]["z"]-.010)<1e-9

def test_sorting_pair_has_clean_finished_exit():
    module=load("perceiving-sorting-pairs","select_pair.py")
    camera={"name":"overhead","rgb":np.zeros((8,8,3),np.uint8)}
    ctx=Context({"vlm.query":{"text":"DONE"}})
    assert module.run(ctx,{"cameras":[camera]},"sort",json.dumps([{"label":"wrench","obb":{}}]),"source") == {
        "status":"finished"}

def test_sorting_pair_uses_semantic_association_not_index():
    module=load("perceiving-sorting-pairs","select_pair.py")
    boxes=[{"detections":[{"score":.9,"box":{"x1":0,"y1":0,"x2":8,"y2":8}}]},
           {"detections":[{"score":.8,"box":{"x1":2,"y1":2,"x2":6,"y2":6}}]}]
    def detect(**_): return boxes.pop(0)
    cloud={"points":np.array([[0,0,0],[.01,0,0],[0,.01,0]],np.float32)}
    ctx=Context({"vlm.query":{"text":"TARGET: adjustable wrench; LABEL: wrench"},
                 "grounding-dino.detect":detect,
                 "sam3.segment_box":{"masks":[np.ones((8,8),np.uint8)]},
                 "geometry.mask_to_world_points":{"points":cloud},
                 "geometry.filter_and_compute_obb":{"obb":{"center":{}}}})
    camera={"name":"overhead","rgb":np.zeros((8,8,3),np.uint8),"depth":np.ones((8,8)),
            "intrinsics":np.eye(3),"pose":pose()}
    regions=[{"label":"pliers","obb":{"center":{"x":1}}},
             {"label":"wrench","obb":{"center":{"x":2}}}]
    out=module.run(ctx,{"cameras":[camera]},"sort tools",json.dumps(regions),"source bin")
    assert out["status"]=="found" and out["destination_obb"]["center"]["x"]==2

def test_functional_feature_fits_planar_loop():
    module=load("perceiving-functional-features","perceive_feature.py")
    mask=np.ones((2,2),np.uint8)*255
    camera={"rgb":np.zeros((2,2,3),np.uint8),"depth":np.ones((2,2)),"intrinsics":np.eye(3),
            "pose":{"position":{"x":0,"y":0,"z":1},"rotation":{"w":1,"x":0,"y":0,"z":0}}}
    cloud={"points":np.column_stack([np.cos(np.arange(8)),np.sin(np.arange(8)),np.zeros(8)]).astype(np.float32)}
    fit={"pose":pose(),"normal":{"x":0,"y":0,"z":1},"radius":0.01,"planarity":1.0}
    ctx=Context({"sam3.segment_text":{"masks":[mask],"scores":[0.9]},
                 "geometry.mask_to_world_points":{"points":cloud},"geometry.fit_planar_feature":fit})
    out=module.run(ctx,[camera],cloud,mask,"ring","loop")
    assert out["feature"]["kind"]=="loop" and out["feature"]["radius_inner"]==0.01

def test_relational_next_source_uses_declared_semantics_workspace_and_arm_partition():
    module=load("perceiving-relational-correspondences","select_next_source.py")
    low=np.ones((10,10),np.uint8)*255
    high=np.ones((10,10),np.uint8)*127
    def backproject(mask, **_):
        center=np.array([.60, .04, .74]) if int(np.max(mask))==255 else np.array([.80,-.05,.95])
        return {"points":{"points":np.tile(center,(25,1)).astype(np.float32)}}
    ctx=Context({"sam3.segment_text":{"masks":[low,high],"scores":[.7,.95]},
                 "geometry.mask_to_world_points":backproject})
    camera={"name":"overhead","rgb":np.zeros((10,10,3),np.uint8),
            "depth":np.ones((10,10)),"intrinsics":np.eye(3),"pose":pose()}
    out=module.run(ctx,{"cameras":[camera]},"place the widget",
                   [{"kind":"widget","query":"blue item","aliases":["widget"]}],
                   {"x_max":.7,"z_max":.8,"minimum_mask_pixels":50},
                   arm_partition={"split":0.0,"positive_arm":8,"negative_arm":9})
    assert out["status"]=="found" and out["target_kind"]=="widget"
    assert out["arm_id"]==8 and abs(out["destination_anchor_y"]-.04)<1e-6
    assert out["source_id"].startswith("widget-")

def test_functional_feature_parent_strategies_are_category_independent():
    module=load("perceiving-functional-features","perceive_feature.py")
    parent={"points":np.array([[0,0,0],[.1,0,0],[0,.1,0],[.1,.1,.1],
                               [.05,.05,.09],[.04,.05,.09],[.06,.05,.09],[.05,.04,.09]],np.float32)}
    out=module.run(Context({}),[],parent,np.ones((2,2),np.uint8),"declared point","tip",
                   method="parent_landmark",
                   feature_options={"offset":[.01,.02,.03],"axis":[1,0,0]})
    assert out["feature"]["method"]=="parent_landmark"
    assert out["feature"]["axis"]=={"x":1.0,"y":0.0,"z":0.0}
    assert "uncertainty_m" in out["feature"]


def test_object_feature_profile_moves_grasp_toward_feature_with_clearance():
    module=load("perceiving-functional-features","perceive_object_feature.py")
    obb={"center":{"x":0.0,"y":0.0,"z":0.4}}
    module._shift_grasp_toward_feature(
        obb, {"x":0.10,"y":0.0,"z":0.5},
        {"grasp_toward_feature_m":0.025,
         "grasp_toward_feature_axes":["x","y"],
         "minimum_grasp_feature_separation_m":0.05})
    assert np.allclose([obb["center"]["x"],obb["center"]["y"],obb["center"]["z"]],
                       [0.025,0.0,0.4])

    # The clearance bound, rather than a semantic object branch, limits a
    # requested displacement when the feature is already nearby.
    close={"center":{"x":0.0,"y":0.0,"z":0.4}}
    module._shift_grasp_toward_feature(
        close, {"x":0.06,"y":0.0,"z":0.4},
        {"grasp_toward_feature_m":0.025,
         "minimum_grasp_feature_separation_m":0.05})
    assert abs(close["center"]["x"]-0.01)<1e-9

def test_register_held_fallback_still_builds_attachment():
    module=load("registering-held-objects","register_held.py")
    cloud={"points":np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],np.float32)}
    ctx=Context({"robot.get_ee_pose":{"pose":pose()},
                 "geometry.cloud_to_attachment":{"attached_object":{"frame":"tcp","spheres":[]}}})
    out=module.run(ctx,[],cloud,{"pose":pose()},"held object")
    assert out["registration_confidence"]==0.25
    assert out["attached_object"]["frame"]=="tcp"
    assert out["registration_method"]=="rigid_grasp_prior"
    assert out["fallback_used"] is True
    assert out["translation_uncertainty_m"]==0.01

def test_overview_loop_recovery_associates_nearest_center_and_preserves_rotation():
    module=load("registering-held-objects","register_held.py")
    masks=[np.ones((2,2),np.uint8),np.ones((2,2),np.uint8)*2]
    def backproject(mask, **_):
        x=.01 if int(np.max(mask))==1 else .03
        return {"points":{"points":np.tile([x,0,0],(25,1)).astype(np.float32)}}
    def fit(points, **_):
        x=float(np.asarray(points["points"])[0,0])
        measured=pose(x=x)
        measured["rotation"]={"w":0.0,"x":1.0,"y":0.0,"z":0.0}
        return {"pose":measured}
    ctx=Context({"sam3.segment_text":{"masks":masks,"scores":[.6,.9]},
                 "geometry.mask_to_world_points":backproject,
                 "geometry.fit_planar_feature":fit})
    camera={"name":"overhead","rgb":np.zeros((2,2,3),np.uint8),
            "depth":np.ones((2,2)),"intrinsics":np.eye(3),"pose":pose()}
    predicted=module._matrix(pose())
    observed,score=module._recover_loop_from_overview(
        ctx,[camera],predicted,"loop",{})
    assert abs(observed[0,3]-.01)<1e-6 and score==.6
    assert np.allclose(observed[:3,:3],predicted[:3,:3])

def test_direct_grasp_profiles_control_clearance_and_verification():
    align=load("grasping-direct-ik","compute_align_pose.py")
    obb={"center":{"x":0,"y":0,"z":.1},"extent":{"x":.02,"y":.01,"z":.01},
         "orientation":{"w":1,"x":0,"y":0,"z":0}}
    grasp=pose(z=.11)
    out=align.run(Context({}),grasp,obb,{"approach_clearance_m":.08})
    assert abs(out["align_pose"]["position"]["z"]-.19)<1e-9

    execute=load("grasping-direct-ik","execute_grasp_align.py")
    ctx=Context({"robot.go_to_pose":{},"robot.get_ee_pose":{"pose":pose(z=.19)}})
    checked=execute.run(ctx,pose(z=.19),grasp_profile={
        "position_tolerance_m":.01,"angular_tolerance_deg":15.0})
    assert checked["recovery_used"] is False
    assert checked["position_error_m"]==0.0

def test_direct_grasp_cartesian_recovery_only_on_failed_verification():
    execute=load("grasping-direct-ik","execute_grasp_align.py")
    angle=np.deg2rad(21.0)
    bad=pose(z=.19)
    bad["rotation"]={"w":float(np.cos(angle/2)),"x":0.0,"y":0.0,
                     "z":float(np.sin(angle/2))}
    reached=[bad,pose(z=.19)]
    ctx=Context({"robot.go_to_pose":{},"robot.go_to_pose_cartesian":{},
                 "robot.get_ee_pose":lambda **_: {"pose":reached.pop(0)}})
    checked=execute.run(ctx,pose(z=.19),grasp_profile={
        "position_tolerance_m":.01,"angular_tolerance_deg":20.0})
    assert checked["recovery_used"] is True
    assert [name for name,_ in ctx.calls].count("robot.go_to_pose_cartesian")==1

def test_direct_grasp_large_error_splits_translation_from_rotation():
    execute=load("grasping-direct-ik","execute_grasp_align.py")
    angle=np.deg2rad(21.0)
    displaced=pose(x=.08,z=.19)
    displaced["rotation"]={"w":float(np.cos(angle/2)),"x":0.0,"y":0.0,
                           "z":float(np.sin(angle/2))}
    translated=pose(z=.19)
    translated["rotation"]=dict(displaced["rotation"])
    reached=[displaced,translated,pose(z=.19)]
    ctx=Context({"robot.go_to_pose":{},"robot.go_to_pose_cartesian":{},
                 "robot.get_ee_pose":lambda **_: {"pose":reached.pop(0)}})
    checked=execute.run(ctx,pose(z=.19),grasp_profile={
        "position_tolerance_m":.01,"angular_tolerance_deg":20.0,
        "cartesian_recovery_threshold_m":.03})
    corrections=[kwargs for name,kwargs in ctx.calls
                 if name=="robot.go_to_pose_cartesian"]
    assert len(corrections)==2
    assert corrections[0]["pose"]["position"]==pose(z=.19)["position"]
    assert corrections[0]["pose"]["rotation"]==displaced["rotation"]
    assert checked["position_error_m"]==0.0
    assert checked["recovery_used"] is True

def test_depth_support_anchor_failure_falls_back_to_declared_partition():
    module=load("perceiving-functional-features","perceive_fixture_feature.py")
    # The narrow anchor ROI sees only board points. The positive partition also
    # contains the support foreground and therefore recovers a valid feature.
    board=np.column_stack((np.full(240,.85),np.linspace(-.12,.12,240),np.full(240,1.0)))
    support=np.column_stack((np.full(80,.76),np.linspace(.075,.11,80),np.full(80,.98)))
    points=np.vstack((board,support)).astype(np.float32)
    camera={"name":"overhead","rgb":np.zeros((8,8,3),np.uint8),
            "depth":np.ones((8,8),np.float32),"intrinsics":{},"pose":pose()}
    ctx=Context({
        "geometry.mask_to_world_points":{"points":{"points":points}},
        "geometry.filter_and_compute_obb":{"obb":{"center":{},"extent":{},"orientation":{}}},
    })
    profile={"source_kind":"generic","strategy":"depth_support",
             "workspace":{"x_min":.72,"x_max":.90,"y_min":-.12,"y_max":.12,
                          "z_min":.90,"z_max":1.30},
             "anchor_tolerance":.01,"anchor_fallback_partition":True,
             "partition_axis":"y","partition_split":0.0}
    out=module.run(ctx,{"cameras":[camera]},target_kind="generic",
                   destination_anchor_y=.18,fixture_profiles=[profile])
    assert out["fixture_kind"]=="support"
    assert out["hook_tip"]["y"]>0.0

def test_held_motion_selects_minimum_rotation_and_separates_phases():
    module=load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.2)},
                 "motion.plan_joint":{"planned":True,"position_error_m":0.0,
                                      "rotation_error_rad":0.0}})
    feature=pose(z=.1)
    fixture={"pose":pose(x=.4,z=.3),"axis":{"x":0.0,"y":0.0,"z":1.0}}
    out=module.run(ctx,feature,fixture,"shaft_into_aperture",{},
                   {"frame":"tcp","spheres":[{"center":[0,0,0],"radius":.01}]})
    waypoints=out["reorientation_plan"]["waypoints"]
    assert [item["cartesian"] for item in waypoints] == [True,False,False]
    assert abs(waypoints[0]["pose"]["position"]["z"]-.24) < 1e-9
    # Eight symmetry candidates are checked, but zero extra roll wins.
    assert len([call for call in ctx.calls if call[0]=="motion.plan_joint"]) == 8
    assert abs(waypoints[1]["pose"]["rotation"]["w"]-1.0) < 1e-8

def test_held_motion_direct_cartesian_profile_needs_no_motion_planner():
    module=load("planning-held-object-motion", "plan_clearance_motion.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.2)}})
    approach=pose(x=.4,z=.5)
    out=module.run(ctx,pose(),{"pose":pose(),"axis":{"x":0,"y":0,"z":1}},
                   "loop_over_shaft",{}, {"frame":"tcp","spheres":[]},
                   approach_pose=approach,
                   motion_profile={"strategy":"direct_cartesian","escape_distance_m":.06,
                                   "time_scale":1.25,"lift_speed_scale":.6,
                                   "transit_speed_scale":.5})
    plan=out["reorientation_plan"]
    assert plan["time_scale"]==1.25
    assert [w["cartesian"] for w in plan["waypoints"]]==[True,True]
    assert plan["waypoints"][0]["speed_scale"]==.6
    assert all(name!="motion.plan_joint" for name,_ in ctx.calls)

def test_held_motion_engagement_is_linear_only_at_fixture():
    module=load("planning-held-object-motion", "plan_linear_engagement.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.4)}})
    fixture={"pose":pose(x=.4,z=.3),"axis":{"x":0.0,"y":0.0,"z":1.0}}
    out=module.run(ctx,pose(z=.1),fixture,"shaft_into_aperture",{},
                   {"frame":"tcp","spheres":[]})
    waypoints=out["placement_plan"]["waypoints"]
    assert [item["cartesian"] for item in waypoints] == [False,True]
    assert waypoints[-1]["allow_goal_contact"] is True

def test_feature_mate_motion_plans_directly_to_approach():
    module=load("planning-held-object-motion", "plan_from_feature_mate.py")
    ctx=Context({"robot.get_ee_pose":{"pose":pose(z=.2)}})
    approach=pose(x=.5,z=.6)
    out=module.run(ctx,approach,{},
                   {"frame":"tcp","spheres":[{"center":[0.0,.12,0.0],"radius":.01}]})
    waypoints=out["reorientation_plan"]["waypoints"]
    assert [w["cartesian"] for w in waypoints] == [False]
    assert waypoints[0]["pose"] == approach
    assert "allow_start_contact" not in waypoints[0]

def test_loop_over_shaft_relation_crosses_before_contact_seating():
    module=load("planning-held-object-motion", "plan_feature_engagement.py")
    out=module.run(Context({}),pose(z=.3),pose(z=.2),pose(z=.1),"loop_over_shaft",{},
                   {"frame":"tcp","spheres":[]})
    waypoints=out["placement_plan"]["waypoints"]
    assert [w["mode"] for w in waypoints] == ["planned_joint","cartesian_cross","contact_seat"]


def test_feature_mating_executor_reports_contact_and_uncertainty():
    module=load("executing-feature-mating", "execute_waypoints.py")
    target=pose(z=.3)
    ctx=Context({
        "robot.describe_arm":{"solver":{"honours_roll":False}},
        "robot.get_ee_pose":{"pose":target},
        "robot.move_cartesian_until_contact":{"status":"stalled"},
    })
    plan={"relation":"loop_over_shaft", "attached_object":{
              "arm_id":1,"translation_uncertainty_m":.006},
          "waypoints":[{"pose":target,"mode":"planned_joint"},
                       {"pose":target,"mode":"contact_seat"}]}
    out=module.run(ctx,plan,relation="loop_over_shaft")
    assert out["registration_uncertainty_m"]==.006
    assert out["waypoint_reports"][0]["fallback"]=="already_reached"
    assert out["waypoint_reports"][1]["contact_status"]=="stalled"


def test_held_motion_executor_reports_declared_cartesian_transition():
    module=load("executing-held-object-motion", "execute_reorientation.py")
    target=pose(z=.4)
    ctx=Context({"robot.go_to_pose_cartesian":{},
                 "robot.get_ee_pose":{"pose":target}})
    out=module.run(ctx,{"attached_object":{"arm_id":1},
                        "waypoints":[{"pose":target,"mode":"contact_transition"}]})
    assert out["fallback_count"]==0
    assert out["waypoint_reports"][0]["mode"]=="contact_transition"


def test_collision_world_disabled_strategy_is_explicit_and_name_independent():
    module=load("constructing-collision-worlds", "build_collision_world.py")
    ctx=Context({})
    out=module.run(ctx,{"cameras":[]},np.ones((2,2),np.uint8),
                   target_kind="novel_object",
                   collision_profiles=[{"source_kind":"novel_object",
                                        "strategy":"disabled"}])
    assert out=={"world_config":{"meshes":[]},"mesh_names":[],
                "removed_mesh_names":[],"strategy":"disabled"}
    assert ctx.calls==[]


def test_collision_world_profile_controls_reconstruction_and_keep_out():
    module=load("constructing-collision-worlds", "build_collision_world.py")
    vertices=np.column_stack((np.linspace(0,.2,40),np.zeros(40),np.full(40,.5)))
    world={"config":{"meshes":[{"name":"scene","vertices":vertices.tolist(),
                                  "faces":[[0,1,2]]}]}}
    ctx=Context({"geometry.build_world_config":world})
    camera={"name":"overhead","rgb":np.zeros((2,2,3),np.uint8),
            "depth":np.ones((2,2)),"intrinsics":np.eye(3),"pose":pose()}
    profile={"source_kind":"novel_object","strategy":"rgbd_mesh",
             "excluded_masks":{"target":True,"cross_view_target":False,"robot":False},
             "fixture_keep_out":{"use_robot_model":False,"boxes":[{
                 "name":"wall","center":{"x":.5,"y":0,"z":.5},
                 "size":{"x":.1,"y":.1,"z":.1}}]},
             "approach_corridor":{"enabled":False},"obstacle_filter":{}}
    out=module.run(ctx,{"cameras":[camera]},np.ones((2,2),np.uint8),
                   target_kind="novel_object",collision_profiles=[profile])
    assert out["strategy"]=="rgbd_mesh"
    assert [mesh["name"] for mesh in out["world_config"]["meshes"]]==["scene","wall"]
