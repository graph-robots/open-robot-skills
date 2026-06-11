---
name: perceiving-object-parts
description: >
  Hierarchical perception for subpart targeting. Detects a parent object
  first (DINO+VLM), crops the camera image to the parent's bounding
  box, then detects and segments the named subpart inside the crop
  (DINO + SAM3), and uncrops + fuses depth to a world-frame
  OBB/mask/cloud — plus the parent object's OBB and cloud for
  downstream placement/collision reasoning. Use when the graspable
  affordance is a subpart of a larger object — pan handle, drawer pull,
  mug rim, moka-pot handle, stove burner — where detecting the subpart
  at full image resolution is unreliable because it occupies few
  pixels.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, subpart, hierarchical, dino, vlm, sam3]
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - vlm.query
    - sam3.segment_text
    - sam3.segment_box
    - geometry.mask_to_world_points
    - geometry.filter_noise
    - geometry.compute_obb
  exit_conditions:
    found: Subpart detected; OBB/mask/cloud bound in subgraph outputs.
    not_found: Parent or subpart not visible in any view. Coordinator routes to abort.
  produces_outputs:
    "<name>_obb": OrientedBoundingBox
    "<name>_mask": Mask
    "<name>_cloud": PointCloud
    # Parent-object geometry — the WHOLE object's OBB and point cloud
    # (e.g. the full frypan when the subpart is its handle). Wired by
    # downstream subgraphs that place / collide against the parent body,
    # not the held subpart — most notably the transport skill's
    # drop-offset pose, which shifts the drop pose so the parent
    # centroid lands at the zone centroid.
    "<name>_parent_obb": OrientedBoundingBox
    "<name>_parent_cloud": PointCloud
  errors:
    - "NOT_FOUND: Parent or subpart not detected in any camera view."
  hard_rules:
    - >
      `parent_prompt` is the WHOLE object DINO localizes; `subpart_prompt`
      is the SUBPART SAM3 segments inside the crop. The two are NOT
      synonyms — passing the same string for both makes the crop step
      redundant and the skill falls back to plain `perceiving-objects`
      behavior. Author distinct prompts (e.g. `parent_prompt="frying pan"`
      + `subpart_prompt="long horizontal handle of the frying pan"`).
    - >
      The skill emits the subpart's OBB/mask/cloud as the primary
      outputs (the parent geometry rides along as
      `<name>_parent_obb`/`<name>_parent_cloud`). If the workflow needs
      the parent as its own grasp/transport target, instantiate
      `perceiving-objects` for the parent and `perceiving-object-parts`
      for the subpart as separate subgraphs.
    - >
      `padding_px` (default 30) controls the crop margin around the
      parent's DINO box. Too tight → SAM3 can't see context and the
      subpart mask leaks out of the crop boundary. Too loose → defeats
      the zoom-in benefit. 30 is a good default for 1024-wide images.
  canonical_scripts:
    - perceive_subpart: scripts/perceive_subpart.py
  examples:
    - title: Canonical subpart-perception subgraph
      path: examples/canonical_subgraph.json
  streaming: false
---

# perceiving-object-parts

Two-step zoom-in perception. The full image gives a small subpart
(e.g. a frypan handle is ~3% of pixels) bad signal-to-noise for SAM3
text segmentation; cropping to the parent first brings the subpart up
to ~30% of pixels in the cropped image — within SAM3's reliable range.

> **About `parent_prompt` and `subpart_prompt`:** they are **literal
> Python strings**, NOT subgraph inputs. They are author-time constants
> per subgraph instance. **DO NOT** declare them in the subgraph's
> top-level `inputs` block, and **DO NOT** write `Ref("in.parent_prompt")`
> or any other `Ref(...)` for them. Write the strings directly on the
> inner script node, e.g.
> `"parent_prompt": "frying pan", "subpart_prompt": "long horizontal handle of the frying pan"`.
> Only `cameras` is a flowed subgraph input (wired from the workflow's
> observation source, identical to `perceiving-objects`'s `cameras`
> input).

## When to use

- The grasp/place affordance is a part of a larger object (pan handle,
  drawer pull, moka-pot grip, mug rim, stove burner).
- Plain `perceiving-objects` with `object_name="handle"` fails because
  there are multiple handles in the scene (drawer pull, microwave
  door, cabinet, ...) and DINO can't disambiguate.

## When NOT to use

- The whole object IS the target (`perceiving-objects` is faster and
  produces a cleaner OBB).
- The subpart spans the majority of the image already (skip the crop).

## Pipeline

```
observation                       # rgb + depth + intrinsics + camera pose
   │
   ▼ grounding-dino.detect(rgb, parent_prompt)
parent_box (BoundingBox2D)         # broadest of the boxes, or VLM-picked
   │
   ▼ crop_rgb_to_box(parent_box, padding=30)
cropped_image                      # H_new × W_new × 3 uint8
   │
   ▼ grounding-dino.detect(crop, subpart_prompt) → sam3.segment_text
cropped_mask                       # subpart mask in crop coordinates
   │
   ▼ uncrop(cropped_mask → original H × W)
full_mask                          # H × W uint8, zeros outside crop
   │
   ▼ geometry.mask_to_world_points(full_mask, depth, K, T_cam)
world_cloud (PointCloud)
   │
   ▼ geometry.filter_noise → geometry.compute_obb
subpart_obb            # the split calls keep the unfiltered-cloud
                       # fallback when DBSCAN strips too many points
```

## Canonical subgraph layout (mirror `perceiving-objects`)

```json
{
  "skill": "perceiving-object-parts",
  "inputs": {},
  "nodes": {
    "observe": {
      "type": "tool",
      "tool": "robot.get_observation"
    },
    "perceive_handle": {
      "type": "script",
      "script": "scripts/perceive_subpart.py",
      "inputs": {
        "cameras":        {"$ref": "observe.cameras"},
        "parent_prompt":  "frying pan",
        "subpart_prompt": "long horizontal handle of the frying pan",
        "padding_px": 30
      }
    },
    "found": {"type": "noop"}
  },
  "edges": [
    ["START", "observe"],
    ["observe", "perceive_handle"],
    ["perceive_handle", "found"],
    ["found", "END"]
  ],
  "outputs": {
    "target_obb":   {"$ref": "perceive_handle.obb"},
    "target_mask":  {"$ref": "perceive_handle.mask"},
    "target_cloud": {"$ref": "perceive_handle.cloud"},
    "target_parent_obb":   {"$ref": "perceive_handle.parent_obb"},
    "target_parent_cloud": {"$ref": "perceive_handle.parent_cloud"}
  },
  "exit": {"router_field": null, "success_values": ["found"]},
  "on_error": "not_found"
}
```

> **HARD RULE — do NOT add a `geometry.filter_and_compute_obb` node
> and bind `target_obb` to it.** Unlike `perceiving-objects`, this
> skill's script already returns a clean, noise-filtered OBB in its
> `obb` output (computed via `geometry.filter_noise` +
> `geometry.compute_obb`, with a fallback to the unfiltered cloud when
> DBSCAN strips a thin part below `min_points`). You MUST bind
> `target_obb` directly to `{"$ref": "perceive_handle.obb"}`. A
> redundant `filter_and_compute_obb` node re-filters an already-tiny
> subpart cloud — DBSCAN on a thin handle shell routinely classifies
> most of it as noise, collapsing the OBB — and loses the script's
> unfiltered-cloud fallback.

Key points:
- `"inputs": {}` — no subgraph-level inputs.
- `cameras` is produced inside the subgraph by `robot.get_observation`, identical to `perceiving-objects`.
- `parent_prompt` and `subpart_prompt` are **literal strings** on the `perceive_handle` node, NOT subgraph inputs.
- Note the output `mask` is the PARENT object's mask (used for collision
  isolation downstream); the subpart's own mask is the `subpart_mask`
  output. The `obb`/`cloud` outputs ARE the subpart's.
