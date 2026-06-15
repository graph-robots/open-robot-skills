---
name: sam3
description: Segment Anything 3 — text-, point-, and box-prompted instance
  segmentation, plus a stateful streaming video tracker that carries object
  identity through SAM3's memory bank. Use when a workflow needs open-vocabulary
  masks from an RGB image or needs to follow one object across frames.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, segmentation, tracking, gpu]}
gap:
  requires: {gpu: true, weights: true}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "sam3"]
    protocol: stdio-msgpack
    requires_gpu: true
  tools:
    - sam3.segment_text: Segment all instances matching a text description (masks, scores, boxes best-first).
    - sam3.segment_point: Segment the object at a pixel coordinate (multimask, best-first).
    - sam3.segment_box: Segment within a bounding box, optionally refined by a foreground point.
    - sam3.tracker_init: Open a tracker session seeded with one frame + a text/box/point prompt.
    - sam3.tracker_update: Advance a tracker session by one frame (mask, box, confidence).
    - sam3.tracker_close: Free a tracker session (idempotent).
---

# sam3

The SAM3 image servicer + video-tracker servicer as in-process tools. Images
are RGB uint8 `[H, W, 3]` numpy arrays; masks come back as gap `Mask`
(uint8 `[H, W]`, 0 background / 255 foreground), score-sorted best-first.

## When to use

- `segment_text` for open-vocabulary "find the X" masks (one mask per
  instance; check `scores[0]` — callers typically reject below ~0.3).
- `segment_box` after a detector (e.g. `grounding-dino.detect`) for a
  pixel-accurate mask inside the detection box; add the point prompt
  (`use_point=True`) when a pointing model supplies one.
- `tracker_init` / `tracker_update` / `tracker_close` to follow a single
  target across an observation stream (e.g. for visual servoing).

## Install

```bash
uv sync --extra sam3       # torch + torchvision + the upstream sam3 package
# (pip: pip install -e ".[sam3]")
```

Model weights download on first model build. Device is taken from
`GAP_SAM3_DEVICE` (default `cuda`); the image model also runs on `cpu`
(slow), the video tracker is CUDA-only in practice.

## Gotchas (carried over from the servicers)

- **Lazy singletons**: the image model and the video predictor each load on
  first call and stay resident; importing the bundle never imports torch.
- `segment_text` caps results at `max_results=5` by default — cluttered
  scenes emit 100+ instances (~1 MB/mask at 720p) and downstream consumes
  only the top mask. Pass `max_results<=0` for everything.
- The video tracker JIT-compiles **Triton NMS kernels** via the `CC` env
  var; a stale `CC` (e.g. a Ray env pointing at a non-existent gcc-13)
  surfaces as `FileNotFoundError` inside `tracker_init`. The bundle forces
  `CC` to a real compiler before tracker use (`_ensure_cc_compiler`).
- Tracker prompt precedence is **box > point > text**; a point prompt is
  converted to a small (10% of image) box because the predictor's box path
  is more reliable for init than a single point.
- The tracker is built with `apply_temporal_disambiguation=False` — the
  default hotstart heuristics silently delete the masklet around frame 3 in
  streaming mode (no fresh text re-detection per frame).
- Drift handling in `tracker_update`: a mask-area jump >1.5x the running
  median or confidence <0.30 keeps the LAST GOOD mask and reports
  `confidence=0.0` with `object_present=True` (skip this frame); after 5
  consecutive drift hits `object_present=False` — re-init the tracker.
- Sessions idle longer than 120 s are evicted lazily on the next tracker
  call; an evicted/unknown `tracker_id` raises `ToolError`.
- `tracker_init` returns `object_present=False` with an empty `tracker_id`
  (no exception) when the initial detection finds nothing.
