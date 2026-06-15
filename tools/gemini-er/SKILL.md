---
name: gemini-er
description: Open-vocabulary 2D object detection via the Gemini Robotics-ER
  API — one call returns pixel-space bounding boxes with labels and scores for
  a text query. Use when a workflow needs a detection box to seed segmentation
  (e.g. sam3.segment_box) or coarse localization without any local GPU model.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, detection, gemini, api]}
gap:
  requires: {env_any: [GOOGLE_API_KEY, GEMINI_API_KEY]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "gemini-er"]
    protocol: stdio-msgpack
  tools:
    - gemini-er.detect: Detect pixel-space 2D bounding boxes for a text query via Gemini Robotics-ER.
---

# gemini-er

API-backed 2D detection on Gemini Robotics-ER. Zero GPU. The canonical
perception recipe (from the dev tree's `perceive_gemini_er` workflow script)
is: `gemini-er.detect` → best box by `score` → `sam3.segment_box` on the full
frame at that box → depth projection → OBB fit.

## Install

```bash
uv sync --extra gemini-er    # google-genai  (pip: pip install -e ".[gemini-er]")
export GOOGLE_API_KEY=...    # or GEMINI_API_KEY
```

## Config

| Env                   | Meaning                                  | Default                          |
|-----------------------|------------------------------------------|----------------------------------|
| `GAP_GEMINI_ER_MODEL` | Gemini model name                        | `gemini-robotics-er-1.5-preview` |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | API key (SDK default resolution) | —                  |

## Contract

`gemini-er.detect(image, query)` returns
`{"detections": [{"box": BoundingBox2D, "label": str, "score": float}]}`:

- `box` is pixel-space `{x1, y1, x2, y2}` (top-left → bottom-right), clamped
  to the image bounds. The model emits the Gemini `box_2d` convention
  (`[ymin, xmin, ymax, xmax]` normalized 0–1000); conversion happens here.
- `score` defaults to 1.0 when the model reports none — callers select the
  best detection with `max(..., key=score)`.
- No match (or unparseable model output) → empty `detections`, never an
  error. Treat empty as "object not visible".

## When to use

- Detection boxes for open-vocabulary prompts, no local weights.
- Prefer `molmo.point_prompt` when a single click point is enough, and
  `vlm.query_yes_no` for semantic checks without localization.
