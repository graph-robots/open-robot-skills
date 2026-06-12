---
name: molmo
description: Visual pointing and Q&A via the Molmo VLM served from a
  self-hosted vLLM endpoint (OpenAI-compatible API). Use when a workflow needs
  a single pixel coordinate for a named object (point_prompt) or Molmo-grade
  visual question answering and a vLLM server is available; for an API-only
  zero-GPU alternative use the gemini-er bundle.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, pointing, molmo, vllm]}
gap:
  requires: {env: [GAP_MOLMO_BASE_URL]}
  tools:
    - molmo.point_prompt: Point at a named object in an image; returns pixel coordinates.
    - molmo.query: Free-form visual question answering via Molmo.
    - molmo.query_yes_no: Yes/no visual question answering via Molmo; coerces the reply to a bool.
---

# molmo

Molmo visual pointing + Q&A. The bundle itself is zero-GPU (httpx client
only), but Molmo has **no hosted API** — you must serve it yourself with
vLLM and point `GAP_MOLMO_BASE_URL` at it. If you can't self-host, use
`gemini-er.detect` (hosted Gemini Robotics-ER) instead.

## Hosting recipe (vLLM)

Lifted from the dev tree's run book (`training/README.md`):

```bash
# Serve Molmo2-8B on an OpenAI-compatible endpoint
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  python -m vllm.entrypoints.openai.api_server \
  --model allenai/Molmo2-8B \
  --trust-remote-code \
  --dtype bfloat16 \
  --port 8122 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096

# Smoke-test
curl -s http://127.0.0.1:8122/v1/models | jq '.data[0].id'

# Point the bundle at it
export GAP_MOLMO_BASE_URL=http://127.0.0.1:8122/v1
```

Operational notes from the dev tree: pin Molmo to its own GPU when running
alongside other perception services — under heavy parallel evaluation it
becomes the throughput bottleneck if co-located; on a dedicated GPU you can
push `--gpu-memory-utilization 0.85 --max-num-batched-tokens 8192` for ~2×
perception throughput. The server can also run on a remote machine and be
port-forwarded in (the 4090 real-robot profile did exactly this).

## Config

| Env                  | Meaning                              | Default              |
|----------------------|--------------------------------------|----------------------|
| `GAP_MOLMO_BASE_URL` | vLLM OpenAI-compatible base URL      | — (required)         |
| `GAP_MOLMO_MODEL`    | Model name served by vLLM            | `allenai/Molmo2-8B`  |

## Notes

- `molmo.point_prompt` sends the canonical `"Point at <query>"` prompt and
  parses all four Molmo point output formats (Molmo2 `<points coords=...>`,
  Molmo1 `<point x= y=>`, legacy `<points x1= y1= ...>`, plain `x, y`
  fallback), converting normalized coordinates to pixels. `found=False`
  means the model emitted no parseable point.
- `molmo.query_yes_no` coerces with the source-verbatim rule: answer is true
  iff `"yes"` appears in the lowercased reply.
- Backend unreachable after 3 retries raises `ToolError` (route `on_error`).
