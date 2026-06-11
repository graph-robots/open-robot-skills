---
name: vlm
description: Free-form and yes/no visual question answering against a hosted
  vision-language model (Anthropic API by default; OpenAI-compatible endpoints
  and Vertex AI selectable by config). Use when a workflow needs scene
  descriptions, semantic checks ("is the drawer open?"), or LLM-judged
  verification of a camera frame — no GPU required.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, vlm, api]}
gap:
  tools:
    - vlm.query: Free-form visual question answering via a hosted VLM.
    - vlm.query_yes_no: Yes/no visual question answering; coerces the model reply to a bool.
---

# vlm

API-backed vision-language Q&A. Zero GPU: every provider is a remote
endpoint. Images are gap-native `uint8 [H, W, 3]` numpy arrays, PNG-encoded
on the wire.

## Providers

Selected by `GAP_VLM_PROVIDER` (default `anthropic`); each tool also accepts
a per-call `provider=` override.

| Provider    | Backend                                            | Config (env)                                              |
|-------------|----------------------------------------------------|-----------------------------------------------------------|
| `anthropic` | Anthropic messages API (gap core dependency)       | `ANTHROPIC_API_KEY`; `GAP_VLM_MODEL` (default `claude-opus-4-8`, see `DEFAULT_ANTHROPIC_MODEL` in `tools.py`) |
| `openai`    | Any OpenAI-compatible chat-completions endpoint    | `GAP_VLM_BASE_URL`, `GAP_VLM_API_KEY`, `GAP_VLM_MODEL`    |
| `vertex`    | Vertex AI direct (AnthropicVertex / google-genai)  | `GAP_VLM_MODEL`, `GAP_VLM_PROJECT_ID`, `GAP_VLM_REGION`   |

The `vertex` provider lazy-imports its SDKs — install the engine's vertex
extra first: `pip install "graph-as-policy[vertex]"`.

## When to use

- Semantic scene checks and checkpoint verification (`vlm.query_yes_no`).
- Free-form scene descriptions or attribute queries (`vlm.query`).
- Prefer `gemini-er.detect` when you need pixel-space bounding boxes, and
  `molmo.point_prompt` when you need a single click point.

## Notes

- `vlm.query_yes_no` coerces with the source-verbatim rule: answer is true
  iff `"yes"` appears in the lowercased reply.
- Requests carry no system prompt and no temperature knob (mirrors the
  original `vlm.v1` proto); the `openai` path pins `temperature: 0.0`.
