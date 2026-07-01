---
name: vlm
description: Free-form and yes/no visual question answering against a hosted
  vision-language model (OpenRouter API by default; Vertex AI Gemini
  selectable by config). Use when a workflow needs scene descriptions,
  semantic checks ("is the drawer open?"), or LLM-judged verification of a
  camera frame — no GPU required.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, vlm, api]}
gap:
  # The vlm bundle inherits provider/model/project/region from GAP_LLM_*
  # and GOOGLE_CLOUD_* when its own GAP_VLM_* knobs are unset (see
  # tools.py:_resolve_provider). env_any lists the inheritance sources
  # so a shell configured solely for the agent LLM doesn't trip this
  # readiness gate. `gap check` (and `gap check --probe`) report the
  # RESOLVED config — use them to verify the bundle will dispatch where
  # you expect.
  requires: {env_any: [
    OPENROUTER_API_KEY, GAP_VLM_API_KEY, GAP_VLM_BASE_URL,
    GAP_VLM_PROJECT_ID, GOOGLE_CLOUD_PROJECT,
    GAP_LLM_PROVIDER,
  ]}
  serving:
    command: ["python", "-m", "gap_core.rpc.server", "--bundle", "vlm"]
    protocol: stdio-msgpack
  tools:
    - vlm.query: Free-form visual question answering via a hosted VLM.
    - vlm.query_yes_no: Yes/no visual question answering; coerces the model reply to a bool.
---

# vlm

API-backed vision-language Q&A. Zero GPU: every provider is a remote
endpoint. Images are gap-native `uint8 [H, W, 3]` numpy arrays, PNG-encoded
on the wire.

## Providers

Selected by `GAP_VLM_PROVIDER` (default `openrouter`); each tool also accepts
a per-call `provider=` override.

| Provider     | Backend                                             | Config (env)                                              |
|--------------|-----------------------------------------------------|-----------------------------------------------------------|
| `openrouter` | OpenRouter's OpenAI-compatible chat-completions API | `OPENROUTER_API_KEY` (or `GAP_VLM_API_KEY`); `GAP_VLM_MODEL` (default `gemini-3.1-flash-lite-preview`, see `DEFAULT_MODEL` in `tools.py`); set `GAP_VLM_BASE_URL` to point at another OpenAI-compatible server (e.g. a local vLLM) |
| `vertex`     | Vertex AI via google-genai (Gemini models)          | `GAP_VLM_MODEL`, `GAP_VLM_PROJECT_ID`, `GAP_VLM_REGION`   |

The `vertex` provider lazy-imports google-genai — install the engine's vertex
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
  original `vlm.v1` proto); both providers pin `temperature: 0.0`.
