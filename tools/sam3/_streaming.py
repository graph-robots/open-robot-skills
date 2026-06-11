"""Online frame addition for the SAM3 video predictor.

Ported verbatim from the dev tree's SAM3-tracker streaming module. Imports
torch + the upstream ``sam3`` package at module level — only import this
module lazily, from inside the tracker tool functions.

The upstream ``Sam3VideoPredictor.start_session(resource_path)`` is offline-batch:
it calls ``model.init_state`` which pre-loads every frame and pre-allocates
fixed-size per-frame state arrays (``find_inputs``, ``previous_stages_out``,
``per_frame_*``, etc.). Live tracking needs a frame-at-a-time API.

Rather than modify the upstream model, we extend an existing inference_state
in place by appending one new frame's image tensor + per-frame slots. The
encoder format must match the list-of-PIL-images path of
``load_resource_as_video_frames`` (io_utils.py:42-67) exactly so the model
sees identical embeddings to a fully-batched session.

Usage from the tracker tools:
    1. ``predictor.handle_request({"type": "start_session",
        "resource_path": [first_pil]})`` to get a session_id.
    2. ``predictor.handle_request({"type": "add_prompt", ...,
        "bounding_boxes": [[x,y,w,h]]})`` to seed the prompt — outputs hold
        the initial mask.
    3. Per new frame: ``add_new_frame(predictor.model, state, pil)`` to extend
        the inference state by one frame, then
        ``predictor.handle_stream_request({"type": "propagate_in_video",
        ..., "start_frame_index": new_idx, "max_frame_num_to_track": 1})`` to
        propagate the memory bank onto the new frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from sam3.model.data_misc import FindStage, convert_my_tensors
from sam3.model.utils.misc import copy_data_to_device

# Embedding dims used by `_construct_initial_input_batch` in the upstream
# `Sam3VideoInference` (sam3_video_inference.py:122-138). Must match exactly.
_INPUT_BOX_EMBED_DIM = 258
_INPUT_POINT_EMBED_DIM = 257


@torch.inference_mode()
def add_new_frame(model: Any, inference_state: dict, image_pil: Image.Image) -> int:
    """Append one PIL frame to a streaming SAM3 inference state.

    Mirrors the encoding of `load_resource_as_video_frames` for list inputs
    (resize → uint8 → /255.0 → permute → fp16 → mean/std normalize) and the
    per-frame state shape used by `_construct_initial_input_batch`.

    Returns the new frame index (= num_frames - 1 after the append).
    """
    img_mean_t = torch.tensor(
        model.image_mean, dtype=torch.float16,
    )[:, None, None]
    img_std_t = torch.tensor(
        model.image_std, dtype=torch.float16,
    )[:, None, None]
    img_np = np.array(
        image_pil.convert("RGB").resize((model.image_size, model.image_size)),
    )
    img_t = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).to(
        dtype=torch.float16,
    )
    img_t -= img_mean_t
    img_t /= img_std_t
    img_t = img_t.unsqueeze(0)  # (1, 3, H, W)

    input_batch = inference_state["input_batch"]
    img_t = img_t.to(input_batch.img_batch.device)
    input_batch.img_batch = torch.cat([input_batch.img_batch, img_t], dim=0)

    new_frame_idx = inference_state["num_frames"]

    # Inherit text_ids from the existing frames. add_prompt writes the
    # text-vs-visual prompt mode into all current frames at prompt time
    # (sam3_video_inference.py:874-875); new frames must agree or they'll
    # silently revert to text-grounding regardless of the original prompt.
    inherited_text_id = 0
    if len(input_batch.find_inputs) > 0:
        existing = input_batch.find_inputs[0]
        existing_text_ids = getattr(existing, "text_ids", None)
        if existing_text_ids is not None and existing_text_ids.numel() > 0:
            inherited_text_id = int(existing_text_ids.flatten()[0].item())

    new_stage = FindStage(
        img_ids=[new_frame_idx],
        text_ids=[inherited_text_id],
        input_boxes=[torch.zeros(_INPUT_BOX_EMBED_DIM)],
        input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
        input_boxes_label=[torch.empty(0, dtype=torch.long)],
        input_points=[torch.empty(0, _INPUT_POINT_EMBED_DIM)],
        input_points_mask=[torch.empty(0)],
        object_ids=[],
    )
    new_stage = convert_my_tensors(new_stage)
    new_stage = copy_data_to_device(new_stage, model.device, non_blocking=True)
    input_batch.find_inputs.append(new_stage)
    input_batch.find_targets.append(None)
    input_batch.find_metadatas.append(None)

    inference_state["previous_stages_out"].append(None)
    inference_state["per_frame_raw_point_input"].append(None)
    inference_state["per_frame_raw_box_input"].append(None)
    inference_state["per_frame_visual_prompt"].append(None)
    inference_state["per_frame_geometric_prompt"].append(None)
    inference_state["per_frame_cur_step"].append(0)

    inference_state["num_frames"] += 1
    # `is_image_only` was set True in init_state when the session was seeded
    # with a single PIL image (is_image_type([pil]) == True). Once a second
    # frame arrives we are tracking a video, not a single image.
    inference_state["is_image_only"] = False

    # Each tracked object also has its own sub-state under
    # `tracker_inference_states` (a list of dicts produced by
    # Sam3TrackerPredictor.init_state). Those have their own scalar
    # `num_frames`; if we don't bump them in lockstep, the inner
    # `tracker.propagate_in_video` computes
    # `end_frame_idx = min(start + 0, num_frames - 1)` and yields zero
    # iterations — surfacing as
    # "UnboundLocalError: out_frame_idx where it is not associated with a value"
    # at sam3_video_base.py:1133.
    for tracker_state in inference_state.get("tracker_inference_states", []) or []:
        if isinstance(tracker_state, dict) and "num_frames" in tracker_state:
            existing = tracker_state["num_frames"]
            # tracker.num_frames may legitimately be None if it was lazily
            # initialised; only bump when we have a count to bump.
            if isinstance(existing, int):
                tracker_state["num_frames"] = existing + 1

    return new_frame_idx


def first_object_from_outputs(
    outputs: dict,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None, float, bool]:
    """Pull (mask, box_pixels, score, present) for the first object from a
    post-processed SAM3 video output dict.

    SAM3 returns `out_binary_masks` (N, H, W), `out_probs` (N,),
    `out_boxes_xywh` (N, 4) in normalized image coordinates. For the
    single-target tracker we use the highest-score mask; if there are zero
    objects on this frame, return present=False.
    """
    masks = outputs.get("out_binary_masks")
    probs = outputs.get("out_probs")
    if masks is None or probs is None or masks.shape[0] == 0:
        return None, None, 0.0, False

    if masks.ndim != 3:
        masks = masks.reshape(-1, *masks.shape[-2:])

    best = int(np.argmax(np.asarray(probs)))
    mask = np.asarray(masks[best]).astype(bool)
    score = float(probs[best])

    ys, xs = np.where(mask)
    if xs.size == 0:
        return mask, None, score, False
    box = (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )
    return mask, box, score, True
