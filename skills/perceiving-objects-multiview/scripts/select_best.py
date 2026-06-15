"""Select the best perception mask from N candidates using VLM comparison."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import Mask

logger = logging.getLogger(__name__)

_LABELS = ["A", "B", "C", "D"]
_OVERLAY_ALPHA = 0.45


def _side_by_side_panels(
    image: np.ndarray,
    masks: list[Any],
    color: tuple[int, int, int] = (50, 255, 50),
    alpha: float = _OVERLAY_ALPHA,
) -> np.ndarray:
    """Create a side-by-side image: Original + each mask on its own labeled panel."""
    import cv2

    base = np.asarray(image)
    overlay_color = np.array(color, dtype=np.float32)

    panels = []

    original_panel = base.copy()
    cv2.putText(
        original_panel, "Original",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        (255, 255, 255), 4, cv2.LINE_AA,
    )
    cv2.putText(
        original_panel, "Original",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        (0, 0, 0), 2, cv2.LINE_AA,
    )
    panels.append(original_panel)

    for i, mask in enumerate(masks):
        panel = base.copy()
        msk = np.asarray(mask) > 0

        msk_u8 = msk.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            msk_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(panel, contours, -1, color, thickness=4)

        light_alpha = 0.15
        panel[msk] = (
            panel[msk].astype(np.float32) * (1 - light_alpha)
            + overlay_color * light_alpha
        ).astype(np.uint8)

        label = _LABELS[i] if i < len(_LABELS) else str(i)
        cv2.putText(
            panel, label,
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            (255, 255, 255), 4, cv2.LINE_AA,
        )
        cv2.putText(
            panel, label,
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            (0, 0, 0), 2, cv2.LINE_AA,
        )
        panels.append(panel)

    composite = np.concatenate(panels, axis=1)
    return composite


def _parse_letter(text: str, n: int) -> int:
    """Parse VLM response for a letter label and return 0-based index."""
    text = text.strip().upper()
    if len(text) == 1 and text in _LABELS[:n]:
        return _LABELS.index(text)

    last_idx = -1
    for match in re.finditer(r'\b([A-D])\b', text):
        letter = match.group(1)
        idx = _LABELS.index(letter) if letter in _LABELS[:n] else -1
        if idx >= 0:
            last_idx = idx

    return last_idx


class Output(TypedDict):
    selected_index: int
    vlm_response: str


def run(
    ctx: NodeContext,
    image: np.ndarray,
    masks: list[Mask],
    object_name: str,
    labels: list[str] | None = None,
    language_description: str = "",
    prompt_override: str = "",
) -> Output:
    """Show each mask on a labeled panel and ask VLM to pick the best."""
    if len(masks) < 2:
        return {"selected_index": -1, "vlm_response": "need at least 2 candidates"}

    if image is None or np.asarray(image).size == 0:
        return {"selected_index": -1, "vlm_response": "missing image"}

    panel_image = _side_by_side_panels(image, masks)

    n = len(masks)
    if prompt_override:
        prompt = prompt_override
    else:
        from gap.skills import load_prompt

        positions = ["left", "right", "center-left", "center-right"]
        parts = []
        for i in range(n):
            label = _LABELS[i] if i < len(_LABELS) else str(i)
            pos = positions[i] if i < len(positions) else f"panel {i}"
            source = (labels[i] if labels and i < len(labels)
                      else f"method {i}")
            parts.append(f"{label} ({pos}): from {source}")
        listing = "\n".join(parts)

        desc_line = f"\nContext: {language_description}\n" if language_description else "\n"

        prompt = load_prompt(
            __package__, "vlm_select_best",
            n=n,
            n_plus_1=n + 1,
            listing=listing,
            desc_line=desc_line,
            object_name=object_name,
            label_list=", ".join(_LABELS[:n]),
        )

    vlm_resp = ctx.tool(
        "vlm.query",
        prompt=prompt, image=panel_image,
    )

    selected = _parse_letter(vlm_resp["text"], n)

    if 0 <= selected < n:
        winner_label = (labels[selected] if labels and selected < len(labels)
                        else f"candidate {selected}")
    else:
        winner_label = "none (fallback to 0)"

    logger.info(
        "select_best: VLM chose %s (index=%d, text=%r) for '%s'",
        winner_label, selected, vlm_resp["text"][:80], object_name,
    )

    return {
        "selected_index": selected,
        "vlm_response": vlm_resp["text"],
    }
