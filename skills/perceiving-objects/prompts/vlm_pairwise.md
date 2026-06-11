---
name: vlm_pairwise
input_vars: [object_name, object_description]
description: VLM prompt for one binary "A or B?" comparison of two upscaled object crops in the perception tournament.
---
This image shows two zoomed-in object close-ups side by side, labeled A (left) and B (right). Each is a crop of one object from the same scene.

Exactly one of them is the "{{ object_name }}".{% if object_description %} Appearance hint (approximate, drawn from a catalog — the rendered object may differ): {{ object_description }}. If the hint conflicts with what you see, match the object name.{% endif %} Which one is it — A or B? If neither is a clear match, pick the closer one.

Answer with just the letter (A or B).
