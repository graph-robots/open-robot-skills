---
name: vlm_select_box
input_vars: [n, label_list, object_name, object_description]
description: VLM prompt asking which labeled bounding box contains the target.
---
This image shows two side-by-side panels. The left panel (labeled "Original") is the unmodified scene. The right panel shows the same scene with {{ n }} bounding boxes labeled {{ label_list }}.

Which box contains the "{{ object_name }}"?{% if object_description %} It looks like: {{ object_description }}.{% endif %}

Answer with just the letter ({{ label_list }}).
