---
name: vlm_select_best
input_vars: [n, n_plus_1, listing, desc_line, object_name, label_list]
description: VLM prompt asking which segmentation panel most accurately covers the target.
---
This image shows {{ n_plus_1 }} side-by-side panels. The first panel (labeled "Original") is the unmodified image for reference. The remaining {{ n }} panels each outline a segmentation mask with a bright green contour:
{{ listing }}
{{ desc_line }}
Which panel most accurately segments the "{{ object_name }}"?
Answer with just the letter ({{ label_list }}).
