---
name: vlm_select_zone
input_vars: [n, label_list, placement_description]
description: VLM prompt asking which labeled bounding box best contains a placement region.
---
This image shows two side-by-side panels. The left panel (labeled "Original") is the unmodified scene. The right panel shows the same scene with {{ n }} bounding boxes labeled {{ label_list }}.

The robot needs to release its currently-held object inside this region:

**{{ placement_description }}**

Which labeled box best contains that placement region — the spot the object should land on, NOT the named reference object that the region is described relative to? For example, for "the left compartment of the caddy", choose the box that frames the *left compartment opening*, not a box that frames the whole caddy.

Answer with just the letter ({{ label_list }}), or `none` if no labeled box covers the placement region.
