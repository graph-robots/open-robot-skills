---
name: vlm_one_shot
input_vars: [object_name, object_description]
description: VLM prompt for a single set-of-marks pick from N letter-labeled boxes — used by the perceiving-objects-oneshot skill.
---
This image shows a scene with several object detections, each drawn as a coloured box labeled with a letter (A, B, C, …).

Pick a letter labeling a box that matches: **{{ object_name }}**.{% if object_description %} It should look like: {{ object_description }}.{% endif %}

Rules:
- **Pick the box that bounds the WHOLE object.** Detectors often draw an extra box around just a part of an object — a printed label, a logo, a sticker, a lid, a cap, a handle. When several boxes overlap the same object, choose the LARGEST one that encloses the entire physical object, NOT the tight box around a label or sub-part. The grasp needs the full object extent.
- If **multiple distinct objects** plausibly match (e.g. several similar items), pick **any one** of them — you do not need to pick "the best". Just return one letter whose box bounds a whole matching object.
- Boxes drawn around the **robot arm**, the **destination container** (basket, bin, bowl), or the **table/floor surface itself** do not count as matches — these are not graspable items to pack.
- Reply ``none`` ONLY if every visible labeled box is a robot / container / surface (i.e. the scene is genuinely empty of items to pack).
- The prompt {{ object_name }} may be generic ("any grocery item", "item on the floor") — match it loosely: anything graspable that isn't the robot or destination qualifies.

Reply with just the letter (or the word ``none``). No punctuation, no explanation.
