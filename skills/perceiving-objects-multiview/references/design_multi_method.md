# How the multiview merge works

Each perception method returns a single-camera result per camera with three
fields: `cloud`, `mask`, `score`. The merge script (`merge.py`) fuses
results from up to three methods (DINO, point, DINO+VLM) into one final
`{cloud, mask, obb}`.

## Algorithm

1. **Collect candidates.** Filter to candidates where `found=True` and
   `mask` is non-empty. If none survive, raise
   `PerceptionFailed("No perception path found 'X'")` so the subgraph
   routes to `not_found`.

2. **Pick the best mask.**
   - With one candidate: use it.
   - With two or more: invoke `select_best.py` (a sibling script in this
     bundle). It composites all candidate masks into a side-by-side panel,
     asks the VLM to pick the panel that most accurately segments the
     target, and returns the chosen index. If the VLM call fails, fall
     back to candidate index 0 (the order is `dino → point → dino_vlm`,
     so the bare DINO result is the default).

3. **Compute the OBB.** Pipe the chosen cloud into
   `geometry.filter_and_compute_obb` with `eps=0.005, min_samples=10`.
   The DBSCAN filter removes outlier points; the OBB extraction returns
   `{"obb": OrientedBoundingBox}` and `merge` unwraps the `obb` field.

## Why select_best is internal, not a separate skill

In the legacy skill library, `select_best_perception` was a top-level
skill. In open-robot-skills it folds into `perceiving-objects-multiview` as a
bundled script — it's only ever called by this composite, and bundling
keeps the dependency local. Other composites that want VLM-disambiguated
mask selection would copy the script into their own bundle (bundles are
self-contained; there is no shared library).

## Why three methods, not two or four

- **DINO alone** is fast but misses small / occluded / low-contrast
  targets.
- **Molmo point** localizes well but its mask quality (when fed to SAM3
  via point prompt) is inconsistent for small objects.
- **DINO + VLM** disambiguates between similar candidates but is slow
  and depends on VLM availability.

Two methods (DINO + Molmo) cover the easy case but fail on clutter. Three
methods cover the cluttered case at the cost of one extra perception
inference. Adding a fourth (e.g. CLIP-based search) hasn't shown empirical
gains in our LIBERO ablations and adds latency.
