# Changes relative to stock cap-x (this milestone)

Two kinds: small **patches to cap-x core** (to make existing capabilities actually run),
and **new code added on top** (the skill + its tooling, all under `capx-se/open_drawer/`).
Full rationale in `../../GAPS.md` / `../../DISCUSSION.md` / `../../DESIGN_SUMMARY.md`.

## A. Patches to cap-x core
- `capx/integrations/libero/__init__.py` — `load_libero_task` now reads the BDDL
  `(:language …)` (same source as the scored goal) and warns when it diverges from suite
  metadata. Fixes the perturbed `_task`/`_swap` suites where the instruction shown to the
  agent ("middle") contradicted the scored goal ("bottom").
- `capx/integrations/franka/libero_reduced.py` — added `get_ee_pose()` (reads
  `robot_cartesian_pos`, real-robot legitimate) and exposed the CuRobo planning functions
  in `functions()` (they had never been callable from the reduced API).
- vendored CuRobo `geom/sdf/world_mesh.py` — warp 1.14 compat shim
  (`wp.torch.device_from_torch` → `wp.device_from_torch`) so collision-world builds stop
  raising `AttributeError`.
- (skill-side bug, not cap-x core) the env video getter is `get_video_frames()`, not
  `get_recorded_frames()` — the latter (used in our `harness.py`) never worked; fixed.

## B. New code added on top (`capx-se/open_drawer/`)
- `robust_skill.py` — the deliverable: sensing-only solve. Gap approach → pose-verified
  descend → result-gated deterministic IK seat → ratcheting over-pull → trajopt retreat.
  Annotated with the dataflow logger (llm-decision vs local-model).
- `horl_planner.py` — HORL RRT+trajopt wrapped for cap-x; collision world built from the
  agentview depth point cloud (compile-once, ~0.6 s/solve). The collision-aware motion
  planner cap-x lacked.
- `runlog.py` — structured run logger: tags each step `llm` (decision boundary — purpose +
  data used [single frame / sequence / proprio / text] + decision) vs `local` (model run).
- `run_robust.py` — headless runner; writes cap-x-style per-run artifacts (multi-view
  replays + `trace.json`/`summary.txt` with the dataflow map). Flags: `--max-steps`,
  `--out-dir`, `--no-video`, `--no-artifacts`.
- `interactive.py` + `ictl.sh` — persistent multi-round sim-driving harness (load env +
  planner once, exec snippets, `reset_env(s)`). The "drive a segment, observe, drive the
  next" loop cap-x had no standalone form of.
- `harness.py` / `skill.py` — baseline primitive skill + standalone runner.
- `paper/` — the cap-x gaps writeup (`gaps.tex` → `gaps.pdf`, tectonic).
