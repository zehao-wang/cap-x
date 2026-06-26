# gen6 — fully GT-validated `on_top_of` loop (physics-settled placement stub)

gen5 left `on_top_of` only *geometrically* validated: LIBERO's "bowl on plate" GT predicate
needs physical CONTACT, which a positional teleport can't make, so GT stayed False even when the
judge was right. gen6 fixes the STUB (not the judge): teleport the bowl just above the plate,
then let MuJoCo settle it with the robot frozen, so the real GT predicate fires. Probe confirmed
this is a contact predicate — every teleported height read `task_completed=False`, but 60 settle
steps → `True` (bowl z 0.928→0.906).

Run: `MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python demo_ontopof_gt.py <out>`

## Result — `on_top_of(bowl, plate)` vs TRUE GT across a REGENERATE→FINISH loop
| turn | execution | judge | decision | GT `task_completed` | judge-vs-GT |
|---|---|---|---|---|---|
| 1 | lift, not over plate | not-done ("dz +9.6cm, xy off 18.1cm") | REGENERATE | False | ✅ AGREE |
| 2 | place on plate (settle) | **done** ("on top, dz +2.3cm"), prog 1.00 | **FINISH** | **True** | ✅ AGREE |

## Status of GT-validation across the core relations
- **`opened`** (drawer, gen4): fully GT-validated loop (joint GT). ✅
- **`on_top_of`** (bowl→plate, gen6): fully GT-validated loop (contact GT, settled). ✅
- **`place_in`**: geometry + positive validated offline (gen1/gen3); a *fully GT-validated* loop
  needs a place-in task whose target object SAM can ground (libero_goal's cheese scores 0.02 —
  the grounding residual, reported honestly by the judge, not a geometry failure).

Two of the three core relations now drive the agent loop correctly against the **real LIBERO
success predicate**, REGENERATE then FINISH, with zero LLM at judgment.
