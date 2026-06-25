# gen1 validation — the declarative-DSL signal path, measured on REAL libero data

Validates the gen1 hypothesis **without the LLM loop**: a weak runtime LLM should be able
to author the per-turn judge in one line (`J.judge(ctx, RELATION, target=..., reference=...)`)
and have a correct, metric verdict come back from **local models only** (SAM3 segmentation +
TAPIP3D 3D tracking + deterministic geometry). Scene: `libero_goal/task0` seed1
("open the middle drawer"); agentview 800×512. Services: SAM3 :8114, TAPIP3D socket.

Reproduce:
```
MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python track_judge/results/gen1_validation/val_sam_resolution.py  <out>
MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python track_judge/results/gen1_validation/val_fullchain.py       <out>
```

## A. Plain-word name → local SAM mask (the DSL's premise)
Does a name a person would say resolve to the right object? (best-mask SAM3 score)

| plain-word prompt | SAM score | area% | verdict |
|---|---|---|---|
| plate         | **0.922** | 1.3% | ✅ correct |
| bowl          | **0.918** | 0.7% | ✅ correct |
| wine bottle   | **0.926** | 0.3% | ✅ correct |
| drawer handle | **0.789** | 0.2% | ✅ correct (mask on the cabinet handle; see viz) |
| wooden cabinet| 0.582 | 9.4% | ✅ plausible |
| cheese        | 0.076 | 0.3% | ❌ low — rejected by gate |
| robot gripper | 0.010 | 1.5% | ❌ fails — **don't resolve the gripper via SAM** |

**Finding 1 (shipped):** the SAM confidence score cleanly separates good resolutions
(0.58–0.93) from failures (≤0.08), so `ctx.segment` now **gates at MIN_SEG_SCORE=0.30** — a
low-confidence mask is worse than none (it would track the wrong region); below the gate we
return `None` → DSL falls back to the whole-frame grid instead of tracking garbage.

**Finding 2 (next):** `robot gripper` is unresolvable by SAM-by-text (0.010). The `grasped`
co-motion relation should take the gripper trajectory from **proprioception** (exact, always
available), not SAM. Tracked as the gen2 candidate.

## B. Full chain on real data: tracked WORLD centroid vs GT (decisive)
176 dense frames → 12 subsampled → TAPIP3D world tracks (12, 576, 3). For each plain-word
object: local SAM mask → grid tracks seeded in it → last-frame world centroid, compared to
the simulator's **privileged** body position (measuring stick only, never in the solve path):

| object | tracked xy (m) | GT xy (m) | **xy error** | tracked z | GT z |
|---|---|---|---|---|---|
| plate       | [0.063, 0.000]  | [0.061, −0.006] | **0.6 cm** | +0.910 | 0.903 |
| bowl        | [−0.129, 0.001] | [−0.102, 0.006] | **2.7 cm** | +0.928 | 0.898 |
| wine bottle | [−0.195, −0.029]| [−0.209, −0.038]| **1.7 cm** | +0.968 | 0.899 |

**The whole pipeline (plain word → SAM → TAPIP3D world track → centroid) localises objects to
0.6–2.7 cm of truth.** DSL relations then read the scene correctly:
- `opened("drawer handle")` → "handle moved 0.2cm of ~15cm" (correct: drawer was not opened)
- `place_in(bowl, plate)` → "bowl 19.2cm from container center; 0% inside" (correct: not placed)
- `next_to(bowl, plate)` → "gap 19.2cm (need <10cm)" (correct)

## C. Positive done=True on real tracks (guard vs a degenerate judge)
A/B (above) are not-yet-done / metric cases — a judge that NEVER says done would also pass
them (the failure mode AGENT.md §7 warns about). `val_positive.py` closes this: it teleports
the bowl onto the plate over a rendered window, tracks it, and checks the relation FLIPS.

- bowl moved [−0.102, 0.006, 0.898] → onto plate top [0.061, −0.006, 0.938]; tracked final
  centroid xy=[0.03, −0.008] z=+0.967.
- `on_top_of(bowl, plate)` → **done=True, progress=1.0**, "on top (dz +5.4cm)". ✅ PASS.

Fix made here: `on_top_of` progress now = fraction of the last-k centroids inside the on-top
region, so the graded signal AGREES with `done` (it read 0.097 while done=True before — a
dz-penalty artifact; now 1.0 when seated).

## Conclusion
The gen1 design is empirically sound: **a one-line declarative judge gets a correct, metric
verdict from local models with zero LLM at judgment time.** The ~1–3 cm sensing noise sits
comfortably below the DSL decision thresholds (place_in pad ~2 cm + footprint, next_to 10 cm,
opened 15 cm vs a 0.2 cm static-noise floor), so it is enough to drive both FINISH and the
next code revision. What remains is the **A/B benchmark** (success + speed vs the VDM baseline),
blocked only on bringing up the runtime LLM endpoint (:8110, mid-migration to the Codex CLI
server) — not on the judge signal, which is now validated.
