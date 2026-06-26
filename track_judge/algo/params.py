"""params — DEFAULTS + append-only CHANGELOG for the tracking-judge harness.

Keep CHANGELOG append-only: one entry per generation, newest last. Each entry is
``(gen:int, "<change> — ACCEPTED/REJECTED. hypothesis. metrics. finding.")``.
"""

from __future__ import annotations

DEFAULTS = {
    "window": 12,               # frames sub-sampled per turn for the tracker (>= 12)
    "resolution_factor": 1.0,   # TAPIP3D internal resolution scale
    "num_iters": 6,             # TAPIP3D refinement iterations
    "query_grid": 24,           # default whole-frame query grid (NxN seed)
    "filter_depth": False,      # TAPIP3D depth filtering (off per AGENT.md §5)
}

CHANGELOG: list[tuple[int, str]] = [
    (0, "gen0 — seed: initial self-evaluate tracking-judge harness + judge_lib + prompt."),
    (1, "gen1 — declarative goal-relation DSL (judge_dsl) + ctx.points_of (local SAM, "
        "score-gated) + weak-LLM judge prompt — SIGNAL-PATH VALIDATED on real libero data; "
        "A/B benchmark still PENDING (runtime LLM endpoint :8110 mid-migration to Codex CLI). "
        "hypothesis: a WEAK runtime LLM can author the per-turn judge in ONE line by naming "
        "a RELATION (place_in/on_top_of/opened/lifted/grasped/...) over OBJECTS in plain "
        "words; all segmentation (local SAM), tracking (TAPIP3D) and metric geometry pushed "
        "into deterministic harness; verdict is graded {done,abort,progress,feedback} with a "
        "METRIC residual to drive the next code revision. VALIDATED (results/gen1_validation/): "
        "(A) plain-word->SAM resolves distinct objects at score 0.58-0.93 vs failures <=0.08 "
        "-> added MIN_SEG_SCORE=0.30 gate; (B) full chain plain-word->SAM->TAPIP3D world track "
        "centroid lands 0.6-2.7cm from GT object pose, well under DSL thresholds; DSL relations "
        "read the real scene correctly. FINDING: 'robot gripper' unresolvable by SAM (0.010) -> "
        "gen2 should take the gripper traj from proprioception for grasped(). NEXT: run A/B."),
    (2, "gen2 — all_of() multi-condition helper + DSL coverage map (COVERAGE.md) — offline. "
        "Mapped every base task of the dev suites to a relation: place_in covers all 10 "
        "libero_object (place in basket), on_top_of covers all 10 libero_spatial (bowl on "
        "plate), and place_in/on_top_of/opened cover 8/10 libero_goal -> 28/30 dev tasks with "
        "THREE relations. all_of(ctx, (rel,kwargs), ...) AND-combines for 2-condition goals "
        "like goal[3] 'open the top drawer and put the bowl inside'. Residuals (anticipated "
        "geometry boundary, neither needs a per-turn VLM): goal[5] push-to-front (proximity, "
        "approx via next_to) and goal[7] turn-on-stove (non-spatial state). self-test passes "
        "(relations + dispatch + all_of). Still PENDING A/B (endpoint)."),
    (3, "gen3 — POSITIVE done=True validation on real tracks + on_top_of progress fix — offline. "
        "Guards the degenerate-judge failure mode (a judge that never says done would pass all "
        "the not-done cases): teleport the bowl onto the plate over a rendered window -> TAPIP3D "
        "-> on_top_of(bowl,plate) FLIPS to done=True, progress=1.0 (val_positive.py). Fixed "
        "on_top_of progress to be the last-k region-occupancy fraction so the graded signal "
        "AGREES with done (was 0.097 while done=True via a dz-penalty artifact; now 1.0 seated). "
        "judge now verified in all 3 directions on real data: not-done/metric (A,B), done (C). "
        "Still PENDING A/B (endpoint)."),
    (4, "gen4 — arm-B loop with the runtime LLM SIMULATED (no :8110) + 2 harness fixes it "
        "surfaced. Stood in for cap-agent0's LLM: wrote the 1-line judge "
        "J.judge(ctx,'opened',target='drawer handle') and ran the REAL TrackJudge.judge_turn "
        "seam across 2 turns on libero_goal/task0. RESULT: geometry drives the loop correctly "
        "(turn1 short pull -> not-done -> REGENERATE; turn2 full pull -> done 'moved 15.8cm of "
        "16cm' -> FINISH), judge-vs-GT AGREE both turns. FIX1: opened() keeps a persistent "
        "CLOSED baseline in ctx.state so multi-turn openness accumulates (was per-window -> "
        "never reached travel). FIX2: ctx.points_of densely seeds query_points INSIDE a small "
        "object's mask + dedicated TAPIP3D track (ctx.track_mask) when the 24x24 grid yields "
        "<6 hits -- a drawer handle (~0.2% of frame) got 0-1 grid hits so its centroid was "
        "static background ('moved 0.2cm' while fully open); dense seeding tracks the full "
        "15.8cm. A/B numbers still need the real runtime LLM, but the mechanism is demonstrated."),
    (5, "gen5 — CALL-COUNT A/B across 3 tasks (runtime LLM simulated) + honest-unresolved fix. "
        "Metric of record = LLM CALL COUNT (not speed). 3 tasks x 2-turn loop: arm B judgment "
        "calls = 0 (geometry) vs arm A VDM = 1/turn (6 total); with codegen 1/turn, total LLM "
        "calls arm B 6 vs arm A 12 = -50%. task0(opened) fully GT-validated (joint GT, faithful "
        "stub) REGENERATE->FINISH AGREE both turns; task8(on_top_of) judge geometrically correct "
        "but teleport stub can't trigger LIBERO's contact-based placement GT; task6(place_in) "
        "SAM can't ground 'cream cheese'(0.02). FIX surfaced by task6: _resolve no longer falls "
        "back to the whole-frame grid for a NAMED object (that faked a confident wrong verdict "
        "from background); judge() now returns an HONEST 'could not resolve target/reference' "
        "verdict (progress None). The semantic-grounding boundary reported as uncertainty, not a "
        "fake pass. Self-test still passes."),
    (6, "gen6 — fully GT-validated on_top_of loop (physics-settled placement stub) — no algo "
        "change. gen5's bowl-on-plate GT stayed False because LIBERO's predicate needs physical "
        "CONTACT a teleport can't make (probed: all teleport heights False; 60 settle steps with "
        "robot frozen -> True). Stub now teleports above the plate then lets MuJoCo settle the "
        "bowl. RESULT: on_top_of(bowl,plate) vs TRUE GT -> turn1 lift not-done REGENERATE (GT "
        "False, AGREE); turn2 place+settle done FINISH (GT task_completed True, AGREE). Two of "
        "three core relations now drive the loop correctly against the real LIBERO success "
        "predicate: opened (drawer, gen4) + on_top_of (gen6). place_in's full GT loop needs a "
        "SAM-groundable target (libero_goal cheese scores 0.02 = the grounding residual)."),
    (7, "gen7 — COMPOSITIONAL generalization 'open the top drawer and put the bowl inside' "
        "(libero_goal task3; cap-x VDM FAILS it on both swap+task) + cross-turn resolution "
        "cache. One all_of(opened top-handle, place_in bowl->top-drawer) judges both subgoals. "
        "3-turn loop, judge-vs-GT AGREE every turn: t1 open-half+bowl-out not-done (residual "
        "'handle 7.8/16cm | bowl 29.4cm 0% inside'); t2 open-full+bowl-out not-done ('handle "
        "15.8/16cm | bowl 0% inside' -- the compositional signal: drawer done, bowl remaining); "
        "t3 bowl-in done FINISH ('in container 2.4cm'), GT True. FIX: SAM-by-text is "
        "STATE-DEPENDENT (top drawer 0.33->0.24 once open+occluded) -> first run failed t3 with "
        "honest 'could not resolve'. Added cross-turn cache in ctx.points_of: a resolved "
        "object's last-frame world points are stashed in ctx.state and REUSED when a later turn "
        "can't resolve it (valid: a just-localized static structure hasn't moved). Composition/"
        "geometry is sound; per-frame SAM grounding is the recurring limiter, now mitigated."),
]
