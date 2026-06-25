"""Run the robust (collision-aware, non-disturbing) drawer-open skill end-to-end,
report task success + per-object disturbance, and persist cap-x-style run artifacts.

Usage (libero venv; perception servers SAM3+pyroki running):
    MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/run_robust.py

Defaults to libero_goal/task0 (open the middle drawer; instruction==goal==middle,
a reachable drawer) so success is meaningful. The object-disturbance + drawer-qpos
reads use privileged sim poses for MEASUREMENT ONLY (never in the solving path).

Artifacts (mirrors cap-x trial.py: a per-run dir under --out-dir, named by outcome):
    runs/<suite>_t<id>_s<seed>__success<0/1>_dist<NN>mm__<ts>/
        trace.json    key-step skill log + privileged measurements + result + summary
        summary.txt   human-readable key steps (the "key step log")
        video_success<0/1>.mp4   the replay (always saved unless --no-video)
Replays + key-step logs are ON by default (per the cap-x convention).
"""
from __future__ import annotations

import argparse
import importlib.util as IU
import json
import os
import sys
import time
import types

import numpy as np


def _load_pkg():
    """Load capx-se/open_drawer as a package so robust_skill's relative import works."""
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = types.ModuleType("open_drawer")
    pkg.__path__ = [here]
    sys.modules["open_drawer"] = pkg
    for name in ("runlog", "horl_planner", "robust_skill"):
        spec = IU.spec_from_file_location(f"open_drawer.{name}", os.path.join(here, f"{name}.py"))
        mod = IU.module_from_spec(spec)
        sys.modules[f"open_drawer.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["open_drawer.robust_skill"]


def _cabinet_level_joints(sim):
    """All cabinet drawer-level joints (measurement-only), e.g. *_bottom/middle/top_level."""
    out = {}
    for jn in sim.model.joint_names:
        if "cabinet" in jn and "level" in jn:
            try:
                out[jn] = sim.model.get_joint_qpos_addr(jn)
            except Exception:  # noqa: BLE001
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=30000,
                    help="episode horizon (sim steps); raise to probe gap-F exhaustion")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs"),
                    help="root for per-run artifact directories")
    ap.add_argument("--no-video", action="store_true", help="skip saving the replay video")
    ap.add_argument("--no-artifacts", action="store_true",
                    help="skip writing the run dir entirely (just print)")
    args = ap.parse_args()

    rs = _load_pkg()
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa: F401
    from capx.integrations.base_api import get_api

    save_video = not args.no_video
    env = FrankaLiberoEnv(args.suite, args.task_id, privileged=False,
                          max_steps=args.max_steps, control_freq=20, enable_render=True)
    env.reset(seed=args.seed)
    if save_video:
        try:
            # multi-view: record BOTH agentview and the wrist (eye-in-hand) camera, so
            # every view of the observation gets a saved replay.
            env.enable_video_capture(True, wrist_camera=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[video] capture unavailable: {exc!r}", flush=True)
            save_video = False
    sim = env.handle.env.sim
    api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
    fns = api.functions()

    # MEASUREMENT-ONLY privileged reads (object poses + drawer joints); never in the solve.
    objs = [b for b in sim.model.body_names
            if any(k in b for k in ("bowl", "cheese", "bottle", "plate")) and "main" in b]
    p0 = {o: sim.data.xpos[sim.model.body_name2id(o)].copy() for o in objs}
    level_joints = _cabinet_level_joints(sim)

    instruction = env.handle.task_language
    print(f"instruction: {instruction!r}", flush=True)

    # ---- artifact collectors -------------------------------------------------
    # RunLogger tags each key step llm (decision boundary, with the data it uses) vs
    # local (SAM3 / pyroki-IK / HORL planner), so the trace shows the agent's data-flow.
    from open_drawer.runlog import RunLogger
    log = RunLogger()
    t_start = log.t0
    measurements: list[dict] = []  # privileged per-stage measurements (measurement-only)

    _last_steps = {"n": 0}

    def _dbg(tag):
        qs = {jn: round(float(sim.data.qpos[a]), 4) for jn, a in level_joints.items()}
        ee = fns["get_observation"]()["robot_cartesian_pos"][:3]
        # horizon accounting: every blocking sub-step counts against max_steps, so track
        # the per-stage sim-step delta to see WHERE the budget goes (gap-F diagnosis).
        nstep = int(getattr(env, "_sim_step_count", 0))
        d_step = nstep - _last_steps["n"]
        _last_steps["n"] = nstep
        rec = {"t": round(time.time() - t_start, 2), "stage": tag,
               "drawer_qpos": qs, "ee": [round(float(x), 3) for x in ee],
               "sim_steps": nstep, "stage_sim_steps": d_step}
        measurements.append(rec)
        print(f"  [gt] {tag:<10} drawer_qpos={qs}  ee={np.round(ee, 3)}  "
              f"sim_steps={nstep} (+{d_step})", flush=True)

    result = rs.solve_robust(fns, instruction=instruction, log=log, debug=_dbg)

    disp = {o.split("_1")[0]: round(float(np.linalg.norm(
        sim.data.xpos[sim.model.body_name2id(o)] - p0[o])) * 1000, 1) for o in objs}
    success = bool(env.task_completed())
    max_dist = max(disp.values()) if disp else 0.0
    print(f"\n===== TASK SUCCESS={success}  max_object_disturbance={max_dist:.1f}mm  "
          f"{disp} =====", flush=True)

    if args.no_artifacts:
        return

    # ---- write the per-run artifact dir (cap-x convention) -------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = (f"{args.suite}_t{args.task_id}_s{args.seed}__"
                f"success{int(success)}_dist{int(round(max_dist))}mm__{ts}")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    trace = {
        "suite": args.suite, "task_id": args.task_id, "seed": args.seed,
        "max_steps": args.max_steps, "instruction": instruction,
        "success": success, "object_disturbance_mm": disp,
        "max_object_disturbance_mm": max_dist, "result": result,
        "wall_seconds": round(time.time() - t_start, 1),
        # where the agent would call the LLM, on what data, and what it decides:
        "dataflow_summary": log.summary(),
        "steps": log.steps, "measurements": measurements,
    }
    with open(os.path.join(run_dir, "trace.json"), "w") as f:
        json.dump(trace, f, indent=2, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else float(o)
                  if isinstance(o, (np.floating, np.integer)) else str(o))

    dsum = log.summary()
    _K = {"llm": "LLM", "local": "loc", "act": "act", "info": "   "}
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"{args.suite}/task{args.task_id} seed{args.seed}\n")
        f.write(f"instruction: {instruction!r}\n")
        f.write(f"SUCCESS={success}  max_object_disturbance={max_dist:.1f}mm  {disp}\n")
        f.write(f"result: {result}\n")
        f.write(f"\ndataflow: {dsum['n_llm_decisions']} llm-decision boundaries, "
                f"{dsum['n_local_model_calls']} local-model calls "
                f"(models: {', '.join(dsum['local_models_used'])})\n")
        f.write("\n--- key steps (kind | step | <= data used | -> decision) ---\n")
        for s in log.steps:
            line = f"  [{s['t']:>6.2f}s] {_K.get(s['kind'],'   ')} {s['msg']}"
            if s.get("inputs"):
                line += f"   <= {s['inputs']}"
            if s.get("decides"):
                line += f"   -> {s['decides']}"
            f.write(line + "\n")
        f.write("\n--- llm-decision data map (for agent design) ---\n")
        for d in dsum["llm_decisions"]:
            f.write(f"  {d['purpose']}\n      uses: {d['inputs']}\n      decides: {d['decides']}\n")
        f.write("\n--- privileged measurements (measurement-only) ---\n")
        for m in measurements:
            f.write(f"  [{m['t']:>6.2f}s] {m['stage']:<10} qpos={m['drawer_qpos']} ee={m['ee']}\n")

    if save_video:
        try:
            from capx.utils.video_utils import _write_video
            views = {"agentview": env.get_video_frames(clear=True)}
            if hasattr(env, "get_wrist_video_frames"):
                views["wrist"] = env.get_wrist_video_frames(clear=True)
            for view, frames in views.items():
                if frames:
                    _write_video(frames, run_dir, suffix=f"{view}_success{int(success)}")
                else:
                    print(f"[video] no {view} frames captured", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[video] save failed: {exc!r}", flush=True)

    print(f"[artifacts] -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
