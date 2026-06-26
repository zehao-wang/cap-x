"""Run the compositional skill (open top drawer + put bowl inside) on task3 over one
or more seeds and report task success + SAFETY (disturbance of the OTHER objects --
the bowl is supposed to move, so it is reported separately, not as a safety cost).

The planner (one-time ~2 min JAX compile) and the env are built ONCE and reused
across seeds via env.reset(seed=s), so a 20-seed benchmark pays the compile once.

Usage (libero venv; SAM3+graspnet+pyroki running):
    MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python \
        capx-se/open_drawer/run_compose.py --seeds 1 2 3 --max-steps 60000
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
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = types.ModuleType("open_drawer")
    pkg.__path__ = [here]
    sys.modules["open_drawer"] = pkg
    for name in ("runlog", "horl_planner", "robust_skill", "compose_skill"):
        spec = IU.spec_from_file_location(f"open_drawer.{name}", os.path.join(here, f"{name}.py"))
        mod = IU.module_from_spec(spec)
        sys.modules[f"open_drawer.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["open_drawer.compose_skill"], sys.modules["open_drawer.horl_planner"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "runs_compose"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()

    cs, hp = _load_pkg()
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa: F401
    from capx.integrations.base_api import get_api
    from open_drawer.runlog import RunLogger

    env = FrankaLiberoEnv(args.suite, args.task_id, privileged=False,
                          max_steps=args.max_steps, control_freq=20, enable_render=True)
    planner = hp.HorlPlanner(n_spheres=96, timesteps=32, plan_time=3.0,
                             sphere_radius=0.02, world_collision_margin=0.01)

    results = []
    for seed in args.seeds:
        env.reset(seed=seed)
        save_video = not args.no_video
        if save_video:
            try:
                env.enable_video_capture(True, wrist_camera=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[video] unavailable: {exc!r}", flush=True)
                save_video = False
        sim = env.handle.env.sim
        api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
        fns = api.functions()

        # MEASUREMENT-ONLY privileged reads (never in the solve)
        objs = [b for b in sim.model.body_names
                if any(k in b for k in ("bowl", "cheese", "bottle", "plate")) and "main" in b]
        p0 = {o: sim.data.xpos[sim.model.body_name2id(o)].copy() for o in objs}
        level = {jn: sim.model.get_joint_qpos_addr(jn) for jn in sim.model.joint_names
                 if "cabinet" in jn and "level" in jn}

        log = RunLogger()
        t0 = log.t0
        measurements = []
        _last = {"n": 0}

        bowl_body = next((o for o in objs if "bowl" in o), None)

        def _dbg(tag):
            qs = {jn.split("cabinet_1_")[-1]: round(float(sim.data.qpos[a]), 4)
                  for jn, a in level.items()}
            ee = fns["get_observation"]()["robot_cartesian_pos"][:3]
            g = float(fns["get_observation"]()["robot_joint_pos"][-1])
            bw = (sim.data.xpos[sim.model.body_name2id(bowl_body)].copy()
                  if bowl_body else np.zeros(3))
            n = int(getattr(env, "_sim_step_count", 0))
            d = n - _last["n"]; _last["n"] = n
            measurements.append({"t": round(time.time() - t0, 2), "stage": tag,
                                 "drawer_qpos": qs, "ee": [round(float(x), 3) for x in ee],
                                 "grip": round(g, 3), "bowl": [round(float(x), 3) for x in bw],
                                 "sim_steps": n, "stage_sim_steps": d})
            print(f"  [gt] {tag:<12} qpos={qs} ee={np.round(ee,3)} grip={g:.3f} "
                  f"bowl={np.round(bw,3)} steps={n}(+{d})", flush=True)

        instruction = env.handle.task_language
        print(f"\n##### seed {seed}  instruction={instruction!r} #####", flush=True)
        result = cs.solve_compose(fns, instruction=instruction, planner=planner,
                                  log=log, debug=_dbg)

        disp = {o.split("_1")[0]: round(float(np.linalg.norm(
            sim.data.xpos[sim.model.body_name2id(o)] - p0[o])) * 1000, 1) for o in objs}
        bowl_disp = {k: v for k, v in disp.items() if "bowl" in k}
        safety_disp = {k: v for k, v in disp.items() if "bowl" not in k}
        success = bool(env.task_completed())
        max_safety = max(safety_disp.values()) if safety_disp else 0.0
        print(f"===== seed {seed}: SUCCESS={success}  placed={result.get('placed')}  "
              f"safety_disturb(max,other-objs)={max_safety:.1f}mm  bowl_moved={bowl_disp}  "
              f"all={disp} =====", flush=True)
        results.append({"seed": seed, "success": success, "placed": result.get("placed"),
                        "max_safety_disturb_mm": max_safety, "bowl_disp_mm": bowl_disp,
                        "disp_mm": disp, "result": result})

        if not args.no_artifacts:
            ts = time.strftime("%Y%m%d_%H%M%S")
            rn = (f"{args.suite}_t{args.task_id}_s{seed}__success{int(success)}_"
                  f"safety{int(round(max_safety))}mm__{ts}")
            rd = os.path.join(args.out_dir, rn)
            os.makedirs(rd, exist_ok=True)
            with open(os.path.join(rd, "trace.json"), "w") as f:
                json.dump({"suite": args.suite, "task_id": args.task_id, "seed": seed,
                           "instruction": instruction, "success": success,
                           "result": result, "object_disturbance_mm": disp,
                           "max_safety_disturbance_mm": max_safety,
                           "dataflow_summary": log.summary(), "steps": log.steps,
                           "measurements": measurements,
                           "wall_seconds": round(time.time() - t0, 1)}, f, indent=2,
                          default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                          else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
            if save_video:
                try:
                    from capx.utils.video_utils import _write_video
                    views = {"agentview": env.get_video_frames(clear=True)}
                    if hasattr(env, "get_wrist_video_frames"):
                        views["wrist"] = env.get_wrist_video_frames(clear=True)
                    for v, fr in views.items():
                        if fr:
                            _write_video(fr, rd, suffix=f"{v}_success{int(success)}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[video] save failed: {exc!r}", flush=True)
            print(f"[artifacts] -> {rd}", flush=True)

    # ---- summary ----
    ns = sum(r["success"] for r in results)
    print(f"\n========== BENCHMARK {args.suite}/task{args.task_id}: "
          f"{ns}/{len(results)} success ==========", flush=True)
    for r in results:
        print(f"  seed {r['seed']:>2}: success={r['success']} placed={r['placed']} "
              f"safety_disturb={r['max_safety_disturb_mm']:.1f}mm bowl_moved={r['bowl_disp_mm']}",
              flush=True)
    if not args.no_artifacts:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "benchmark_summary.json"), "w") as f:
            json.dump({"suite": args.suite, "task_id": args.task_id,
                       "n_success": ns, "n_total": len(results), "results": results}, f,
                      indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray)
                      else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()
