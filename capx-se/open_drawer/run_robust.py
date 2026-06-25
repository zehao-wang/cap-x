"""Run the robust (collision-aware, non-disturbing) drawer-open skill end-to-end
and report task success + how much each other object was disturbed.

Usage (libero venv, perception servers + nothing else needed beyond SAM3+pyroki):
    MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/run_robust.py

Defaults to libero_goal/task0 (open the middle drawer; instruction==goal==middle,
a reachable drawer) so success is meaningful. The object-disturbance read uses
privileged sim poses for MEASUREMENT ONLY (not in the solving path).
"""
from __future__ import annotations

import argparse
import importlib.util as IU
import sys
import types

import numpy as np


def _load_pkg():
    """Load capx-se/open_drawer as a package so robust_skill's relative import works."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = types.ModuleType("open_drawer")
    pkg.__path__ = [here]
    sys.modules["open_drawer"] = pkg
    for name in ("horl_planner", "robust_skill"):
        spec = IU.spec_from_file_location(f"open_drawer.{name}", os.path.join(here, f"{name}.py"))
        mod = IU.module_from_spec(spec)
        sys.modules[f"open_drawer.{name}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["open_drawer.robust_skill"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=30000,
                    help="episode horizon (sim steps); raise to probe gap-F exhaustion")
    ap.add_argument("--video", default=None,
                    help="if set, save an agentview replay (mp4/gif) to this path")
    args = ap.parse_args()

    rs = _load_pkg()
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa: F401
    from capx.integrations.base_api import get_api

    env = FrankaLiberoEnv(args.suite, args.task_id, privileged=False,
                          max_steps=args.max_steps, control_freq=20, enable_render=True)
    env.reset(seed=args.seed)
    if args.video:
        try:
            env.enable_video_capture(True, wrist_camera=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[video] capture unavailable: {exc!r}", flush=True)
    sim = env.handle.env.sim
    api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)

    # MEASUREMENT-ONLY privileged read of other objects' poses
    objs = [b for b in sim.model.body_names
            if any(k in b for k in ("bowl", "cheese", "bottle", "plate")) and "main" in b]
    p0 = {o: sim.data.xpos[sim.model.body_name2id(o)].copy() for o in objs}

    print(f"instruction: {env.handle.task_language!r}", flush=True)

    # MEASUREMENT-ONLY ground-truth probe (drawer qpos + EE) for tuning the pump.
    qadr = sim.model.get_joint_qpos_addr("wooden_cabinet_1_middle_level")
    fns = api.functions()

    def _dbg(tag):
        q = float(sim.data.qpos[qadr])
        ee = fns["get_observation"]()["robot_cartesian_pos"][:3]
        print(f"  [gt] {tag:<10} drawer_qpos={q:+.4f}  ee={np.round(ee, 3)}", flush=True)

    rs.solve_robust(fns, instruction=env.handle.task_language, debug=_dbg)

    disp = {o.split("_1")[0]: round(float(np.linalg.norm(
        sim.data.xpos[sim.model.body_name2id(o)] - p0[o])) * 1000, 1) for o in objs}
    print(f"\n===== TASK SUCCESS={env.task_completed()}  "
          f"max_object_disturbance={max(disp.values()):.1f}mm  {disp} =====", flush=True)

    if args.video:
        try:
            frames = env.get_recorded_frames()
            if frames:
                import imageio
                imageio.mimsave(args.video, frames, fps=20)
                print(f"[video] replay ({len(frames)} frames) -> {args.video}", flush=True)
            else:
                print("[video] no frames captured", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[video] save failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
