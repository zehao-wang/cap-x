"""Standalone harness: run ONE hand-written skill against ONE LIBERO task.

cap-x has no such path today — every skill runs inside the LLM-driven exec() loop
(``capx/envs/tasks/base.py`` / ``trial.py``).  This harness is the minimal
"inject ``functions()`` into a namespace, run the code, report ``check_success()``"
runner that both (a) lets us hand-author/debug skills and (b) is exactly the
shape of the ``eval_fn`` the self-evolve Benchmark Evaluator (GOAL.md §⑤) still
needs.  Run it with the libero venv:

    MUJOCO_GL=egl .venv-libero/bin/python -m capx-se.open_drawer.harness \
        --suite libero_goal_task --task-id 0 --seed 1

(See run.py for a convenience wrapper that also boots the perception servers.)
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np


def _find_drawer_joints(sim):
    """DEBUG-ONLY: cabinet drawer joint addresses, for measuring how far it opened.
    Never used by the solution — purely to validate / find gaps."""
    out = {}
    for jn in sim.model.joint_names:
        low = jn.lower()
        if "cabinet" in low or "drawer" in low:
            try:
                addr = sim.model.get_joint_qpos_addr(jn)
                addr = addr[0] if isinstance(addr, tuple) else addr
                out[jn] = int(addr)
            except Exception:  # noqa: BLE001
                pass
    return out


def run(suite: str, task_id: int, seed: int, *, use_curobo: bool = True,
        instruction: str | None = None, record_video_path: str | None = None) -> bool:
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa: F401  (importing populates the API registry)
    from capx.integrations.base_api import get_api

    print(f"[harness] building env {suite}/task{task_id} ...", flush=True)
    env = FrankaLiberoEnv(
        suite_name=suite,
        task_id=task_id,
        privileged=False,
        # Large horizon: the blocking joint-position controller spends up to
        # max_steps sim-steps per waypoint, so a full grasp+pull easily exceeds
        # the default 8000 (see GAPS.md, step-budget issue).
        max_steps=30000,
        control_freq=20,
        enable_render=True,
        viser_debug=False,
    )
    env.reset(seed=seed)
    print(f"[harness] task prompt: {env.handle.task_language!r}", flush=True)

    if record_video_path:
        try:
            env.enable_video_capture(True, wrist_camera=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] video capture unavailable: {exc!r}")

    print("[harness] constructing FrankaLiberoApiReducedSkillLibrary ...", flush=True)
    api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
    fns = api.functions()
    print(f"[harness] exposed vocabulary ({len(fns)}): {sorted(fns)}", flush=True)

    # ---- privileged DEBUG hook (logging only) ----------------------------- #
    sim = env.handle.env.sim
    drawer_joints = _find_drawer_joints(sim)
    print(f"[harness][debug] drawer joints: {drawer_joints}", flush=True)

    def debug(stage: str):
        qs = {jn: round(float(sim.data.qpos[a]), 4) for jn, a in drawer_joints.items()}
        ee = env.get_observation()["robot_cartesian_pos"][:3]
        print(f"[harness][debug] {stage:>10}  drawer={qs}  ee={np.round(ee,3)}",
              flush=True)

    # ---- run the skill ---------------------------------------------------- #
    from capx_se_skill import solve  # bound below
    ok_run = True
    try:
        result = solve(fns, instruction=instruction or env.handle.task_language,
                       debug=debug, use_curobo=use_curobo)
        print(f"[harness] skill returned: {result}", flush=True)
    except Exception:  # noqa: BLE001
        ok_run = False
        print("[harness] skill raised:", flush=True)
        traceback.print_exc()

    success = bool(env.task_completed())
    print(f"\n[harness] ===== run_ok={ok_run}  TASK SUCCESS={success} =====",
          flush=True)

    if record_video_path:
        try:
            frames = env.get_video_frames()
            if frames:
                import imageio
                imageio.mimsave(record_video_path, frames, fps=20)
                print(f"[harness] video -> {record_video_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[harness] video save failed: {exc!r}")

    return success


def _bind_skill_module():
    """Make ``from capx_se_skill import solve`` work without packaging gymnastics."""
    import importlib.util
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "capx_se_skill", os.path.join(here, "skill.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["capx_se_skill"] = mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal_task")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-curobo", action="store_true")
    ap.add_argument("--instruction", default=None,
                    help="override the task instruction (e.g. 'open the middle drawer')")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    _bind_skill_module()
    success = run(args.suite, args.task_id, args.seed,
                  use_curobo=not args.no_curobo, instruction=args.instruction,
                  record_video_path=args.video)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
