"""Persistent interactive harness for multi-round open_drawer coding.

Loads the LIBERO env + HORL planner ONCE (pays the ~2 min JAX compile once), then
serves Python snippets from a file-protocol inbox so I can drive the arm a segment
at a time, inspect sensing + GT between moves, and RESET cleanly when a move
collides / moves objects (per the "reset to initial then re-drive via code" rule).

Protocol (token-gated, so a command runs exactly once):
  - client writes  <io>/in.py   whose FIRST line is  `#TOKEN <id>` then the body.
  - server execs the body in a PERSISTENT namespace G (vars survive across calls),
    captures stdout+traceback, writes <io>/out.txt, then writes <io>/done with <id>.
  - client polls <io>/done until it contains <id>, then reads <io>/out.txt.

Namespace G pre-bound: np, env, sim, api, t (=api.functions()), planner, and the
helpers below (reset_env, state, snap, run, jts, ee, tcp, detect_handles,
obstacle_cloud, frame, quat_from_z). GT reads (sim.*) are MEASUREMENT-ONLY.

Start (background):
  MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python \
      capx-se/open_drawer/interactive.py --seed 1 --io <iodir> &
"""
from __future__ import annotations

import argparse
import io as _io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout

import numpy as np

# make `open_drawer` importable (capx-se/ on path) regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_globals(suite, task_id, seed, io_dir):
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa: F401
    from capx.integrations.base_api import get_api
    from open_drawer.horl_planner import HorlPlanner

    env = FrankaLiberoEnv(suite, task_id, privileged=False,
                          max_steps=30000, control_freq=20, enable_render=True)
    env.reset(seed=seed)
    sim = env.handle.env.sim
    api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
    t = api.functions()
    print("building HORL planner (one-time compile) ...", flush=True)
    planner = HorlPlanner(n_spheres=96, timesteps=32, plan_time=3.0,
                          sphere_radius=0.02, world_collision_margin=0.01)

    G = dict(np=np, env=env, sim=sim, api=api, t=t, planner=planner,
             suite=suite, task_id=task_id, seed=seed)

    # GT object bodies (measurement only)
    objs = [b for b in sim.model.body_names
            if any(k in b for k in ("bowl", "cheese", "bottle", "plate")) and "main" in b]
    qadr = sim.model.get_joint_qpos_addr("wooden_cabinet_1_middle_level")
    G["_objs"] = objs
    G["_qadr"] = qadr
    G["_p0"] = {o: sim.data.xpos[sim.model.body_name2id(o)].copy() for o in objs}

    def _S():
        # env.reset() rebuilds the libero simulator, so always re-fetch sim fresh.
        return env.handle.env.sim

    def reset_env(s=None):
        env.reset(seed=seed if s is None else s)
        sm = _S()
        G["sim"] = sm
        G["_p0"] = {o: sm.data.xpos[sm.model.body_name2id(o)].copy() for o in objs}
        print(f"reset to seed {seed if s is None else s}")

    def state(tag=""):
        sm = _S()
        q = float(sm.data.qpos[qadr])
        e = np.asarray(t["get_observation"]()["robot_cartesian_pos"][:3], float)
        disp = {o.split("_1")[0]: round(float(np.linalg.norm(
            sm.data.xpos[sm.model.body_name2id(o)] - G["_p0"][o])) * 1000, 1) for o in objs}
        md = max(disp.values()) if disp else 0.0
        print(f"[state {tag}] drawer_qpos={q:+.4f}  ee={np.round(e,3)}  "
              f"max_disturb={md:.1f}mm  {disp}  SUCCESS={env.task_completed()}")
        return dict(qpos=q, ee=e, disp=disp, max_disturb=md, success=env.task_completed())

    def jts():
        return np.asarray(t["get_observation"]()["robot_joint_pos"][:7], float)

    def ee():
        return np.asarray(t["get_observation"]()["robot_cartesian_pos"][:3], float)

    def tcp(approach):
        return ee() + np.asarray(approach, float) * 0.1034

    def run(traj):
        for cfg in traj:
            t["move_to_joints"](np.asarray(cfg, float))

    def snap(prefix="snap"):
        import imageio.v2 as imageio
        obs = t["get_observation"]()
        for camk, name in [("agentview", "agent"), ("robot0_eye_in_hand", "wrist")]:
            if camk in obs:
                imageio.imwrite(os.path.join(io_dir, f"{prefix}_{name}.png"),
                                obs[camk]["images"]["rgb"])
        print(f"saved {prefix}_agent.png / {prefix}_wrist.png in io dir")

    def _unit(v):
        v = np.asarray(v, float); n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def frame(z, up=np.array([0., 0., 1.])):
        z = _unit(z); y = up - np.dot(up, z) * z
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(z, [1., 0., 0.])
        y = _unit(y); x = _unit(np.cross(y, z)); y = _unit(np.cross(z, x))
        return np.column_stack([x, y, z])

    def quat_from_z(z, up=np.array([0., 0., 1.])):
        return t["rotation_matrix_to_quaternion"](frame(z, up))

    def _mask_pt(mask, depth, K, ext):
        p = t["mask_to_world_points"](mask, depth, K, ext)
        if len(p) == 0:
            return None
        p, _ = t["filter_noise"](p)
        return np.median(p, axis=0) if len(p) else None

    def detect_handles(cam_key="agentview"):
        obs = t["get_observation"](); cam = obs[cam_key]
        rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
        K, ext = cam["intrinsics"], cam["pose_mat"]
        masks = sorted(t["segment_sam3_text_prompt"](rgb, "drawer handle"),
                       key=lambda d: -d.get("score", 0.0))
        found = []
        for d in masks:
            p = _mask_pt(d["mask"], depth, K, ext)
            if p is None or any(np.linalg.norm(p - q) < 0.03 for q in found):
                continue
            found.append(p)
            if len(found) >= 3:
                break
        found.sort(key=lambda q: q[2])
        names = ["bottom", "middle", "top"][:len(found)]
        return dict(zip(names, found)), found

    def obstacle_cloud(handle_pt, cam_key="agentview"):
        obs = t["get_observation"](); cam = obs[cam_key]
        depth, K, ext = cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
        pc = t["transform_points"](t["depth_to_point_cloud"](depth, K).reshape(-1, 3), ext)
        m = ((pc[:, 2] > 0.005) & (pc[:, 2] < 0.30) & (pc[:, 1] > handle_pt[1] + 0.03)
             & (pc[:, 0] > 0.50) & (pc[:, 0] < 0.98))
        return pc[m]

    G.update(reset_env=reset_env, state=state, jts=jts, ee=ee, tcp=tcp, run=run,
             snap=snap, frame=frame, quat_from_z=quat_from_z,
             detect_handles=detect_handles, obstacle_cloud=obstacle_cloud)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--io", required=True)
    args = ap.parse_args()
    os.makedirs(args.io, exist_ok=True)
    for f in ("in.py", "out.txt", "done", "ready"):
        try:
            os.remove(os.path.join(args.io, f))
        except OSError:
            pass

    G = build_globals(args.suite, args.task_id, args.seed, args.io)
    with open(os.path.join(args.io, "ready"), "w") as f:
        f.write("ready")
    print("INTERACTIVE READY", flush=True)

    in_path = os.path.join(args.io, "in.py")
    last_token = None
    while True:
        if os.path.exists(in_path):
            try:
                with open(in_path) as f:
                    src = f.read()
            except OSError:
                time.sleep(0.1); continue
            tok = None
            if src.startswith("#TOKEN "):
                tok = src.splitlines()[0][7:].strip()
                body = "\n".join(src.splitlines()[1:])
            else:
                body = src
            if tok is not None and tok != last_token:
                buf = _io.StringIO()
                try:
                    with redirect_stdout(buf):
                        exec(compile(body, "<cmd>", "exec"), G)
                except Exception:
                    buf.write("\n" + traceback.format_exc())
                with open(os.path.join(args.io, "out.txt"), "w") as f:
                    f.write(buf.getvalue())
                with open(os.path.join(args.io, "done"), "w") as f:
                    f.write(tok)
                last_token = tok
        time.sleep(0.15)


if __name__ == "__main__":
    main()
