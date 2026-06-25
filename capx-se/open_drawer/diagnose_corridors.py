"""Fast diagnostic (NO planner compile): for each seed, characterize whether the
frontal (-Y) approach corridor to the middle handle is blocked, and whether a
top-down corridor is clearer. Uses ONLY sensing for the corridors (agentview
RGB-D + SAM3); reads GT object poses for MEASUREMENT/labeling only.

Run:
  MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python \
      capx-se/open_drawer/diagnose_corridors.py --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import numpy as np


def _mask_pt(t, mask, depth, K, ext):
    p = t["mask_to_world_points"](mask, depth, K, ext)
    if len(p) == 0:
        return None
    p, _ = t["filter_noise"](p)
    return np.median(p, axis=0) if len(p) else None


def detect_middle_handle(t, rgb, depth, K, ext):
    masks = sorted(t["segment_sam3_text_prompt"](rgb, "drawer handle"),
                   key=lambda d: -d.get("score", 0.0))
    found = []
    for d in masks:
        p = _mask_pt(t, d["mask"], depth, K, ext)
        if p is None or any(np.linalg.norm(p - q) < 0.03 for q in found):
            continue
        found.append(p)
        if len(found) >= 3:
            break
    found.sort(key=lambda q: q[2])
    if not found:
        return None
    return found[len(found) // 2]   # middle by height


def obstacle_cloud(t, depth, K, ext, handle_pt):
    pc = t["transform_points"](t["depth_to_point_cloud"](depth, K).reshape(-1, 3), ext)
    # everything on the table that is IN FRONT of the handle face (toward robot)
    m = ((pc[:, 2] > 0.005) & (pc[:, 2] < 0.30) & (pc[:, 1] > handle_pt[1] + 0.03)
         & (pc[:, 0] > 0.50) & (pc[:, 0] < 0.98))
    return pc[m]


def corridor_clearance(obstacles, pts):
    """min distance from any corridor sample point to the obstacle cloud."""
    if len(obstacles) == 0:
        return 9.9
    return float(min(np.linalg.norm(obstacles - p, axis=1).min() for p in pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa: F401
    from capx.integrations.base_api import get_api

    for seed in args.seeds:
        env = FrankaLiberoEnv(args.suite, args.task_id, privileged=False,
                              max_steps=30000, control_freq=20, enable_render=True)
        env.reset(seed=seed)
        sim = env.handle.env.sim
        api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
        t = api.functions()

        obs = t["get_observation"]()
        cam = obs["agentview"]
        rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
        K, ext = cam["intrinsics"], cam["pose_mat"]

        h = detect_middle_handle(t, rgb, depth, K, ext)
        if h is None:
            print(f"[seed {seed}] NO HANDLE DETECTED")
            continue
        obstacles = obstacle_cloud(t, depth, K, ext, h)

        # frontal corridor: pre (10cm in front) -> deep (2cm into face), at handle z
        front = [h + np.array([0, dy, 0]) for dy in np.linspace(0.10, -0.02, 13)]
        # top-down corridor: descend 2.7cm IN FRONT of the face (clears top handle),
        # from 18cm above down to handle z, then hook back onto the bar
        topdown = ([h + np.array([0, 0.027, dz]) for dz in np.linspace(0.18, 0.0, 13)]
                   + [h + np.array([0, dy, 0.0]) for dy in np.linspace(0.027, -0.005, 5)])

        cf = corridor_clearance(obstacles, front)
        ct = corridor_clearance(obstacles, topdown)

        # GT object positions (measurement only) relative to handle
        objs = [b for b in sim.model.body_names
                if any(k in b for k in ("bowl", "cheese", "bottle", "plate")) and "main" in b]
        near = []
        for o in objs:
            p = sim.data.xpos[sim.model.body_name2id(o)]
            d = float(np.linalg.norm(p[:2] - h[:2]))
            near.append((o.split("_1")[0], round(d * 1000), np.round(p - h, 3).tolist()))
        near.sort(key=lambda x: x[1])

        print(f"[seed {seed}] handle@{np.round(h,3)}  obstacles={len(obstacles)}pts")
        print(f"    FRONTAL corridor min-clearance = {cf*1000:6.1f} mm   "
              f"{'BLOCKED' if cf < 0.04 else 'clear'}")
        print(f"    TOPDOWN corridor min-clearance = {ct*1000:6.1f} mm   "
              f"{'BLOCKED' if ct < 0.04 else 'clear'}")
        print(f"    nearest objects (gt, mm / rel-xyz): {near[:3]}")
        env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
