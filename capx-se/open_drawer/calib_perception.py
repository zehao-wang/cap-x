"""Calibrate SENSING (solve-frame) for the pick+place phases of task3.

Stages the post-open scene by setting top_level qpos (privileged setup only), then
runs SAM3 the way the solve will -- 'akita bowl' for the pick target and
'drawer'/'open drawer'/'wooden cabinet' for the place target -- and prints the
solve-frame (camera-extrinsic) 3D centroids. This tells me whether the bowl and the
open-drawer interior are reliably sensable and where they land in the frame the
planner/IK use, so the place phase can aim at a real sensed point (no sim-world).
"""
from __future__ import annotations
import argparse
import numpy as np


def mask_pt(fns, d, depth, K, ext):
    p = fns["mask_to_world_points"](d["mask"], depth, K, ext)
    if len(p) == 0:
        return None, 0
    p, _ = fns["filter_noise"](p)
    return (np.median(p, axis=0) if len(p) else None), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--open", action="store_true", help="stage drawer-open via qpos first")
    args = ap.parse_args()

    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa
    from capx.integrations.base_api import get_api
    env = FrankaLiberoEnv("libero_goal", 3, privileged=False, max_steps=30000,
                          control_freq=20, enable_render=True)
    env.reset(seed=args.seed)
    sim = env.handle.env.sim
    api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
    fns = api.functions()

    if args.open:
        a = sim.model.get_joint_qpos_addr("wooden_cabinet_1_top_level")
        sim.data.qpos[a] = -0.16
        sim.forward()
        for _ in range(5):
            sim.step()

    obs = fns["get_observation"]()
    cam = obs["agentview"]
    rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
    K, ext = cam["intrinsics"], cam["pose_mat"]
    print("TCP/ee (solve frame):", np.round(obs["robot_cartesian_pos"][:3], 3))

    for prompt in ("akita bowl", "black bowl", "bowl", "drawer handle",
                   "open drawer", "drawer", "wooden cabinet"):
        masks = sorted(fns["segment_sam3_text_prompt"](rgb, prompt),
                       key=lambda d: -d.get("score", 0.0))
        print(f"\n[{prompt}] {len(masks)} masks")
        for d in masks[:3]:
            p, n = mask_pt(fns, d, depth, K, ext)
            print(f"   score={d.get('score',0):.2f} npts={n} centroid(solve)="
                  f"{np.round(p,3) if p is not None else None}")


if __name__ == "__main__":
    main()
