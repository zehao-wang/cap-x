"""Privileged probe of libero_goal/task3 ('open the top drawer and put the bowl
inside') -- GROUNDING/DEBUG ONLY (these privileged reads never enter the solve).

Prints, per seed: bowl xpos, the three drawer-handle site/body z's, the cabinet
pose, the top_region success bbox (what `In bowl top_region` checks), and saves an
agentview RGB so I can see where the bowl sits relative to the cabinet.
"""
from __future__ import annotations
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa
    from capx.integrations.base_api import get_api

    env = FrankaLiberoEnv("libero_goal", 3, privileged=False, max_steps=30000,
                          control_freq=20, enable_render=True)
    here = os.path.dirname(os.path.abspath(__file__))

    for s in args.seeds:
        env.reset(seed=s)
        sim = env.handle.env.sim
        api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
        fns = api.functions()

        def bpos(substr):
            ids = [b for b in sim.model.body_names if substr in b]
            out = {}
            for b in ids:
                out[b] = np.round(sim.data.xpos[sim.model.body_name2id(b)], 3)
            return out

        print(f"\n===== seed {s} =====")
        print("instruction:", env.handle.task_language)
        print("-- bowl bodies --");    [print(f"   {k}: {v}") for k, v in bpos("bowl").items()]
        print("-- cabinet bodies --"); [print(f"   {k}: {v}") for k, v in bpos("cabinet").items()]
        print("-- plate/cheese/bottle --")
        for kk in ("plate", "cheese", "bottle"):
            for k, v in bpos(kk).items():
                print(f"   {k}: {v}")

        # drawer-level joints (qpos) + handle sites
        for jn in sim.model.joint_names:
            if "cabinet" in jn and "level" in jn:
                a = sim.model.get_joint_qpos_addr(jn)
                print(f"   joint {jn}: qpos={float(sim.data.qpos[a]):.4f}")
        # sites (handles often are sites)
        for sn in sim.model.site_names:
            if "cabinet" in sn or "handle" in sn or "region" in sn:
                sid = sim.model.site_name2id(sn)
                print(f"   site {sn}: {np.round(sim.data.site_xpos[sid],3)}")

        # the In-region check: find the region object's extent if present
        # save agentview RGB
        obs = fns["get_observation"]()
        rgb = obs["agentview"]["images"]["rgb"]
        try:
            import imageio.v2 as imageio
            p = os.path.join(here, "logs", f"probe_t3_seed{s}.png")
            imageio.imwrite(p, rgb.astype(np.uint8))
            print("   saved", p)
        except Exception as e:
            print("   img save failed", repr(e))


if __name__ == "__main__":
    main()
