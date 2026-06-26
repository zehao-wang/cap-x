"""Empirically pin down the bowl-placement target for task3 (privileged; debug only).

Opens the top drawer (set top_level qpos), reads the open top_region site, then
teleports the bowl to candidate points and reports which satisfy env.task_completed().
Tells me exactly where the bowl center must end up (esp. the Z), and the open-region
world position, so the place phase has a concrete sensing target to aim at.
"""
from __future__ import annotations
import numpy as np


def set_body_pos(sim, body, pos):
    # bowl is a free joint; set its qpos (xyz + quat)
    jname = None
    bid = sim.model.body_name2id(body)
    # find the free joint of this body
    for jn in sim.model.joint_names:
        try:
            if sim.model.body_jntadr[bid] >= 0 and sim.model.joint_name2id(jn) == sim.model.body_jntadr[bid]:
                jname = jn
        except Exception:
            pass
    if jname is None:
        return False
    adr = sim.model.get_joint_qpos_addr(jname)
    lo = adr[0] if isinstance(adr, tuple) else adr
    sim.data.qpos[lo:lo + 3] = pos
    return True


def main():
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa
    env = FrankaLiberoEnv("libero_goal", 3, privileged=False, max_steps=30000,
                          control_freq=20, enable_render=False)
    env.reset(seed=1)
    sim = env.handle.env.sim

    sid = sim.model.site_name2id("wooden_cabinet_1_top_region")
    bowl = "akita_black_bowl_1_main"

    def show(tag):
        print(f"{tag}: top_region site = {np.round(sim.data.site_xpos[sid],3)}  "
              f"bowl = {np.round(sim.data.xpos[sim.model.body_name2id(bowl)],3)}  "
              f"completed = {env.task_completed()}")

    show("closed")

    # open the top drawer: top_level joint qpos -> -0.16
    a = sim.model.get_joint_qpos_addr("wooden_cabinet_1_top_level")
    sim.data.qpos[a] = -0.16
    sim.forward()
    show("opened (drawer slid)")
    sp = sim.data.site_xpos[sid].copy()
    print("open top_region center:", np.round(sp, 3))

    # teleport bowl to candidate centers (sweep z) and check completion
    for dz in (-0.06, -0.03, 0.0, 0.03):
        target = sp + np.array([0.0, 0.0, dz])
        set_body_pos(sim, bowl, target)
        sim.forward()
        # let it settle a few steps without controller
        for _ in range(30):
            sim.step()
        bp = sim.data.xpos[sim.model.body_name2id(bowl)]
        print(f"  bowl-> center+dz({dz:+.2f})  settled bowl={np.round(bp,3)}  "
              f"completed={env.task_completed()}")


if __name__ == "__main__":
    main()
