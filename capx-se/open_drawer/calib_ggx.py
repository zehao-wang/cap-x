"""Calibrate GraspGenX (NVlabs) on the akita bowl and document the integration status.

What works:
  - The private GraspGenX service (see graspgenx_client.py) returns high-confidence 6-DOF
    grasps (conf 0.83-0.94, vs Contact-GraspNet's 0.15-0.29) directly in the SOLVE frame.
  - Convention (graspgenx/robot.py): approach = +Z, closing = +X, depth=0.1034 (the grasp
    transform's origin is panda_hand; the fingertips/TCP sit at pos + approach*0.1034).
    solve_ik(pos+approach*depth, quat_from_R) places the gripper exactly at the grasp pose
    (verified: panda_hand lands at the grasp's base translation).

What does NOT work yet (the finding):
  - On the akita bowl seen from a SINGLE agentview (top-only, ~2600 pts, z in [-0.005,0.04]),
    GraspGenX's grasps GRIP the rim (closed-grip 0.17-0.32) but the bowl does NOT lift
    (slips immediately; up-lift and approach-axis-retract both give dz=0). Contact-GraspNet's
    grasps, from the same partial cloud, happened to capture the rim antipodally and held the
    carry (-> the 9/20 CGN benchmark). The thin rounded rim is a marginal target; GraspGenX
    (which conditions on OBJECT SHAPE) is under-served by a top-only partial cloud.

Next step to realise GraspGenX's advantage: feed it a FULLER object point cloud
(multi-view / wrist-cam-fused), which is its intended input -- a standing cap-x sensing gap.

Run (service must be up -- see graspgenx_client.SOCKET_PATH):
    MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python capx-se/open_drawer/calib_ggx.py
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graspgenx_client import graspgenx_grasps  # noqa: E402

DEPTH = 0.1034  # panda base->tip; fingertip (sim TCP) = grasp_pos + approach*DEPTH


def main():
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import capx.integrations  # noqa
    from capx.integrations.base_api import get_api
    env = FrankaLiberoEnv("libero_goal", 3, privileged=False, max_steps=60000,
                          control_freq=20, enable_render=False)
    api = get_api("FrankaLiberoApiReducedSkillLibrary")(env); t = api.functions()
    grip = lambda: float(t["get_observation"]()["robot_joint_pos"][-1])  # noqa: E731

    env.reset(seed=1); sim = env.handle.env.sim
    a = sim.model.get_joint_qpos_addr("wooden_cabinet_1_top_level"); sim.data.qpos[a] = -0.16; sim.forward()
    cam = t["get_observation"]()["agentview"]; rgb, depth, K, ext = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
    m = sorted(t["segment_sam3_text_prompt"](rgb, "bowl"), key=lambda d: -d.get("score", 0))[0]
    pc = t["mask_to_world_points"](m["mask"], depth, K, ext); pc, _ = t["filter_noise"](pc)
    print(f"bowl cloud: {len(pc)} pts (single agentview, top-only), centroid={np.round(np.median(pc,0),3)}")

    grasps, conf = graspgenx_grasps(pc, num_grasps=200, topk=40)
    print(f"GraspGenX: {len(grasps)} grasps, conf {conf.min():.2f}-{conf.max():.2f}")
    bid = sim.model.body_name2id("akita_black_bowl_1_main")
    for i in np.argsort(-conf)[:8]:
        g = grasps[i]; appr = g[:3, 2]
        if appr[2] > -0.3:
            continue
        quat = t["rotation_matrix_to_quaternion"](g[:3, :3])
        tip = g[:3, 3] + appr * DEPTH
        env.reset(seed=1); s2 = env.handle.env.sim
        aa = s2.model.get_joint_qpos_addr("wooden_cabinet_1_top_level"); s2.data.qpos[aa] = -0.16; s2.forward()
        bp0 = s2.data.xpos[bid].copy()
        t["solve_ik"].__self__.cfg = None; t["open_gripper"]()
        pre = tip - appr * 0.13
        try:
            t["move_to_joints"](np.asarray(t["solve_ik"](pre, quat), float))
        except Exception:
            continue
        for k in range(1, 9):
            t["move_to_joints"](np.asarray(t["solve_ik"](pre + (tip - pre) * k / 8, quat), float))
        t["close_gripper"](); g0 = grip()
        for k in range(1, 6):
            t["move_to_joints"](np.asarray(t["solve_ik"](tip + np.array([0, 0, 0.18]) * k / 5, quat), float))
        dz = (s2.data.xpos[bid][2] - bp0[2]) * 1000
        print(f"  g{i} conf={conf[i]:.3f} grip={g0:.3f} lift_dz={dz:.0f}mm "
              f"-> {'LIFTED' if dz > 30 else 'grip-but-slip' if g0 > 0.05 else 'empty'}")


if __name__ == "__main__":
    main()
