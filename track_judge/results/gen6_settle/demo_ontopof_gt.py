"""gen6 — fully GT-validated on_top_of loop (physics-settled placement stub).

gen5 showed LIBERO's "bowl on plate" GT predicate needs physical CONTACT, which a positional
teleport can't make, so task8's GT stayed False even when the judge was geometrically right.
Here the placement stub TELEPORTS the bowl just above the plate then lets MuJoCo settle it
(robot frozen) so the real GT predicate fires — giving a second relation (after `opened`) whose
geometric judge is checked against TRUE ground truth across a REGENERATE→FINISH loop.

Runtime LLM simulated; judge = one declarative line; GT = env.task_completed().
Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this> <outdir>
"""
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from track_judge.algo.track_judge import TrackJudge
from track_judge.algo import judge_dsl as J

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gen6"
os.makedirs(OUT, exist_ok=True)

env = FrankaLiberoEnv("libero_goal", 8, privileged=False, max_steps=30000,
                      control_freq=20, enable_render=True)
env.reset(seed=1)
sim = env.handle.env.sim
B = slice(9, 16)                       # akita_black_bowl_1_joint0
plate = sim.data.xpos[sim.model.body_name2id("plate_1_main")].copy()
cam = env.camera_params()
judge = TrackJudge(socket_path=None, viz_dir=OUT)
state = {}


def judge_state(ctx):
    return J.judge(ctx, "on_top_of", target="bowl", reference="plate")


def window():
    rgb, depth = env.get_rgbd_frames_range(0, 10**9)
    idx = np.linspace(0, len(rgb) - 1, 12).round().astype(int)
    return np.ascontiguousarray(rgb[idx], np.uint8), np.ascontiguousarray(depth[idx], np.float32)


def teleport(to_xyz, n=14, settle=0):
    env.enable_dense_rgbd_capture(True, clear=True)
    p0 = sim.data.qpos[B][:3].copy(); quat = sim.data.qpos[B][3:].copy()
    for i in range(n):
        a = min(1.0, i / (n - 4))
        sim.data.qpos[B] = np.concatenate([(1 - a) * p0 + a * np.asarray(to_xyz), quat])
        sim.forward(); env._record_frame()
    if settle:                          # let gravity SEAT the object (robot frozen) -> GT fires
        rq = sim.data.qpos[0:9].copy()
        for _ in range(settle):
            sim.step()
            sim.data.qpos[0:9] = rq; sim.data.qvel[0:9] = 0; sim.forward()
            env._record_frame()
    return window()


start = sim.data.qpos[B][:3].copy()
STAGES = [("turn1: lift, not over plate", start + [0.0, 0.0, 0.08], 0),
          ("turn2: place on plate (settle)", [plate[0], plate[1], plate[2] + 0.03], 60)]

print("on_top_of(bowl, plate) — geometric judge vs TRUE GT across a loop\n")
for ti, (label, to_xyz, settle) in enumerate(STAGES):
    rgb, depth = teleport(to_xyz, settle=settle)
    v = judge.judge_turn(rgb=rgb, depth=depth, K=cam["K"], world_to_cam=cam["world_to_cam"],
                         task_description="put the bowl on the plate", judge_fn=judge_state,
                         state=state, viz_tag=f"turn_{ti+1}")
    gt = bool(env.task_completed())
    dec = "FINISH" if v["done"] else "REGENERATE"
    agree = "AGREE" if v["done"] == gt else "DISAGREE"
    print(f"[{label}]")
    print(f"   judge: done={v['done']} prog={v['progress']:.2f} -> {dec} :: {v['feedback']}")
    print(f"   GT task_completed={gt}  bowl_z={sim.data.qpos[B][2]:.3f}  | judge-vs-GT: {agree}\n")
    if v["done"]:
        break
