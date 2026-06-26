"""gen7 — COMPOSITIONAL task generalization: "open the top drawer and put the bowl inside".

libero_goal task3 — a multi-step task the official cap-x VDM baseline FAILS on both
libero_goal_swap and _task (objects are SAM-resolvable, so the failure is operation/judgment,
not grounding). It is the strongest generalization test for the paradigm: two subgoals judged
by ONE declarative `all_of` over geometry, with a residual that says WHICH subgoal remains.

GT (from the BDDL): (In akita_black_bowl_1 wooden_cabinet_1_top_region) — a positional predicate.
Our judge is STRICTER and matches the instruction's TWO subgoals:
    J.all_of(ctx, ("opened",   {target:"top drawer handle", travel:0.16}),
                  ("place_in", {target:"bowl", reference:"top drawer"}))

Runtime LLM simulated; execution stubbed by driving the drawer joint + bowl free-joint.
Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this> <outdir>
"""
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from track_judge.algo.track_judge import TrackJudge
from track_judge.algo import judge_dsl as J

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gen7"
os.makedirs(OUT, exist_ok=True)

env = FrankaLiberoEnv("libero_goal", 3, privileged=False, max_steps=30000,
                      control_freq=20, enable_render=True)
env.reset(seed=1)
sim = env.handle.env.sim
TOP, B = 37, slice(9, 16)                       # top_level joint, bowl free-joint
SID = sim.model.site_name2id("wooden_cabinet_1_top_region")
quat = sim.data.qpos[B][3:].copy()
bowl_start = sim.data.qpos[B][:3].copy()
cam = env.camera_params()
judge = TrackJudge(socket_path=None, viz_dir=OUT)
state = {}


def judge_state(ctx):
    return J.all_of(ctx,
                    ("opened", {"target": "top drawer handle", "travel": 0.16}),
                    ("place_in", {"target": "bowl", "reference": "top drawer"}))


def region_open():
    return sim.data.site_xpos[SID].copy()


def drive(top_q, bowl_xyz, n=16):
    env.enable_dense_rgbd_capture(True, clear=True)
    q_top0 = float(sim.data.qpos[TOP]); p_b0 = sim.data.qpos[B][:3].copy()
    for i in range(n):
        a = min(1.0, i / (n - 4))
        sim.data.qpos[TOP] = (1 - a) * q_top0 + a * top_q
        if bowl_xyz is not None:
            sim.data.qpos[B] = np.concatenate([(1 - a) * p_b0 + a * np.asarray(bowl_xyz), quat])
        sim.forward(); env._record_frame()
    rgb, depth = env.get_rgbd_frames_range(0, 10**9)
    idx = np.linspace(0, len(rgb) - 1, 12).round().astype(int)
    return np.ascontiguousarray(rgb[idx], np.uint8), np.ascontiguousarray(depth[idx], np.float32)


# 3-stage compositional loop (subgoal-by-subgoal), the "runtime LLM"'s execution attempts:
def stages():
    yield "turn1: open drawer halfway, bowl outside", -0.08, bowl_start
    yield "turn2: open drawer fully, bowl still outside", -0.16, bowl_start
    sim.data.qpos[TOP] = -0.16; sim.forward()
    r = region_open()
    yield "turn3: place bowl into the open drawer", -0.16, [r[0], r[1], r[2] - 0.02]


print("COMPOSITIONAL: open the top drawer AND put the bowl inside  (cap-x VDM FAILS this)\n")
for ti, (label, top_q, bowl_xyz) in enumerate(stages()):
    rgb, depth = drive(top_q, bowl_xyz)
    v = judge.judge_turn(rgb=rgb, depth=depth, K=cam["K"], world_to_cam=cam["world_to_cam"],
                         task_description="open the top drawer and put the bowl inside",
                         judge_fn=judge_state, state=state, viz_tag=f"turn_{ti+1}")
    gt = bool(env.task_completed())
    dec = "FINISH" if v["done"] else "REGENERATE"
    agree = "AGREE" if v["done"] == gt else "DISAGREE"
    ps = "n/a" if v["progress"] is None else f"{v['progress']:.2f}"
    print(f"[{label}]")
    print(f"   judge: done={v['done']} prog={ps} -> {dec}")
    print(f"     subgoal residual :: {v['feedback']}")
    print(f"   GT(In bowl top_region)={gt}  bowl_z={sim.data.qpos[B][2]:.3f}  | judge-vs-GT: {agree}\n")
    if v["done"]:
        break
