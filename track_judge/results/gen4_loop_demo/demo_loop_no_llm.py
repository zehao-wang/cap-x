"""Arm-B loop with the runtime LLM SIMULATED (no :8110 endpoint).

A human-strong model (or the experimenter) stands in for the cap-agent0 runtime LLM: it
reads the task and, per the judge_prompt, writes the ONE-LINE declarative judge. Execution
is stubbed by driving the drawer joint directly (standing in for the LLM's motion plan), so
we can exercise the REAL arm-B decision path — TrackJudge.judge_turn (the production seam) →
geometric verdict → FINISH/REGENERATE — end to end, without the blocked endpoint.

Demonstrates "geometry drives the next code revision":
  turn 1: execution opens the drawer only a little -> judge: NOT done + metric residual
          ("moved Xcm of ~16cm") -> REGENERATE
  turn 2: revised execution (read the residual) pulls fully open -> judge: DONE -> FINISH
And checks the judge verdict AGREES with libero ground-truth at each turn.

Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this> <outdir>
"""
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from track_judge.algo.track_judge import TrackJudge
from track_judge.algo import judge_dsl as J

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gen4_demo"
os.makedirs(OUT, exist_ok=True)
SUITE, TASK, SEED = "libero_goal", 0, 1

env = FrankaLiberoEnv(SUITE, TASK, privileged=False, max_steps=30000,
                      control_freq=20, enable_render=True)
env.reset(seed=SEED)
sim = env.handle.env.sim
QADR = sim.model.get_joint_qpos_addr("wooden_cabinet_1_middle_level")
QADR = QADR[0] if isinstance(QADR, tuple) else QADR

judge = TrackJudge(socket_path=None, viz_dir=OUT)        # production harness seam
cam = env.camera_params()
state = {}                                               # persists ACROSS turns (ctx.state)

# === the runtime LLM's judge (per judge_prompt: ONE declarative line) ===
def judge_state(ctx):
    from track_judge.algo import judge_dsl as J
    return J.judge(ctx, "opened", target="drawer handle", travel=0.16)

def execute_open(to_qpos, n=16):
    """Stand-in for the LLM's execution code: drive the drawer to `to_qpos`, recording a
    dense RGB-D window (what the tracker consumes)."""
    env.enable_dense_rgbd_capture(True, clear=True)
    q0 = float(sim.data.qpos[QADR])
    for i in range(n):
        a = min(1.0, i / (n - 4))
        sim.data.qpos[QADR] = (1 - a) * q0 + a * to_qpos
        sim.forward()
        env._record_frame()
    rgb, depth = env.get_rgbd_frames_range(0, 10**9)
    idx = np.linspace(0, len(rgb) - 1, 12).round().astype(int)
    return np.ascontiguousarray(rgb[idx], np.uint8), np.ascontiguousarray(depth[idx], np.float32)

PLAN = [("turn1: short pull", -0.05), ("turn2: full pull (revised)", -0.16)]
print(f"task: '{env.handle.env.language_instruction}'  (success = drawer qpos < -0.14)\n")
for ti, (label, to_q) in enumerate(PLAN):
    rgb, depth = execute_open(to_q)
    verdict = judge.judge_turn(
        rgb=rgb, depth=depth, K=cam["K"], world_to_cam=cam["world_to_cam"],
        task_description="open the middle drawer", judge_fn=judge_state,
        state=state, viz_tag=f"turn_{ti+1}")
    gt_qpos = float(sim.data.qpos[QADR])
    gt_done = bool(env.task_completed())
    decision = "FINISH" if verdict["done"] else ("REGENERATE" if not verdict["done"] else "?")
    agree = "AGREE" if verdict["done"] == gt_done else "DISAGREE"
    print(f"[{label}]")
    print(f"   judge: done={verdict['done']} progress={verdict['progress']:.2f} :: {verdict['feedback']}")
    print(f"   -> agent decision: {decision}")
    print(f"   GT: drawer_qpos={gt_qpos:+.3f} task_completed={gt_done}  | judge-vs-GT: {agree}\n")

print(f"viz (per-turn tracked points) written to {OUT}/turn_*.png")
