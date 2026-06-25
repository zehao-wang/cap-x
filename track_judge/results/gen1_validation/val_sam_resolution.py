"""Validation A: does a PLAIN-WORD object name resolve to a sensible mask via local
SAM3 on a REAL libero scene? This is the crux of the weak-LLM-friendly DSL: the agent
names objects in plain words and the harness must find them.

Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this>
"""
import os, sys
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.getcwd(), "capx-se"))
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from capx.integrations.base_api import get_api

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_sam"
os.makedirs(OUT, exist_ok=True)
SUITE, TASK = "libero_goal", 0

env = FrankaLiberoEnv(SUITE, TASK, privileged=False, max_steps=3000,
                      control_freq=20, enable_render=True)
env.reset(seed=1)
api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
t = api.functions()

obs = t["get_observation"]()
cam = obs["agentview"]
rgb = np.asarray(cam["images"]["rgb"])
H, W = rgb.shape[:2]
imageio.imwrite(os.path.join(OUT, "scene_agentview.png"), rgb)
print(f"scene: {SUITE}/task{TASK} seed1, agentview {W}x{H}")
print(f"task instruction: {env.handle.env.language_instruction if hasattr(env.handle.env,'language_instruction') else '(n/a)'}")

# plain-word names a person would use for THIS scene
PROMPTS = ["drawer handle", "wooden cabinet", "plate", "bowl",
           "wine bottle", "cheese", "robot gripper", "cabinet drawer"]

print(f"\n{'prompt':<16} {'n':>2} {'best_score':>10} {'area_px':>8} {'area%':>6}  centroid(px)")
print("-" * 70)
for p in PROMPTS:
    try:
        res = t["segment_sam3_text_prompt"](rgb, p)
    except Exception as e:
        print(f"{p:<16}  ERR {str(e)[:40]}")
        continue
    if not res:
        print(f"{p:<16} {0:>2} {'-':>10} {'-':>8} {'-':>6}  (no mask)")
        continue
    res = sorted(res, key=lambda d: -d.get("score", 0.0))
    best = res[0]
    m = np.asarray(best["mask"], dtype=bool)
    area = int(m.sum())
    ys, xs = np.nonzero(m)
    cen = (int(xs.mean()), int(ys.mean())) if area else (-1, -1)
    print(f"{p:<16} {len(res):>2} {best.get('score',0):>10.3f} {area:>8} "
          f"{100*area/(H*W):>5.1f}%  {cen}")
    # overlay
    ov = rgb.copy()
    if m.shape[:2] == ov.shape[:2]:
        ov[m] = (0.5 * ov[m] + 0.5 * np.array([30, 144, 255])).astype(np.uint8)
    imageio.imwrite(os.path.join(OUT, f"mask_{p.replace(' ','_')}.png"), ov)

print(f"\noverlays + scene written to {OUT}")
