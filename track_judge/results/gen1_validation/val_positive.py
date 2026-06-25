"""Validation C (POSITIVE done=True on REAL tracks): guard against a degenerate judge
that never says done. Teleport the bowl onto the plate over a rendered window, track it,
and check on_top_of(bowl, plate) flips done=False -> done=True.

Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this>
"""
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from track_judge.algo.track_judge import TrackJudge, _Ctx, _subsample_indices
from track_judge.algo import judge_dsl as J
from track_judge.tapip3d_client import Tapip3DClient

SUITE, TASK, SEED, WINDOW = "libero_goal", 0, 1, 12
env = FrankaLiberoEnv(SUITE, TASK, privileged=False, max_steps=30000,
                      control_freq=20, enable_render=True)
env.reset(seed=SEED)
env.enable_dense_rgbd_capture(True, clear=True)
sim = env.handle.env.sim

BOWL = slice(9, 16)               # akita_black_bowl_1_joint0 qpos addr
plate = sim.data.xpos[sim.model.body_name2id("plate_1_main")].copy()
q0 = sim.data.qpos[BOWL].copy()
start_xyz = q0[:3].copy()
goal_xyz = np.array([plate[0], plate[1], plate[2] + 0.035])   # resting on the plate
quat = q0[3:].copy()
print(f"bowl {np.round(start_xyz,3)} -> plate-top {np.round(goal_xyz,3)}")

NSTEP = 18
for i in range(NSTEP):
    a = min(1.0, i / (NSTEP - 5))            # arrive a few frames early, then HOLD (persistence)
    xyz = (1 - a) * start_xyz + a * goal_xyz
    sim.data.qpos[BOWL] = np.concatenate([xyz, quat])
    sim.forward()
    env._record_frame()

rgb_all, depth_all = env.get_rgbd_frames_range(0, 10**9)
n = len(rgb_all)
idx = _subsample_indices(n, WINDOW)
rgb = np.ascontiguousarray(rgb_all[idx], np.uint8)
depth = np.ascontiguousarray(depth_all[idx], np.float32)
cam = env.camera_params()
cl = Tapip3DClient("/tmp/demo_bridge/sockets/tapip3d.sock", connect_timeout=15)
out = cl.track(video=rgb, depths=depth, intrinsics=cam["K"], extrinsics=cam["world_to_cam"],
               query_grid=24, num_iters=6)
coords = np.asarray(out["coords"], float)
visibs = np.asarray(out["visibs"]) if out["visibs"] is not None else None
qxy = TrackJudge._grid_query_xy(rgb.shape[1], rgb.shape[2], 24, coords.shape[1])
ctx = _Ctx(coords=coords, visibs=visibs, rgb=rgb, depth=depth, K=cam["K"],
           query_xy=qxy, task="put the bowl on the plate", state={})

# where did the tracked bowl end up?
bpts = ctx.points_of("bowl")
if bpts is not None:
    c = np.nanmean(bpts[-1], axis=0)
    print(f"tracked bowl final centroid xy={np.round(c[:2],3).tolist()} z={c[2]:+.3f} "
          f"| GT goal xy={np.round(goal_xyz[:2],3).tolist()} z={goal_xyz[2]:+.3f}")

v = J.judge(ctx, "on_top_of", target="bowl", reference="plate")
print(f"\non_top_of(bowl, plate): done={v['done']} progress={v['progress']} :: {v['feedback']}")
print("RESULT:", "PASS (positive done=True on real tracks)" if v["done"] else "FAIL (did not flip to done)")
