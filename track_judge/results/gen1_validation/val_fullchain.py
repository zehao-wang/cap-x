"""Validation B (full arm-B chain on REAL data, NO LLM):
  real libero window -> dense RGB-D (production capture) -> TAPIP3D world tracks
  -> ctx.points_of("plate") [local SAM] -> compare WORLD centroid to GT plate pose
  -> run judge_dsl relations.

Decisive check: does a plain-word-named object's TRACKED world centroid land on its
true (privileged) position? If yes, the whole SAM->track->world->DSL pipeline is sound.

Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this> <outdir>
"""
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())  # repo root, for `track_judge`
sys.path.insert(0, os.path.join(os.getcwd(), "capx-se"))
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from capx.integrations.base_api import get_api
from track_judge.algo.track_judge import TrackJudge, _Ctx, _subsample_indices
from track_judge.algo import judge_dsl as J
from track_judge.tapip3d_client import Tapip3DClient

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_full"
os.makedirs(OUT, exist_ok=True)
SUITE, TASK, SEED = "libero_goal", 0, 1
WINDOW = 12

env = FrankaLiberoEnv(SUITE, TASK, privileged=False, max_steps=30000,
                      control_freq=20, enable_render=True)
env.reset(seed=SEED)
env.enable_dense_rgbd_capture(True, clear=True)
sim = env.handle.env.sim
api = get_api("FrankaLiberoApiReducedSkillLibrary")(env)
t = api.functions()

# small joint jog to generate a temporal window of frames (records densely)
j0 = np.asarray(t["get_observation"]()["robot_joint_pos"][:7], float)
for dq in (+0.30, -0.30, +0.15):
    tgt = j0.copy(); tgt[0] += dq
    t["move_to_joints"](tgt)
rgb_all, depth_all = env.get_rgbd_frames_range(0, 10**9)
n = len(rgb_all)
idx = _subsample_indices(n, WINDOW)
rgb = np.ascontiguousarray(rgb_all[idx], dtype=np.uint8)
depth = np.ascontiguousarray(depth_all[idx], dtype=np.float32)
cam = env.camera_params()
K, w2c = cam["K"], cam["world_to_cam"]
print(f"window: {n} dense frames -> {len(idx)} subsampled; rgb {rgb.shape} depth {depth.shape}")
print(f"depth range (m): {np.nanmin(depth):.3f}..{np.nanmax(depth):.3f}")

# GT object positions (PRIVILEGED — validation measuring stick only, never in solve)
def gt_xpos(substr):
    names = [b for b in sim.model.body_names if substr in b and "main" in b]
    if not names:
        names = [b for b in sim.model.body_names if substr in b]
    return None if not names else sim.data.xpos[sim.model.body_name2id(names[0])].copy()
gt = {k: gt_xpos(k) for k in ("plate", "bowl", "wine", "cabinet")}
print("GT positions:", {k: None if v is None else np.round(v, 3).tolist() for k, v in gt.items()})

# TAPIP3D track the whole-frame grid in WORLD frame
cl = Tapip3DClient("/tmp/demo_bridge/sockets/tapip3d.sock", connect_timeout=15)
out = cl.track(video=rgb, depths=depth, intrinsics=K, extrinsics=w2c,
               query_grid=24, num_iters=6)
coords = np.asarray(out["coords"], float)
visibs = np.asarray(out["visibs"]) if out["visibs"] is not None else None
print(f"TAPIP3D coords {coords.shape}  world xyz range: "
      f"x[{coords[...,0].min():.2f},{coords[...,0].max():.2f}] "
      f"y[{coords[...,1].min():.2f},{coords[...,1].max():.2f}] "
      f"z[{coords[...,2].min():.2f},{coords[...,2].max():.2f}]")

# build the same ctx the harness builds
qxy = TrackJudge._grid_query_xy(rgb.shape[1], rgb.shape[2], 24, coords.shape[1])
ctx = _Ctx(coords=coords, visibs=visibs, rgb=rgb, depth=depth, K=K,
           query_xy=qxy, task="open the middle drawer", state={})

print("\n=== DECISIVE: named-object tracked centroid vs GT (world frame) ===")
for name, key in [("plate", "plate"), ("bowl", "bowl"), ("wine bottle", "wine")]:
    pts = ctx.points_of(name)
    if pts is None:
        print(f"  {name:<12} -> NOT RESOLVED (SAM low-confidence / fallback)")
        continue
    c = np.nanmean(pts[-1], axis=0)            # last-frame world centroid
    g = gt[key]
    err = None if g is None else float(np.linalg.norm(c[:2] - g[:2]))  # xy error (m)
    print(f"  {name:<12} -> {pts.shape[1]:>3} tracks, centroid xy={np.round(c[:2],3).tolist()} "
          f"z={c[2]:+.3f}  | GT xy={None if g is None else np.round(g[:2],3).tolist()} "
          f"| xy_err={'n/a' if err is None else f'{err*100:.1f}cm'}")

print("\n=== DSL relations (sanity; nothing was actually manipulated) ===")
print("  opened(drawer handle):", J.judge(ctx, "opened", target="drawer handle", travel=0.15)["feedback"])
print("  placed_in(bowl in plate):", J.judge(ctx, "place_in", target="bowl", reference="plate")["feedback"])
print("  next_to(bowl, plate):", J.judge(ctx, "next_to", target="bowl", reference="plate")["feedback"])

np.savez(os.path.join(OUT, "fullchain.npz"), coords=coords, K=K, w2c=w2c)
print(f"\nsaved tracks to {OUT}/fullchain.npz")
