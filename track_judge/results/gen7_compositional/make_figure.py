"""Make a one-figure explainer for the bowl-in-drawer judgment: two panels (bowl OUTSIDE
the open drawer -> REGENERATE, bowl INSIDE -> FINISH), each overlaying the TRACKED 3D points
(drawer handle = red, bowl = green) and the drawer containment BOX (cyan) onto the camera view.

Run: MUJOCO_GL=egl HF_HUB_OFFLINE=1 .venv-libero/bin/python <this> <out.png>
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from capx.envs.simulators.libero import FrankaLiberoEnv
import capx.integrations  # noqa
from track_judge.algo.track_judge import TrackJudge, _Ctx, _subsample_indices
from track_judge.algo import judge_dsl as J
from track_judge.tapip3d_client import Tapip3DClient

OUT = sys.argv[1] if len(sys.argv) > 1 else "judgment_figure.png"

env = FrankaLiberoEnv("libero_goal", 3, privileged=False, max_steps=30000,
                      control_freq=20, enable_render=True)
env.reset(seed=1); sim = env.handle.env.sim
TOP, B = 37, slice(9, 16)
SID = sim.model.site_name2id("wooden_cabinet_1_top_region")
quat = sim.data.qpos[B][3:].copy(); bowl_start = sim.data.qpos[B][:3].copy()
cam = env.camera_params(); K, w2c = cam["K"], cam["world_to_cam"]
cl = Tapip3DClient("/tmp/demo_bridge/sockets/tapip3d.sock", connect_timeout=15)


def project(P):
    """world [N,3] -> pixel [N,2] (OpenCV cam frame from camera_params)."""
    P = np.asarray(P, float)
    cp = (w2c @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    z = np.where(np.abs(cp[:, 2]) < 1e-6, np.nan, cp[:, 2])
    uv = (K @ (cp / z[:, None]).T).T[:, :2]
    return uv


def drive(top_q, bowl_xyz, n=16):
    env.enable_dense_rgbd_capture(True, clear=True)
    qt0 = float(sim.data.qpos[TOP]); pb0 = sim.data.qpos[B][:3].copy()
    for i in range(n):
        a = min(1.0, i / (n - 4))
        sim.data.qpos[TOP] = (1 - a) * qt0 + a * top_q
        sim.data.qpos[B] = np.concatenate([(1 - a) * pb0 + a * np.asarray(bowl_xyz), quat])
        sim.forward(); env._record_frame()
    rgb, depth = env.get_rgbd_frames_range(0, 10**9)
    idx = _subsample_indices(len(rgb), 12)
    return np.ascontiguousarray(rgb[idx], np.uint8), np.ascontiguousarray(depth[idx], np.float32)


def make_ctx(rgb, depth):
    out = cl.track(video=rgb, depths=depth, intrinsics=K, extrinsics=w2c, query_grid=24, num_iters=6)
    coords = np.asarray(out["coords"], float)
    qxy = TrackJudge._grid_query_xy(rgb.shape[1], rgb.shape[2], 24, coords.shape[1])
    ctx = _Ctx(coords=coords, visibs=out.get("visibs"), rgb=rgb, depth=depth, K=K,
               query_xy=qxy, task="open the top drawer and put the bowl inside", state=state)
    ctx._track_client = cl; ctx._world_to_cam = w2c; ctx._track_iters = 6
    return ctx


def draw_box(ax, region, color="cyan"):
    lo, hi = np.asarray(region["lo"]), np.asarray(region["hi"])
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    uv = project(c)
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in E:
        ax.plot([uv[i,0], uv[j,0]], [uv[i,1], uv[j,1]], color=color, lw=1.4, alpha=0.9)


JUDGE = lambda ctx: J.all_of(ctx,
        ("opened", {"target": "top drawer handle", "travel": 0.16}),
        ("place_in", {"target": "bowl", "reference": "top drawer"}))


def region_from(ctx):
    from track_judge.algo import judge_dsl as _J
    ref = ctx.points_of("top drawer")            # cached after the priming turn
    return _J._footprint_region(ref) if ref is not None else None


state = {}
# Follow gen7's flow so the cross-turn cache is seeded by a resolvable (half-open) turn.
# prime: drawer half-open, bowl outside -> resolves & caches handle + drawer
_ = JUDGE(make_ctx(*drive(-0.08, bowl_start)))

# Panel A: drawer fully open, bowl STILL outside -> REGENERATE
ctxA = make_ctx(*drive(-0.16, bowl_start))
vA = JUDGE(ctxA); bA = ctxA.points_of("bowl"); hA = ctxA.points_of("top drawer handle")
regA = region_from(ctxA)

# Panel B: bowl placed INSIDE the open drawer -> FINISH
sim.data.qpos[TOP] = -0.16; sim.forward(); r = sim.data.site_xpos[SID].copy()
ctxB = make_ctx(*drive(-0.16, [r[0], r[1], r[2] - 0.02]))
vB = JUDGE(ctxB); bB = ctxB.points_of("bowl"); hB = ctxB.points_of("top drawer handle")
regB = region_from(ctxB)


fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
panels = [("(1) bowl OUTSIDE the open drawer  ->  REGENERATE", ctxA, hA, bA, vA),
          ("(2) bowl INSIDE the drawer  ->  FINISH", ctxB, hB, bB, vB)]
for ax, (title, ctx, h, b, v) in zip(axes, panels):
    ax.imshow(ctx.rgb[-1]); ax.set_xlim(0, ctx.rgb.shape[2]); ax.set_ylim(ctx.rgb.shape[1], 0); ax.axis("off")
    reg = region_from(ctx)
    if reg is not None:
        draw_box(ax, reg, "cyan")
    if h is not None:
        uv = project(h[-1]); ax.scatter(uv[:,0], uv[:,1], s=14, c="red", edgecolors="k", lw=0.3, label="drawer handle (tracked)")
    if b is not None:
        uv = project(b[-1]); ax.scatter(uv[:,0], uv[:,1], s=14, c="lime", edgecolors="k", lw=0.3, label="bowl (tracked)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0.02, 0.02, v["feedback"], transform=ax.transAxes, fontsize=8.5, va="bottom",
            color="white", bbox=dict(boxstyle="round", fc="black", alpha=0.6))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
axes[0].text(0.02, 0.98, "cyan box = drawer containment region\n(is the bowl inside it?)",
             transform=axes[0].transAxes, fontsize=8.5, va="top",
             color="white", bbox=dict(boxstyle="round", fc="navy", alpha=0.6))
fig.suptitle("Geometric judgment of \"open the top drawer and put the bowl inside\"  (no LLM at judgment)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT, "| panelA done=", vA["done"], "| panelB done=", vB["done"])
