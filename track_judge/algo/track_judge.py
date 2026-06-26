"""track_judge — the harness seam ``trial.py`` calls for the tracking arm.

Dense per-turn RGB-D video + camera params → TAPIP3D 3D tracks (world frame) →
build a ``ctx`` → run the agent-written ``judge_fn(ctx)`` (pure code, NO LLM) →
normalize its verdict → save a tracking-viz overlay.

No backstop (AGENT.md §7): if the TAPIP3D server is unreachable this raises a clear
error; it never silently falls back to a VLM or to ground truth.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import numpy as np

from track_judge import algo
from track_judge.algo import judge_lib, params
from track_judge.tapip3d_client import Tapip3DClient, Tapip3DError


class TrackJudgeError(RuntimeError):
    pass


def _subsample_indices(n_frames: int, window: int):
    """Evenly pick ``window`` frame indices from [0, n_frames). Keeps last frame."""
    if n_frames <= window:
        return list(range(n_frames))
    idx = np.linspace(0, n_frames - 1, window).round().astype(int)
    return sorted(set(int(i) for i in idx))


class _Ctx:
    """What ``judge_fn(ctx)`` sees. See judge_prompt.JUDGE_PROMPT_FRAGMENT for the
    agent-facing description (the two must stay in sync)."""

    def __init__(self, *, coords, visibs, rgb, depth, K, query_xy, task, state):
        self.coords = coords        # [T, N, 3] world frame, metres
        self.visibs = visibs        # [T, N] bool/float visibility
        self.rgb = rgb              # [T, H, W, 3] uint8 (sub-sampled turn video)
        self.depth = depth          # [T, H, W] float32 metres
        self.K = K                  # [3, 3] camera intrinsics
        self.lib = judge_lib        # convenience geometric primitives
        self.task = task            # task_description (str)
        self.state = state          # persistent dict across turns
        self._query_xy = query_xy   # [N, 2] pixel (x, y) seed of each track on frame 0

    def mask_points(self, mask) -> np.ndarray:
        """Map a 2D bool mask [H, W] → indices of tracked points seeded inside it.

        Returns an int index array selecting columns of ``coords``/``visibs``.
        """
        mask = np.asarray(mask, dtype=bool)
        if self._query_xy is None:
            return np.arange(self.coords.shape[1])
        H, W = mask.shape[:2]
        xs = np.clip(self._query_xy[:, 0].round().astype(int), 0, W - 1)
        ys = np.clip(self._query_xy[:, 1].round().astype(int), 0, H - 1)
        keep = mask[ys, xs]
        return np.nonzero(keep)[0]

    # ── named-object resolution (local SAM model, NOT an LLM) ────────────────────
    _segment_fn = None  # lazy, shared per ctx
    # Min SAM confidence to TRUST a named-object mask. Validated on a real libero
    # scene: distinct objects (plate/bowl/bottle/handle/cabinet) score 0.58-0.93,
    # while failed resolutions (robot gripper 0.01, cheese 0.076) sit far below —
    # so a ~0.3 gate cleanly rejects garbage masks instead of tracking them.
    MIN_SEG_SCORE = 0.30

    def segment(self, text, *, min_score=None):
        """Segment a NAMED object on frame 0 via the local SAM3 service → bool mask
        [H, W], or None if SAM is unavailable / finds nothing / is below the confidence
        gate (a low-score mask is worse than no mask — it tracks the wrong region). No
        LLM involved."""
        if self._segment_fn is None:
            try:
                from capx.integrations.vision.sam3 import init_sam3
                self._segment_fn = init_sam3()
            except Exception as e:  # SAM service down / import error → caller falls back
                print(f"[track-judge] SAM3 unavailable for segment({text!r}): {e}")
                self._segment_fn = False
        if not self._segment_fn:
            return None
        try:
            results = self._segment_fn(self.rgb[0], text)
        except Exception as e:
            print(f"[track-judge] SAM3 segment({text!r}) failed: {e}")
            return None
        if not results:
            return None
        best = max(results, key=lambda r: r.get("score", 0.0))
        gate = self.MIN_SEG_SCORE if min_score is None else min_score
        if float(best.get("score", 0.0)) < gate:
            print(f"[track-judge] segment({text!r}) low confidence "
                  f"{best.get('score', 0.0):.3f} < {gate}; not trusting this mask")
            return None
        return np.asarray(best["mask"], dtype=bool)

    # set by TrackJudge.judge_turn so points_of can DENSELY re-seed tracks inside a small
    # object's mask (the global grid is too coarse for e.g. a drawer handle ~0.2% of frame)
    _track_client = None
    _world_to_cam = None
    _track_iters = 6
    MIN_GRID_PTS = 6  # below this many grid hits, do a dedicated mask-seeded track

    def points_of(self, text):
        """Named object → its tracked sub-trajectory ``[T, Nt, 3]`` (world frame).

        Segments the object (frame 0, local SAM). If enough whole-frame grid tracks fall
        inside the mask, use those (cheap); otherwise DENSELY seed query points inside the
        mask and run a dedicated TAPIP3D track (small objects like a drawer handle get 0-1
        grid hits, so the grid centroid barely moves — the dedicated track follows them).
        Returns ``None`` when nothing resolves (DSL then falls back to the whole grid)."""
        mask = self.segment(text)
        if mask is None:
            return None
        idx = self.mask_points(mask)
        if idx.size >= self.MIN_GRID_PTS:
            return self.coords[:, idx, :]
        dense = self.track_mask(mask)            # too few grid hits → dedicated dense track
        if dense is not None and dense.shape[1] > 0:
            return dense
        return self.coords[:, idx, :] if idx.size else None

    def track_mask(self, mask, max_pts=150):
        """Track points DENSELY seeded inside ``mask`` (frame 0) via a dedicated TAPIP3D
        call → ``[T, Nt, 3]`` world frame, or None if the tracker context is absent.

        Back-projects masked pixels to world (depth + K + extrinsics, the convention in
        camera_params: X_cam = depth · K⁻¹[u,v,1], world = cam_to_world · X_cam) and passes
        them as ``query_point`` [N,4] = [t=0, Xw, Yw, Zw] (TAPIP3D's grid-query format)."""
        if self._track_client is None or self._world_to_cam is None:
            return None
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        if xs.size == 0:
            return None
        if xs.size > max_pts:                    # cap cost: even spread of mask pixels
            sel = np.linspace(0, xs.size - 1, max_pts).round().astype(int)
            xs, ys = xs[sel], ys[sel]
        z = self.depth[0][ys, xs].astype(float)
        good = z > 1e-3
        xs, ys, z = xs[good], ys[good], z[good]
        if xs.size == 0:
            return None
        uv1 = np.stack([xs, ys, np.ones_like(xs)], 0).astype(float)   # [3,N]
        x_cam = (np.linalg.inv(self.K) @ uv1) * z[None, :]            # [3,N]
        cam_to_world = np.linalg.inv(self._world_to_cam)
        x_world = cam_to_world[:3, :3] @ x_cam + cam_to_world[:3, 3:4]  # [3,N]
        q = np.concatenate([np.zeros((1, x_world.shape[1])), x_world], 0).T.astype(np.float32)
        try:
            out = self._track_client.track(
                video=self.rgb, depths=self.depth, intrinsics=self.K,
                extrinsics=self._world_to_cam, query_point=q, num_iters=self._track_iters)
        except Exception as e:
            print(f"[track-judge] dense mask-seeded track failed: {e}")
            return None
        c = out.get("coords")
        return None if c is None else np.asarray(c, dtype=float)


class TrackJudge:
    def __init__(self, socket_path="/tmp/demo_bridge/sockets/tapip3d.sock", viz_dir=None,
                 window=12, resolution_factor=1.0, num_iters=6):
        # Coalesce None -> default (callers may pass socket_path=None to mean "use default").
        self.socket_path = socket_path or "/tmp/demo_bridge/sockets/tapip3d.sock"
        self.viz_dir = viz_dir
        self.window = int(window)
        self.resolution_factor = float(resolution_factor)
        self.num_iters = int(num_iters)
        self._client: Optional[Tapip3DClient] = None

    def _conn(self) -> Tapip3DClient:
        if self._client is None:
            self._client = Tapip3DClient(self.socket_path)
        try:
            if not self._client.ping():
                raise TrackJudgeError(
                    f"TAPIP3D server at {self.socket_path} did not pong (start it first; "
                    "no VLM/GT backstop)")
        except Tapip3DError as e:
            raise TrackJudgeError(
                f"TAPIP3D server unreachable at {self.socket_path}: {e}. Start the tapip3d "
                "service before an arm-B run; the tracking judge has NO backstop.") from e
        return self._client

    def judge_turn(self, *, rgb, depth, K, world_to_cam, task_description, judge_fn,
                   state: Dict[str, Any], viz_tag: str) -> Dict[str, Any]:
        """Run one turn's geometric judgment.

        rgb [T,H,W,3] u8, depth [T,H,W] f32 metres, K [3,3],
        world_to_cam [4,4] (TAPIP3D ``extrinsics`` convention). ``judge_fn`` is the
        agent-written ``judge_state(ctx)``. Returns a normalized verdict dict.
        """
        rgb = np.asarray(rgb)
        depth = np.asarray(depth, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        world_to_cam = np.asarray(world_to_cam, dtype=np.float32)
        T = rgb.shape[0]
        if T == 0:
            raise TrackJudgeError("judge_turn got an empty turn video")

        idx = _subsample_indices(T, self.window)
        sub_rgb = np.ascontiguousarray(rgb[idx], dtype=np.uint8)
        sub_depth = np.ascontiguousarray(depth[idx], dtype=np.float32)

        client = self._conn()
        grid = int(params.DEFAULTS.get("query_grid", 24))
        out = client.track(
            video=sub_rgb, depths=sub_depth, intrinsics=K, extrinsics=world_to_cam,
            num_iters=self.num_iters, resolution_factor=self.resolution_factor,
            query_grid=grid,
        )
        coords = np.asarray(out["coords"], dtype=float) if out["coords"] is not None else None
        visibs = np.asarray(out["visibs"]) if out["visibs"] is not None else None
        if coords is None or coords.ndim != 3:
            raise TrackJudgeError(f"TAPIP3D returned no usable coords (got {type(out['coords'])})")

        query_xy = self._grid_query_xy(sub_rgb.shape[1], sub_rgb.shape[2], grid, coords.shape[1])
        ctx = _Ctx(coords=coords, visibs=visibs, rgb=sub_rgb, depth=sub_depth, K=K,
                   query_xy=query_xy, task=task_description, state=state)
        # let ctx.points_of densely re-seed tracks inside a small object's mask
        ctx._track_client = client
        ctx._world_to_cam = world_to_cam
        ctx._track_iters = self.num_iters

        verdict = judge_fn(ctx)
        norm = _normalize_verdict(verdict)

        self._save_viz(ctx, norm, viz_tag)
        return norm

    @staticmethod
    def _grid_query_xy(h, w, grid, n_points):
        """Reconstruct the pixel (x, y) of the default NxN grid seed, frame-0.

        Mirrors a centred grid; truncated/padded to ``n_points`` to stay aligned with
        whatever the server returned.
        """
        gx = np.linspace(0, w - 1, grid)
        gy = np.linspace(0, h - 1, grid)
        xx, yy = np.meshgrid(gx, gy)
        xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        if xy.shape[0] >= n_points:
            return xy[:n_points]
        pad = np.repeat(xy[-1:], n_points - xy.shape[0], axis=0)
        return np.concatenate([xy, pad], axis=0)

    def _save_viz(self, ctx: _Ctx, verdict: Dict[str, Any], viz_tag: str) -> None:
        """Overlay tracked points (projected to last RGB frame) + reported regions."""
        if self.viz_dir is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.viz_dir, exist_ok=True)
        last_rgb = ctx.rgb[-1]
        H, W = last_rgb.shape[:2]
        K = ctx.K
        pts = ctx.coords[-1]  # [N, 3] world; project via K (cam frame approx for viz)

        fig, ax = plt.subplots(figsize=(6, 6 * H / max(W, 1)))
        ax.imshow(last_rgb)
        # project world points: viz only — project with K assuming coords already cam-ish;
        # fall back to raw scatter of x/y if projection degenerates.
        uv = self._project(pts, K)
        if uv is not None:
            vis = np.ones(pts.shape[0], bool)
            if ctx.visibs is not None and ctx.visibs.shape[0] == ctx.coords.shape[0]:
                vis = np.asarray(ctx.visibs[-1]).astype(bool)
            ax.scatter(uv[vis, 0], uv[vis, 1], s=6, c="lime", alpha=0.7)
            ax.scatter(uv[~vis, 0], uv[~vis, 1], s=6, c="red", alpha=0.4)
        self._draw_regions(ax, verdict, K)
        ax.set_title(f"{viz_tag}  done={verdict['done']} abort={verdict['abort']} "
                     f"prog={verdict['progress']}", fontsize=8)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(self.viz_dir, f"{viz_tag}.png"), dpi=110)
        plt.close(fig)

    @staticmethod
    def _project(pts, K):
        """Project [N,3] camera-frame points to pixels [N,2]; None if degenerate."""
        pts = np.asarray(pts, dtype=float)
        z = pts[:, 2]
        z = np.where(np.abs(z) < 1e-6, np.nan, z)
        uv = (K @ (pts / z[:, None]).T).T[:, :2]
        if not np.isfinite(uv).any():
            return None
        return uv

    def _draw_regions(self, ax, verdict, K):
        """Draw any regions the verdict reports (under verdict['regions'])."""
        import matplotlib.patches as mpatches
        regions = verdict.get("regions") or []
        for r in regions:
            if r.get("type") == "aabb":
                lo, hi = np.asarray(r["lo"], float), np.asarray(r["hi"], float)
                corners = np.array([[lo[0], lo[1], (lo[2] + hi[2]) / 2],
                                    [hi[0], hi[1], (lo[2] + hi[2]) / 2]])
                uv = self._project(corners, K)
                if uv is not None:
                    x0, y0 = uv[0]; x1, y1 = uv[1]
                    ax.add_patch(mpatches.Rectangle(
                        (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                        fill=False, edgecolor="cyan", lw=1.5))


def _normalize_verdict(v: Any) -> Dict[str, Any]:
    """Normalize a judge return → {feedback:str, done:bool, abort:bool, progress:float|None}.

    Tolerates a bare bool (done) or a bare string (feedback).
    """
    if isinstance(v, bool):
        return {"feedback": "", "done": v, "abort": False, "progress": None, "regions": []}
    if isinstance(v, str):
        return {"feedback": v, "done": False, "abort": False, "progress": None, "regions": []}
    if not isinstance(v, dict):
        raise TrackJudgeError(f"judge_fn returned unsupported type {type(v)}")
    progress = v.get("progress")
    return {
        "feedback": str(v.get("feedback", "")),
        "done": bool(v.get("done", False)),
        "abort": bool(v.get("abort", False)),
        "progress": None if progress is None else float(progress),
        "regions": list(v.get("regions", []) or []),
    }


# silence unused-import linters: ``algo`` re-export anchors the package
__all__ = ["TrackJudge", "TrackJudgeError", "algo"]
