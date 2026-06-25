"""judge_dsl — a DECLARATIVE goal-relation layer so a *weak* LLM can write the judge.

Why this exists
---------------
Writing ``judge_state(ctx)`` from raw ``coords[T,N,3]`` + numpy is hard for a small
model: it has to choose masks, reason in world coordinates, build regions, and get the
geometry right. That is exactly the kind of reasoning AGENT.md wants pushed OUT of the
runtime LLM and INTO deterministic tooling.

So instead the agent names a **goal relation** over **named objects** and we do the rest
(local SAM segmentation → TAPIP3D tracks → metric geometry). The agent's whole judge
collapses to one line:

    def judge_state(ctx):
        from track_judge.algo import judge_dsl as J
        return J.judge(ctx, "place_in", target="bowl", reference="plate")

Every relation returns the normalized verdict the harness expects
(``{done, abort, progress, feedback, regions}``) and — crucially for driving the next
code revision — a **metric, diagnostic** ``feedback`` (a residual the next turn can act
on), not a vague opinion. The relation functions below are PURE (operate on point
trajectories), so they are unit-testable with no services; ``judge(ctx, ...)`` is the
thin convenience that resolves named objects to trajectories via ``ctx``.

Conventions: ``traj`` is ``[T, N, 3]`` world-frame metres; a region is
``{"type": "aabb", "lo": [x,y,z], "hi": [x,y,z]}`` (drawn in the tracking-viz).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from track_judge.algo import judge_lib as _lib


# ── verdict + small geometry helpers ────────────────────────────────────────────────
def _verdict(done, progress, feedback, *, abort=False, regions=None) -> Dict[str, Any]:
    return {
        "done": bool(done),
        "abort": bool(abort),
        "progress": None if progress is None else float(np.clip(progress, 0.0, 1.0)),
        "feedback": str(feedback),
        "regions": list(regions or []),
    }


def _centers(traj) -> np.ndarray:
    """[T,3] nan-safe centroid per frame for ``traj`` [T,N,3]."""
    traj = np.asarray(traj, dtype=float)
    return np.nanmean(traj, axis=1)


def _aabb(lo, hi) -> Dict[str, Any]:
    return {"type": "aabb", "lo": [float(v) for v in lo], "hi": [float(v) for v in hi]}


def _footprint_region(ref_traj, *, xy_pad=0.02, z_height=0.15) -> Dict[str, Any]:
    """Container/support region: the xy bounding box of ``ref`` (last frame), from its
    base up by ``z_height``. Padded outward by ``xy_pad`` so a centred object counts."""
    pts = np.asarray(ref_traj, dtype=float)[-1]
    pts = pts[np.isfinite(pts).all(axis=-1)]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return _aabb(
        [lo[0] - xy_pad, lo[1] - xy_pad, lo[2]],
        [hi[0] + xy_pad, hi[1] + xy_pad, lo[2] + z_height],
    )


def _horizontal(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


# ── relations (PURE: operate on trajectories) ───────────────────────────────────────
def placed_in(target, reference, *, k=4, xy_pad=0.02, z_height=0.15, watch_drop=True):
    """``target`` came to rest INSIDE ``reference``'s 3D footprint and STAYED (last k)."""
    region = _footprint_region(reference, xy_pad=xy_pad, z_height=z_height)
    tc = _centers(target)
    frac = _lib.fraction_in_region(np.asarray(target)[-1], region)
    placed = _lib.persistence_in_region(target, region, k)
    center = (np.asarray(region["lo"]) + np.asarray(region["hi"])) / 2.0
    d = _horizontal(tc[-1], center)
    abort = False
    if watch_drop and not placed:
        dropped, frame = _lib.object_dropped(target, support_z=float(region["lo"][2]))
        if dropped:
            abort = True
            return _verdict(False, frac, f"target fell near frame {frame}; regrasp",
                            abort=True, regions=[region])
    fb = (f"in container (xy off {d*100:.1f}cm)" if placed
          else f"target {d*100:.1f}cm from container center; {frac*100:.0f}% of pts inside")
    return _verdict(placed, frac, fb, abort=abort, regions=[region])


def on_top_of(target, reference, *, k=3, xy_tol=0.04, z_gap=0.06):
    """``target`` rests ON the top surface of ``reference`` (xy-overlap + just above)."""
    ref = np.asarray(reference, dtype=float)
    ref_last = ref[-1][np.isfinite(ref[-1]).all(axis=-1)]
    lo, hi = ref_last.min(axis=0), ref_last.max(axis=0)
    ref_top = float(hi[2])
    region = _aabb([lo[0] - xy_tol, lo[1] - xy_tol, ref_top - 0.01],
                   [hi[0] + xy_tol, hi[1] + xy_tol, ref_top + z_gap])
    on = _lib.persistence_in_region(target, region, k)
    tc = _centers(target)[-1]
    dz = tc[2] - ref_top
    cx = (lo[:2] + hi[:2]) / 2.0
    dxy = _horizontal(tc, [cx[0], cx[1], 0])
    fb = (f"on top (dz {dz*100:+.1f}cm)" if on
          else f"not seated: dz {dz*100:+.1f}cm, xy off {dxy*100:.1f}cm")
    prog = float(np.clip(1.0 - abs(dz) / max(z_gap, 1e-6), 0, 1)) if dxy < (hi[0]-lo[0])/2 + xy_tol else 0.0
    return _verdict(on, prog, fb, regions=[region])


def next_to(target, reference, *, dist=0.10):
    """``target`` centroid within ``dist`` (horizontal) of ``reference`` centroid."""
    d = _horizontal(_centers(target)[-1], _centers(reference)[-1])
    done = d < dist
    prog = float(np.clip((2 * dist - d) / (2 * dist), 0, 1))
    return _verdict(done, prog, f"horizontal gap {d*100:.1f}cm (need <{dist*100:.0f}cm)")


def opened(target, *, travel=0.15, axis=None, open_frac=0.8):
    """Articulated open: ``target`` (handle/door) displaced ~its full travel this turn.

    ``travel`` is a PHYSICAL constant (metres). ``axis`` optional opening direction; if
    None, uses horizontal displacement magnitude (drawers/doors slide horizontally)."""
    c = _centers(target)
    disp = c[-1] - c[0]
    if axis is not None:
        u = np.asarray(axis, float)
        u = u / (np.linalg.norm(u) + 1e-9)
        along = float(np.dot(disp, u))
    else:
        along = float(np.linalg.norm(disp[:2]))
    prog = float(np.clip(along / max(travel, 1e-6), 0, 1))
    done = along >= open_frac * travel
    return _verdict(done, prog, f"handle moved {along*100:.1f}cm of ~{travel*100:.0f}cm")


def closed(target, *, travel=0.15, axis=None, close_frac=0.8, state=None):
    """Articulated close: ``target`` moved back toward closed by ~its travel this turn.

    Needs an OPEN baseline; if ``state`` (ctx.state) carries ``_open_pos`` we measure
    against it, else we use this turn's start as the open reference (a within-turn close)."""
    c = _centers(target)
    base = np.asarray(state["_open_pos"]) if (state and "_open_pos" in state) else c[0]
    disp = c[-1] - base
    along = float(np.linalg.norm(disp[:2])) if axis is None else \
        abs(float(np.dot(disp, np.asarray(axis, float) / (np.linalg.norm(axis) + 1e-9))))
    prog = float(np.clip(along / max(travel, 1e-6), 0, 1))
    return _verdict(along >= close_frac * travel, prog,
                    f"handle returned {along*100:.1f}cm of ~{travel*100:.0f}cm")


def lifted(target, *, support_z=None, lift=0.05, k=3):
    """``target`` picked up: centroid raised ``lift`` m above its support, sustained."""
    c = _centers(target)
    if support_z is None:
        early = max(1, len(c) // 4)
        support_z = float(np.nanmin(c[:early, 2]))
    done = bool(np.all(c[-k:, 2] > support_z + lift))
    prog = float(np.clip((c[-1, 2] - support_z) / max(lift, 1e-6), 0, 1))
    return _verdict(done, prog, f"lifted {(c[-1,2]-support_z)*100:.1f}cm (need >{lift*100:.0f}cm)")


def grasped(target, gripper, *, move_thresh=0.04, slip_thresh=0.03):
    """``target`` is held: it CO-MOVES with the gripper (the decorrelated grasp signal).

    Detects 'right motion, wrong outcome' — gripper moved but object did not follow → slip
    → abort early instead of finishing the whole trajectory."""
    rel = _lib.relative_motion(target, gripper)            # [T]
    grip_path = float(np.nansum(_lib.speed(gripper)))      # how far the gripper travelled
    moved = grip_path > move_thresh
    slip = float(rel[-1])
    done = moved and slip < slip_thresh
    abort = moved and slip > 2 * slip_thresh
    prog = float(np.clip(1.0 - slip / (2 * slip_thresh), 0, 1))
    fb = (f"held (slip {slip*100:.1f}cm)" if done
          else f"gripper moved {grip_path*100:.1f}cm but object slipped {slip*100:.1f}cm"
          if abort else f"slip {slip*100:.1f}cm, gripper travel {grip_path*100:.1f}cm")
    return _verdict(done, prog, fb, abort=abort)


def removed_from(target, reference, *, xy_pad=0.02, z_height=0.15, out_frac=0.15, lift=0.03):
    """``target`` taken OUT of ``reference``: few of its points remain in the footprint
    and it has been raised off the support."""
    region = _footprint_region(reference, xy_pad=xy_pad, z_height=z_height)
    frac = _lib.fraction_in_region(np.asarray(target)[-1], region)
    c = _centers(target)
    raised = c[-1, 2] - float(region["lo"][2])
    done = frac <= out_frac and raised > lift
    prog = float(np.clip(1.0 - frac / max(out_frac, 1e-6), 0, 1))
    return _verdict(done, prog, f"{frac*100:.0f}% still inside; raised {raised*100:.1f}cm",
                    regions=[region])


# ── relation registry + aliases ─────────────────────────────────────────────────────
_RELATIONS = {
    "place_in": placed_in, "on_top_of": on_top_of, "next_to": next_to,
    "opened": opened, "closed": closed, "lifted": lifted, "grasped": grasped,
    "removed_from": removed_from,
}
_ALIASES = {
    "placed_in": "place_in", "put_in": "place_in", "into": "place_in", "in": "place_in",
    "on": "on_top_of", "ontop": "on_top_of", "stack": "on_top_of", "stacked": "on_top_of",
    "beside": "next_to", "near": "next_to",
    "open": "opened", "close": "closed",
    "lift": "lifted", "pick_up": "lifted", "pickup": "lifted", "picked_up": "lifted",
    "grasp": "grasped", "holding": "grasped", "co_moving": "grasped",
    "remove_from": "removed_from", "take_out": "removed_from", "out_of": "removed_from",
}
_NEEDS_REFERENCE = {"place_in", "on_top_of", "next_to", "removed_from"}
_NEEDS_GRIPPER = {"grasped"}


def canonical(relation: str) -> str:
    r = relation.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(r, r)


# ── the one call the agent writes ───────────────────────────────────────────────────
def judge(ctx, relation: str, *, target: Optional[str] = None,
          reference: Optional[str] = None, gripper: str = "robot gripper",
          **opts) -> Dict[str, Any]:
    """Resolve named objects to 3D tracks via ``ctx`` and evaluate ``relation``.

    ``ctx.points_of(text) -> [T,Nt,3]`` is used to segment & track a named object;
    when unavailable (or it finds nothing) we fall back to the whole-frame grid
    (``ctx.coords``) so the judge still returns a verdict instead of crashing.
    """
    rel = canonical(relation)
    fn = _RELATIONS.get(rel)
    if fn is None:
        return _verdict(False, None, f"unknown relation {relation!r}; known: "
                        f"{sorted(_RELATIONS)}", abort=False)

    tgt = _resolve(ctx, target)
    kwargs: Dict[str, Any] = {}
    if rel in _NEEDS_REFERENCE:
        ref = _resolve(ctx, reference)
        if ref is None:
            return _verdict(False, None,
                            f"relation {rel!r} needs a reference object name", abort=False)
        args = (tgt, ref)
    elif rel in _NEEDS_GRIPPER:
        grip = _resolve(ctx, gripper)
        args = (tgt, grip if grip is not None else tgt)
    else:
        args = (tgt,)
    if rel in ("closed",):
        kwargs["state"] = getattr(ctx, "state", None)
    kwargs.update(opts)
    return fn(*args, **kwargs)


def all_of(ctx, *specs) -> Dict[str, Any]:
    """AND-combine several relations into one verdict (for multi-condition goals).

    Each ``spec`` is ``(relation, kwargs_dict)``, e.g. for
    "open the top drawer and put the bowl inside":

        return J.all_of(ctx,
            ("opened",   {"target": "drawer handle", "travel": 0.15}),
            ("place_in", {"target": "bowl", "reference": "top drawer"}))

    done = ALL done · abort = ANY abort (a failure in any sub-goal stops early) ·
    progress = mean · feedback/regions = joined.
    """
    parts = [judge(ctx, rel, **kw) for rel, kw in specs]
    if not parts:
        return _verdict(False, None, "all_of: no sub-goals given")
    progs = [p["progress"] for p in parts if p["progress"] is not None]
    return {
        "done": all(p["done"] for p in parts),
        "abort": any(p["abort"] for p in parts),
        "progress": float(np.mean(progs)) if progs else None,
        "feedback": " | ".join(p["feedback"] for p in parts),
        "regions": [r for p in parts for r in p.get("regions", [])],
    }


def _resolve(ctx, text: Optional[str]):
    """Named object → [T,Nt,3] trajectory. Whole-grid fallback (never raises)."""
    if text is None:
        return None
    pts = None
    if hasattr(ctx, "points_of"):
        try:
            pts = ctx.points_of(text)
        except Exception:
            pts = None
    if pts is None or np.asarray(pts).size == 0 or np.asarray(pts).shape[1] == 0:
        return np.asarray(ctx.coords, dtype=float)
    return np.asarray(pts, dtype=float)


# ── self-test (deterministic; runs with no services) ────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 12, 20

    # placed_in: target settles inside a container footprint and stays
    ref = np.zeros((T, N, 3)); ref[..., 0] = 0.4; ref[..., 1] = 0.0; ref[..., 2] = 0.02
    ref += rng.normal(0, 0.01, ref.shape)                       # a flat plate ~ (0.4, 0)
    tgt = np.zeros((T, N, 3))
    # approaches over the first half, then SETTLES on the plate and stays (persistence)
    xs = np.concatenate([np.linspace(0.7, 0.40, T // 2), np.full(T - T // 2, 0.40)])
    ys = np.concatenate([np.linspace(0.2, 0.00, T // 2), np.full(T - T // 2, 0.00)])
    tgt[..., 0] = xs[:, None]
    tgt[..., 1] = ys[:, None]
    tgt[..., 2] = 0.05
    v = placed_in(tgt, ref, z_height=0.15)
    assert v["done"] and v["progress"] > 0.8, v

    far = tgt.copy(); far[..., 0] += 0.5                        # ends far away
    assert not placed_in(far, ref)["done"]

    # opened: handle slides +Y ~0.15 m
    h = np.zeros((T, N, 3)); h[..., 1] = np.linspace(0.0, -0.15, T)[:, None]
    vo = opened(h, travel=0.15)
    assert vo["done"] and vo["progress"] > 0.95, vo
    assert not opened(h[:2], travel=0.15)["done"]               # barely moved

    # lifted: object rises 8 cm off support
    o = np.zeros((T, N, 3)); o[..., 2] = np.concatenate([np.zeros(4), np.linspace(0, 0.08, 8)])[:, None]
    assert lifted(o, lift=0.05)["done"]
    assert not lifted(np.zeros((T, N, 3)), lift=0.05)["done"]

    # grasped: object co-moves with gripper (rigid) vs slips
    grip = np.cumsum(np.ones((T, N, 3)) * 0.02, axis=0)
    held = grip + np.array([0.0, 0.0, -0.03])
    vg = grasped(held, grip); assert vg["done"] and not vg["abort"], vg
    slip = held.copy(); slip[:, :, 0] += np.linspace(0, 0.1, T)[:, None]
    vs = grasped(slip, grip); assert vs["abort"] and not vs["done"], vs

    # next_to
    a = np.zeros((T, N, 3)); b = np.zeros((T, N, 3)); b[..., 0] = 0.06
    assert next_to(a, b, dist=0.10)["done"]
    assert not next_to(a, b + np.array([0.5, 0, 0]), dist=0.10)["done"]

    # on_top_of: target sits just above a block top
    block = np.zeros((T, N, 3)); block[..., 2] = 0.05
    top = np.zeros((T, N, 3)); top[..., 2] = 0.08
    assert on_top_of(top, block)["done"]

    # removed_from: target leaves the footprint and rises
    out = ref.copy(); out[..., 0] += 0.6; out[..., 2] += 0.10
    assert removed_from(out, ref)["done"]
    assert not removed_from(ref + np.array([0, 0, 0.001]), ref)["done"]

    # judge() dispatch via a stub ctx (whole-grid fallback path)
    class _Stub:
        coords = tgt
        state: dict = {}
        def points_of(self, text):
            return {"bowl": tgt, "plate": ref}.get(text)
    vj = judge(_Stub(), "place_in", target="bowl", reference="plate")
    assert vj["done"], vj
    assert "unknown relation" in judge(_Stub(), "frobnicate", target="bowl")["feedback"]

    # all_of: multi-condition goal (open drawer AND place bowl) — AND semantics
    class _Stub2:
        coords = tgt
        state: dict = {}
        def points_of(self, text):
            return {"bowl": tgt, "plate": ref, "drawer handle": h}.get(text)
    combo = all_of(_Stub2(),
                   ("opened", {"target": "drawer handle", "travel": 0.15}),
                   ("place_in", {"target": "bowl", "reference": "plate"}))
    assert combo["done"] and "|" in combo["feedback"], combo
    combo2 = all_of(_Stub2(),
                    ("opened", {"target": "drawer handle", "travel": 1.0}),  # too-far travel → not done
                    ("place_in", {"target": "bowl", "reference": "plate"}))
    assert not combo2["done"], combo2

    print("judge_dsl self-test: all relations + dispatch + all_of OK")
