"""judge_lib — pure-numpy geometric primitives for the self-evaluate tracking judge.

These are CONVENIENCE helpers the agent-written ``judge_state(ctx)`` may call. They
are NOT a cap: the judge may import any package. Every primitive operates on metric
3D point sets / trajectories (world frame, metres).

Conventions:
  - ``pts``  : [N, 3] point set for one frame.
  - ``traj`` : [T, N, 3] trajectory of N points over T frames.
  - a "region" is a dict; supported forms:
      {"type": "aabb", "lo": [x,y,z], "hi": [x,y,z]}
      {"type": "sphere", "center": [x,y,z], "radius": r}
"""

from __future__ import annotations

import numpy as np


def points_in_aabb(pts, lo, hi):
    """Boolean mask [N] — True where ``pts`` lies inside the axis-aligned box [lo, hi]."""
    pts = np.asarray(pts, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return np.all((pts >= lo) & (pts <= hi), axis=-1)


def points_in_region_3d(pts, region):
    """Boolean mask [N] — True where ``pts`` lies inside ``region`` (aabb or sphere)."""
    pts = np.asarray(pts, dtype=float)
    rtype = region.get("type", "aabb")
    if rtype == "aabb":
        return points_in_aabb(pts, region["lo"], region["hi"])
    if rtype == "sphere":
        center = np.asarray(region["center"], dtype=float)
        radius = float(region["radius"])
        return np.linalg.norm(pts - center, axis=-1) <= radius
    raise ValueError(f"unknown region type: {rtype!r}")


def centroid(pts):
    """Mean position [3] of a point set ``pts`` [N, 3]. NaN-safe (ignores NaN rows)."""
    pts = np.asarray(pts, dtype=float)
    return np.nanmean(pts, axis=0)


def fraction_in_region(pts, region):
    """Fraction (0..1) of ``pts`` inside ``region``."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] == 0:
        return 0.0
    return float(np.mean(points_in_region_3d(pts, region)))


def object_dropped(traj, support_z, drop_eps=0.03):
    """Did the object fall onto/below its support surface during ``traj``?

    ``traj`` [T, N, 3], ``support_z`` the support-surface height (metres). Returns
    ``(dropped: bool, frame: int)`` where ``frame`` is the first frame whose centroid
    z falls within ``drop_eps`` of (or below) ``support_z`` after having been lifted
    clearly above it earlier. ``frame`` is -1 when not dropped.
    """
    traj = np.asarray(traj, dtype=float)
    cz = np.nanmean(traj[..., 2], axis=1)  # [T] centroid height per frame
    lifted = cz > (support_z + 2.0 * drop_eps)
    for t in range(1, len(cz)):
        if np.any(lifted[:t]) and cz[t] <= (support_z + drop_eps):
            return True, t
    return False, -1


def relative_motion(obj_pts, ref_pts):
    """Co-motion / slip metric between two tracked sets over time.

    ``obj_pts`` [T, No, 3] and ``ref_pts`` [T, Nr, 3]. Returns the per-frame
    displacement [T] of the object centroid RELATIVE to the reference centroid,
    measured from frame 0 (i.e. how far the object slipped within the reference's
    frame). A rigid grasp keeps this near 0; a slip makes it grow.
    """
    obj_pts = np.asarray(obj_pts, dtype=float)
    ref_pts = np.asarray(ref_pts, dtype=float)
    obj_c = np.nanmean(obj_pts, axis=1)   # [T, 3]
    ref_c = np.nanmean(ref_pts, axis=1)   # [T, 3]
    rel = obj_c - ref_c                   # [T, 3] object in reference frame
    return np.linalg.norm(rel - rel[0], axis=-1)  # [T]


def min_pairwise_distance(a_pts, b_pts):
    """Minimum Euclidean distance (metres) between two point sets [Na,3], [Nb,3].

    Used for collision / clearance checks between object and arm/other object.
    """
    a_pts = np.asarray(a_pts, dtype=float)
    b_pts = np.asarray(b_pts, dtype=float)
    if a_pts.shape[0] == 0 or b_pts.shape[0] == 0:
        return float("inf")
    diff = a_pts[:, None, :] - b_pts[None, :, :]  # [Na, Nb, 3]
    return float(np.sqrt((diff ** 2).sum(-1)).min())


def speed(traj):
    """Per-frame centroid speed magnitude [T-1] (metres / frame) for ``traj`` [T,N,3]."""
    traj = np.asarray(traj, dtype=float)
    c = np.nanmean(traj, axis=1)  # [T, 3]
    return np.linalg.norm(np.diff(c, axis=0), axis=-1)


def persistence_in_region(traj, region, k):
    """Is the object's centroid inside ``region`` for ALL of the last ``k`` frames?

    Robust 'placed and stayed' test for ``traj`` [T,N,3]. Returns bool.
    """
    traj = np.asarray(traj, dtype=float)
    k = int(max(1, min(k, traj.shape[0])))
    last = traj[-k:]
    cents = np.array([centroid(last[t]) for t in range(last.shape[0])])  # [k, 3]
    return bool(np.all(points_in_region_3d(cents, region)))


# ── self-test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # points_in_aabb
    pts = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]], dtype=float)
    m = points_in_aabb(pts, [-1, -1, -1], [2, 2, 2])
    assert m.tolist() == [True, True, False], m

    # points_in_region_3d (sphere)
    sph = {"type": "sphere", "center": [0, 0, 0], "radius": 2.0}
    assert points_in_region_3d(pts, sph).tolist() == [True, True, False]
    aabb = {"type": "aabb", "lo": [-1, -1, -1], "hi": [2, 2, 2]}
    assert points_in_region_3d(pts, aabb).tolist() == [True, True, False]

    # centroid
    c = centroid(np.array([[0, 0, 0], [2, 2, 2]], dtype=float))
    assert np.allclose(c, [1, 1, 1]), c

    # fraction_in_region
    assert abs(fraction_in_region(pts, aabb) - 2 / 3) < 1e-9

    # object_dropped: lift then fall back to support z=0
    T, N = 10, 4
    traj = np.zeros((T, N, 3))
    heights = np.array([0.0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.2, 0.05, 0.0, 0.0])
    traj[:, :, 2] = heights[:, None]
    dropped, frame = object_dropped(traj, support_z=0.0, drop_eps=0.03)
    assert dropped and 6 < frame <= 9, (dropped, frame)
    # no drop when it stays up
    up = np.zeros((T, N, 3))
    up[:, :, 2] = 0.4
    assert object_dropped(up, support_z=0.0)[0] is False

    # relative_motion: rigid co-motion stays ~0, slip grows
    ref = np.cumsum(np.ones((T, N, 3)) * 0.1, axis=0)
    rigid = ref + np.array([0.5, 0.0, 0.0])           # fixed offset → no slip
    rm = relative_motion(rigid, ref)
    assert np.allclose(rm, 0.0, atol=1e-9), rm
    slipping = rigid.copy()
    slipping[:, :, 0] += np.linspace(0, 0.3, T)[:, None]  # drifts away
    rm2 = relative_motion(slipping, ref)
    assert rm2[-1] > 0.25, rm2

    # min_pairwise_distance
    a = np.array([[0, 0, 0]], dtype=float)
    b = np.array([[3, 4, 0], [1, 0, 0]], dtype=float)
    assert abs(min_pairwise_distance(a, b) - 1.0) < 1e-9
    assert min_pairwise_distance(a, np.zeros((0, 3))) == float("inf")

    # speed
    line = np.zeros((4, 2, 3))
    line[:, :, 0] = np.array([0, 1, 3, 6])[:, None]  # diffs 1,2,3
    sp = speed(line)
    assert np.allclose(sp, [1, 2, 3]), sp

    # persistence_in_region
    region = {"type": "aabb", "lo": [-0.1, -0.1, -0.1], "hi": [0.1, 0.1, 0.1]}
    stay = np.zeros((T, N, 3))            # centroid at origin throughout
    assert persistence_in_region(stay, region, k=3) is True
    leave = np.zeros((T, N, 3))
    leave[-1, :, 0] = 5.0                 # last frame leaves the box
    assert persistence_in_region(leave, region, k=3) is False

    print("judge_lib self-test: all primitives OK")
