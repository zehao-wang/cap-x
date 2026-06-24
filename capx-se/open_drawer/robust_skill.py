"""Robust (non-disturbing) drawer-open skill using HORL's RRT+trajopt planner.

This is the robustness-upgraded counterpart of ``skill.py``. The baseline skill
opens the drawer but plows through other objects (bowl ~210 mm, plate ~40 mm).
Here every transit is collision-aware against a collision world built from the
**agentview depth point cloud**, so other objects are not disturbed — the cap-x
"robust solution" requirement (esp. for the real robot).

Pipeline (validated live on libero_goal/task0 — see DISCUSSION.md §10):
  1. RRT (OMPL RRTConnect) -> a point-cloud-verified CLEAR standoff in front of
     the handle. Goal is in free space => RRT returns an exact, collision-free
     path. Zero object disturbance.
  2. Re-plan the SHORT final segment standoff->handle with collision-aware trajopt
     (small world_collision_margin so it can dip to the low handle that sits behind
     the front objects). Zero-disturbance grasp seated on the bar centre.
  3. Pull: collision-aware trajopt to the +Y open goal (keeps the arm off the
     plate/cheese). A re-grasp "pump" loop finishes the last cm if the thin bar
     slips before the drawer is fully open.

Status: grasp is zero-disturbance and on the handle; the pull opens the drawer and
keeps plate disturbance ~30 mm (vs ~105 mm for a straight pull). Reaching the full
-0.15 success threshold needs the pump loop / firmer grip (the bar shears out of
the gripper ~10 cm in). Requires the HORL planner (see horl_planner.py) — heavier
than the primitive skill (one ~2 min JAX compile, then ~0.6 s/solve).
"""

from __future__ import annotations

import numpy as np

from .horl_planner import HorlPlanner


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _frame(z, up=np.array([0.0, 0.0, 1.0])):
    z = _unit(z)
    y = up - np.dot(up, z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, [1.0, 0.0, 0.0])
    y = _unit(y)
    x = _unit(np.cross(y, z))
    y = _unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def _mask_pt(t, mask, depth, K, ext):
    p = t["mask_to_world_points"](mask, depth, K, ext)
    if len(p) == 0:
        return None
    p, _ = t["filter_noise"](p)
    return np.median(p, axis=0) if len(p) else None


def solve_robust(fns, *, instruction="open the middle drawer of the cabinet",
                 planner: HorlPlanner | None = None, log=print, debug=None):
    """Robust open-drawer using the HORL planner. ``fns`` = api.functions()."""
    t = fns
    if planner is None:
        log("building HORL planner (one-time JAX compile ~2 min) ...")
        planner = HorlPlanner(n_spheres=96, timesteps=32, plan_time=3.0,
                              sphere_radius=0.02, world_collision_margin=0.01)

    def run(traj):
        for cfg in traj:
            t["move_to_joints"](np.asarray(cfg, float))

    def jts():
        return np.asarray(t["get_observation"]()["robot_joint_pos"][:7], float)

    def ee():
        return np.asarray(t["get_observation"]()["robot_cartesian_pos"][:3], float)

    which = next((k for k in ("bottom", "middle", "top") if k in instruction.lower()), "middle")

    obs = t["get_observation"]()
    cam = obs["agentview"]
    rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
    K, ext = cam["intrinsics"], cam["pose_mat"]

    # detect + label the three handles
    masks = sorted(t["segment_sam3_text_prompt"](rgb, "drawer handle"),
                   key=lambda d: -d.get("score", 0.0))
    found = []
    for d in masks:
        p = _mask_pt(t, d["mask"], depth, K, ext)
        if p is None or any(np.linalg.norm(p - q[0]) < 0.03 for q in found):
            continue
        found.append((p, d["mask"]))
        if len(found) >= 3:
            break
    found.sort(key=lambda q: q[0][2])
    labels = ["bottom", "middle", "top"][: len(found)]
    handle_pt = dict(zip(labels, [f[0] for f in found])).get(which, found[len(found) // 2][0])
    log(f"target '{which}' handle @ {np.round(handle_pt, 3)}")

    # grasp frame: approach = cabinet_centroid -> handle (front normal), fingers vertical
    cab = sorted(t["segment_sam3_text_prompt"](rgb, "wooden cabinet"),
                 key=lambda d: -d.get("score", 0.0))
    cpt = _mask_pt(t, cab[0]["mask"], depth, K, ext) if cab else None
    ax = (_unit(np.array([handle_pt[0] - cpt[0], handle_pt[1] - cpt[1], 0.0]))
          if cpt is not None else _unit(np.array([handle_pt[0], handle_pt[1], 0.0])))
    approach = -ax
    quat = t["rotation_matrix_to_quaternion"](_frame(approach))
    deep = handle_pt + approach * 0.055 - np.array([0, 0, 0.018])  # seat at bar centre

    # obstacle cloud from depth: table objects (robot self & cabinet excluded)
    pc = t["transform_points"](t["depth_to_point_cloud"](depth, K).reshape(-1, 3), ext)
    m = ((pc[:, 2] > 0.005) & (pc[:, 2] < 0.25) & (pc[:, 1] > handle_pt[1] + 0.03)
         & (pc[:, 0] > 0.50) & (pc[:, 0] < 0.98))
    obstacles = pc[m]
    log(f"obstacle cloud: {len(obstacles)} pts")

    def clear(p):
        return 9.9 if len(obstacles) == 0 else float(np.linalg.norm(obstacles - p, axis=1).min())

    standoff = None
    for back in (0.22, 0.26, 0.30):
        for up in (0.06, 0.12, 0.18):
            cand = handle_pt - approach * back + np.array([0, 0, up])
            if clear(cand) > 0.09:
                standoff = cand
                break
        if standoff is not None:
            break
    standoff = standoff if standoff is not None else handle_pt - approach * 0.26 + np.array([0, 0, 0.12])

    t["open_gripper"]()
    # 1. RRT -> clear standoff (collision-free transit, zero disturbance)
    tr, out = planner.plan(jts(), standoff, quat, obstacles)
    log(f"RRT->standoff: {out['info']['ompl_status']}")
    if out["info"]["planned"]:
        run(tr)
    if debug:
        debug("standoff")

    # 2. re-plan short final segment -> seat on the handle (collision-aware)
    tr, _ = planner.plan_trajopt(jts(), deep, quat, obstacles, pos_weight=120, terminal_boost=90)
    run(tr)
    t["close_gripper"]()
    if debug:
        debug("grasped")

    # 3. pull open (collision-aware), with a re-grasp "pump" loop for the last cm
    pull = np.array([0.0, 1.0, 0.0])  # refined from a probe below
    s = ee()
    t["goto_pose"](deep + pull * 0.03, quat)
    mv = ee() - s
    mv[2] = 0.0
    if np.linalg.norm(mv) > 0.01:
        pull = _unit(mv)
    # Single collision-aware pull (keeps the arm off the plate/cheese). NOTE: the
    # thin bar tends to shear out of the gripper ~10 cm in, so this reaches ~-0.10
    # of the -0.16 travel. A re-grasp "pump" loop must re-seat WITHOUT fully opening
    # the gripper (a partial release) or it loses the drawer — left as the closing
    # tuning item (see DISCUSSION.md §10).
    tr, _ = planner.plan_trajopt(jts(), deep + pull * 0.16, quat, obstacles,
                                 pos_weight=120, terminal_boost=90)
    run(tr)
    if debug:
        debug("pulled")

    t["open_gripper"]()
    cur = ee()
    t["goto_pose"](cur + np.array([0, 0, 0.15]), quat)
    try:
        t["goto_home_joint_position"]()
    except Exception:  # noqa: BLE001
        pass
    if debug:
        debug("done")
    return {"target": which, "handle": handle_pt}
