"""Solve the LIBERO drawer-opening task with the legitimate cap-x API.

All inputs are real-robot-obtainable: agentview RGB-D (SAM3 segmentation,
depth->3D), and proprioception (``robot_cartesian_pos`` / ``robot_joint_pos``).
No privileged sim object/joint state is read in the solving path.

Pipeline (validated live on libero_goal_task/task0):
  1. Parse which drawer (top/middle/bottom) from the *instruction string*.
  2. SAM3-segment all three drawer handles; lift each mask to a 3D point;
     sort by height -> label top/middle/bottom; pick the requested one.
  3. Estimate the prismatic (pull) axis from perception: horizontal direction
     from the cabinet-body centroid to the handle (the drawer-front outward
     normal).  Refine it later by probing.
  4. Reach a front grasp with CuRobo (collision-free joint plan; accurate IK),
     seating the handle bar a few cm inside the gripper; fall back to pyroki IK.
  5. Close, verify the grip actually holds (gripper not fully closed).
  6. Probe: pull a small step along the estimate, re-observe the handle, and set
     the true axis = measured 3D displacement.  Then pull to full open.

See GAPS.md for the cap-x issues this exercise surfaced.
"""

from __future__ import annotations

import numpy as np


class Tools:
    """Fetch an API function by name; a missing name is a vocabulary hole."""

    def __init__(self, fns: dict):
        self._fns = dict(fns)

    def __getattr__(self, name: str):
        try:
            return self._fns[name]
        except KeyError as exc:  # noqa: TRY003
            raise AttributeError(
                f"cap-x API does not expose '{name}'. Available: {sorted(self._fns)}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self._fns


# ----------------------------- geometry helpers ---------------------------- #
def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _frame_from_approach(approach, up=np.array([0.0, 0.0, 1.0])):
    """Panda-hand rotation: +Z = approach, +Y = finger open/close (~up_hint)."""
    z = _unit(approach)
    y = up - np.dot(up, z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, [1.0, 0.0, 0.0])
    y = _unit(y)
    x = _unit(np.cross(y, z))
    y = _unit(np.cross(z, x))
    return np.column_stack([x, y, z])


def _mask_point(t, mask, depth, K, ext):
    pts = t.mask_to_world_points(mask, depth, K, ext)
    if len(pts) == 0:
        return None
    pts, _ = t.filter_noise(pts)
    if len(pts) == 0:
        return None
    return np.median(pts, axis=0)


def _which_drawer(instruction: str) -> str:
    s = (instruction or "").lower()
    for key in ("bottom", "middle", "top"):
        if key in s:
            return key
    return "middle"


def _detect_handles(t, rgb, depth, K, ext, log):
    """Return {label: (point3d, mask)} for the (up to) 3 stacked handles."""
    masks = t.segment_sam3_text_prompt(rgb, "drawer handle")
    masks = sorted(masks, key=lambda d: -d.get("score", 0.0))
    found = []
    for d in masks:
        p = _mask_point(t, d["mask"], depth, K, ext)
        if p is None:
            continue
        # dedupe by 3D proximity (SAM3 returns near-duplicate masks)
        if any(np.linalg.norm(p - q[0]) < 0.03 for q in found):
            continue
        found.append((p, d["mask"]))
        if len(found) >= 3:
            break
    if not found:
        raise RuntimeError("no drawer handle detected")
    found.sort(key=lambda q: q[0][2])  # ascending height
    labels = ["bottom", "middle", "top"][: len(found)]
    out = {labels[i]: found[i] for i in range(len(found))}
    for lb, (p, _) in out.items():
        log(f"  handle[{lb}] @ {np.round(p, 3)}")
    return out


def _estimate_pull_axis(t, rgb, depth, K, ext, handle_pt, log):
    """Horizontal cabinet-front outward normal ~= dir(cabinet_centroid -> handle)."""
    try:
        cab = sorted(t.segment_sam3_text_prompt(rgb, "wooden cabinet"),
                     key=lambda d: -d.get("score", 0.0))
        cpt = _mask_point(t, cab[0]["mask"], depth, K, ext)
    except Exception:  # noqa: BLE001
        cpt = None
    if cpt is None:
        return _unit(np.array([handle_pt[0], handle_pt[1], 0.0]))
    d = handle_pt - cpt
    d[2] = 0.0
    log(f"  cabinet centroid {np.round(cpt,3)} -> pull axis estimate {np.round(_unit(d),3)}")
    return _unit(d)


# --------------------------------- skill ----------------------------------- #
def solve(fns, *, instruction="open the middle drawer of the cabinet",
          log=print, debug=None, use_curobo=True,
          grasp_depth=0.03, full_open=0.18) -> dict:
    t = Tools(fns)
    np.set_printoptions(precision=3, suppress=True)

    def _dbg(s):
        if debug:
            try:
                debug(s)
            except Exception as e:  # noqa: BLE001
                log(f"  [debug hook: {e!r}]")

    target = _which_drawer(instruction)
    log(f"instruction={instruction!r} -> target drawer = {target}")

    t.open_gripper()
    try:
        t.goto_home_joint_position()
    except Exception as e:  # noqa: BLE001
        log(f"home failed (continuing): {e!r}")
    _dbg("start")

    obs = t.get_observation()
    cam = obs["agentview"]
    rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
    K, ext = cam["intrinsics"], cam["pose_mat"]

    handles = _detect_handles(t, rgb, depth, K, ext, log)
    if target not in handles:
        log(f"requested '{target}' not found; have {sorted(handles)} — using highest-score")
        target = next(iter(handles))
    handle_pt, handle_mask = handles[target]

    pull_axis = _estimate_pull_axis(t, rgb, depth, K, ext, handle_pt, log)
    approach = -pull_axis                      # gripper points into the drawer front
    quat = t.rotation_matrix_to_quaternion(_frame_from_approach(approach))
    log(f"approach={np.round(approach,3)}  grasp_quat={np.round(quat,3)}")

    # ---- reach: pre-grasp then a few cm deeper so the bar seats in the gripper #
    pre = handle_pt - approach * 0.10
    deep = handle_pt + approach * grasp_depth

    def _reach(pos, mask):
        if use_curobo and t.has("plan_grasp_trajectory") and t.has("execute_joint_trajectory"):
            try:
                ok, traj, _ = t.plan_grasp_trajectory(
                    "drawer handle", object_mask=mask,
                    grasp_poses=[(pos, quat)], use_world_collision=False)
                if ok and traj is not None:
                    t.execute_joint_trajectory(traj, subsample=2, max_steps=250)
                    return True
            except Exception as e:  # noqa: BLE001
                log(f"  curobo reach failed ({e!r}); pyroki fallback")
        t.goto_pose(pos, quat)
        return False

    log("Reaching pre-grasp ...")
    _reach(pre, handle_mask)
    _dbg("pre")
    log("Reaching handle (deep) ...")
    _reach(deep, handle_mask)
    _dbg("at_handle")

    t.close_gripper()
    grip = float(t.get_observation()["robot_joint_pos"][-1])
    log(f"closed; gripper opening={grip:.3f} (>~0.02 means holding something)")
    _dbg("grasped")
    if grip < 0.02:
        log("GRASP FAILED: gripper closed empty — handle not reachable/seated.")
        return {"target": target, "grasped": False, "handle": handle_pt}

    # ---- probe to measure the true prismatic axis, then pull to full open ---- #
    def _ee():
        return np.asarray(t.get_observation()["robot_cartesian_pos"][:3], float)

    start_ee = _ee()
    t.goto_pose(handle_pt + pull_axis * 0.03, quat)
    moved = _ee() - start_ee
    moved[2] = 0.0
    if np.linalg.norm(moved) > 0.01:
        pull_axis = _unit(moved)
        log(f"  probe refined pull axis -> {np.round(pull_axis,3)}")
    _dbg("probe")

    log(f"Pulling open along {np.round(pull_axis,3)} ...")
    base = _ee()
    n = max(1, int(np.ceil(full_open / 0.015)))
    for i in range(1, n + 1):
        try:
            t.goto_pose(base + pull_axis * (full_open * i / n), quat)
        except Exception as e:  # noqa: BLE001
            log(f"  pull step {i}/{n} IK failed: {e!r}")
            break
        _dbg(f"pull_{i}")
    t.open_gripper()
    _dbg("released")
    return {"target": target, "grasped": True, "handle": handle_pt, "pull_axis": pull_axis}
