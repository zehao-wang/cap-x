"""Compositional solve for libero_goal/task3: 'open the top drawer and put the
bowl inside'. Sensing-only solve path (camera RGB-D + proprioception); NO privileged
poses. Reuses the proven, robust drawer-open skill for phase A, then adds a
sensing-only bowl PICK (SAM3 + Contact-GraspNet) and a PLACE into the open drawer.

The agentic decomposition (the LLM's job, which I play here): the instruction names
two sub-goals -- (1) open the TOP drawer, (2) put the bowl inside it -- which expand
to three motor phases: OPEN -> PICK bowl -> PLACE in the opened drawer. Success
(checked by the runner, never here) = bowl center inside wooden_cabinet_1_top_region,
which only exists once the drawer is open; so OPEN must succeed before PLACE can.

Geometry pinned from the asset + sensing calibration (solve / robot-base frame):
  - Top handle ~(0.69, -0.14, 0.18); drawer slides +outward (here +Y) by ~0.16 m.
  - The top_region success box rides WITH the drawer (it is under the sliding body):
    when open, its center sits ~0.11 m BEHIND the handle (-outward) at the handle's z.
    So the drop target = (re-detected open handle) - outward*0.11, released from above;
    the bowl then settles onto the drawer floor (center z well inside the box).
  - The bowl (akita_black_bowl) sits on the table in front, ~(0.53, 0.0, 0.02).
"""
from __future__ import annotations

import numpy as np

from .horl_planner import HorlPlanner
from .robust_skill import _frame, _mask_pt, solve_robust
from .runlog import as_logger


# ---- place geometry (cabinet affordance constants, not per-seed magic) ----
DRAWER_TRAVEL = 0.16          # full open travel of the drawer slide (handle moves +outward this far)
REGION_BEHIND_HANDLE = 0.11   # success box center sits this far -outward of the open handle
RELEASE_UP = 0.05             # descend the TCP to box-center + this, then release -> bowl drops in
CARRY_UP = 0.20               # transit height above the table/drawer while carrying
HIGH_APPROACH = 0.25          # approach the drop from this far above, then descend vertically
# ---- pick geometry. The akita bowl (~0.09 wide, rounded rim) is grasped via Contact-
# GraspNet (a hand-built top-down pinch slips it during the carry). solve_ik targets the TCP
# (robot_cartesian_pos reads ~0.11 above it); grip reads ~1.0 open, ~0.015 closed-empty. ----
BOWL_PRE_UP = 0.16            # pre-grasp standoff above the chosen grasp point
HAND_ABOVE_TCP = 0.113        # robot_cartesian_pos (panda_hand) reads this far above the IK'd TCP
GRASP_TRIES = 10             # graspnet candidates to try (by score) before giving up
GRIP_LO, GRIP_HI = 0.05, 0.40  # a firm bowl grasp reads here: excludes empty(~0.015) and
                               # over-grip(>0.4, which crushed past the rim and still slips)


def _best_mask(t, rgb, prompt):
    masks = sorted(t["segment_sam3_text_prompt"](rgb, prompt),
                   key=lambda d: -d.get("score", 0.0))
    return masks[0] if masks else None


def solve_compose(fns, *, instruction="open the top drawer and put the bowl inside",
                  planner: HorlPlanner | None = None, log=print, debug=None):
    """Open the named drawer, pick the bowl, place it inside. ``fns`` = api.functions().
    Returns a result dict. ``debug(tag)`` (optional) is the runner's privileged probe."""
    log = as_logger(log)
    t = fns
    if planner is None:
        log("building HORL planner (one-time JAX compile ~2 min) ...")
        planner = HorlPlanner(n_spheres=96, timesteps=32, plan_time=3.0,
                              sphere_radius=0.02, world_collision_margin=0.01)

    log.llm("decompose instruction -> ordered sub-goals",
            inputs="instruction(text)",
            decides="['open TOP drawer', 'pick bowl', 'place bowl in open drawer']")

    # ---------------- PHASE A: open the (top) drawer ----------------
    openres = solve_robust(t, instruction=instruction, planner=planner, log=log, debug=debug)
    if not openres.get("grasped"):
        log.llm("gate: drawer did not open -> cannot place; abort",
                inputs="open-phase result", decides="ABORT (no place attempted)")
        return {"phase": "open", "opened": False, **openres, "placed": False}

    def jts():
        return np.asarray(t["get_observation"]()["robot_joint_pos"][:7], float)

    def grip():
        return float(t["get_observation"]()["robot_joint_pos"][-1])

    def ee():
        return np.asarray(t["get_observation"]()["robot_cartesian_pos"][:3], float)

    def reset_ik_warmstart():
        # solve_ik warm-starts from the LAST IK result (api.cfg), NOT the current joints.
        # After the open phase + planner moves that cfg is stale (left at the side-approach
        # handle-seat config), so the next solve_ik lands in a bad IK branch the descend
        # can't escape. Reset it to None -> warm-start from the rest pose (~home), matching
        # the robot after goto_home. (Reaches the api instance via the bound method.)
        try:
            t["solve_ik"].__self__.cfg = None
        except Exception:
            pass

    def obs_cam():
        c = t["get_observation"]()["agentview"]
        return c["images"]["rgb"], c["images"]["depth"], c["intrinsics"], c["pose_mat"]

    # ---- place target, computed GEOMETRICALLY from the open phase (NOT re-sensed) ----
    # Re-detecting the open handle post-retreat is fragile (the arm occludes / the cabinet
    # centroid can flip `outward`). The open phase already gave us a reliable handle (pre-open)
    # + pull axis: the handle slid +outward by the full drawer travel (~0.16 m), and the
    # success box sits REGION_BEHIND_HANDLE -outward of the (open) handle, at the handle's z.
    handle = np.asarray(openres["handle"], float)
    outward = np.asarray(openres.get("outward", [0.0, 1.0, 0.0]), float)
    travel = float(openres.get("pull_travel", DRAWER_TRAVEL))
    open_handle = handle + outward * min(max(travel, 0.12), DRAWER_TRAVEL)
    drop = open_handle - outward * REGION_BEHIND_HANDLE
    drop[2] = handle[2]
    log.llm("locate drop target inside the open drawer (from open-phase geometry)",
            inputs="open-phase handle + pull axis + drawer travel [no new frame]",
            decides=f"outward={np.round(outward,2)} open_handle={np.round(open_handle,3)} "
                    f"drop={np.round(drop,3)}")

    # ---------------- PHASE B: pick the bowl ----------------
    rgb, depth, K, ext = obs_cam()
    bowl = _best_mask(t, rgb, "bowl")
    if bowl is None:
        log.llm("gate: bowl not detected -> abort place",
                inputs="agentview[1 frame]", decides="ABORT")
        return {"phase": "pick", "opened": True, "placed": False, "reason": "no bowl"}
    bpts = t["mask_to_world_points"](bowl["mask"], depth, K, ext)
    bpts, _ = t["filter_noise"](bpts)
    bcen = np.median(bpts, axis=0)
    rim_top = float(bpts[:, 2].max())
    log.local("SAM3", "segment bowl", inputs="agentview.rgb[1 frame]",
              outputs=f"bowl centroid {np.round(bcen,3)} rim_top z={rim_top:.3f}")

    # Bowl grasp via Contact-GraspNet. A naive top-down rim PINCH slips during the carry
    # (thin-wall line contact can't hold the bowl airborne); graspnet finds firm ANTIPODAL
    # grasps (grip ~0.17-0.23 vs the pinch's ~0.10) that hold. We rank downward-approaching
    # candidates by score and accept the first that actually descends AND grips in a holding
    # range -- some candidates don't reach (bad IK) or over-grip and still slip.
    # Pool several (stochastic) graspnet calls and keep only DOWNWARD grasps ON the bowl near
    # the rim -- graspnet returns different candidates each call and many off-bowl ones close on
    # air; pooling + filtering gives a reliable set, ranked by score.
    d2 = depth[:, :, 0] if depth.ndim == 3 else depth
    pool = {}
    n_raw = 0
    for _ in range(3):
        try:
            gposes_cam, gscores = t["plan_grasp"](d2, K, bowl["mask"])
        except Exception:  # noqa: BLE001
            continue
        n_raw += len(gscores)
        for i in range(len(gscores)):
            w = ext @ gposes_cam[i]
            pos = w[:3, 3]
            if w[2, 2] > -0.3:                                      # downward-ish approach
                continue
            if np.linalg.norm(pos[:2] - bcen[:2]) > 0.09:          # on the bowl+rim footprint
                continue
            if not (-0.03 < pos[2] < 0.07):                        # near the rim/table
                continue
            key = tuple(np.round(pos, 2))
            if key not in pool or gscores[i] > pool[key][2]:
                pool[key] = (pos.copy(), t["rotation_matrix_to_quaternion"](w[:3, :3]), float(gscores[i]))
    cands = sorted(pool.values(), key=lambda c: -c[2])
    if not cands:
        log.llm("gate: no on-bowl graspnet grasp -> abort",
                inputs=f"{n_raw} raw grasps", decides="ABORT")
        return {"phase": "pick", "opened": True, "placed": False, "reason": "no grasp"}
    log.llm("rank bowl-grasp candidates (Contact-GraspNet, pooled+filtered)",
            inputs=f"{n_raw} raw grasps over 3 calls",
            decides=f"{len(cands)} on-bowl candidates, try best {GRASP_TRIES}")

    grasp_p = grasp_q = None
    placed_ok = False

    def descend_and_grasp():
        """Try graspnet candidates: home -> pre -> stepwise descend -> close; accept the first
        that reaches the grasp AND grips firmly. The TRUE-home reset gives the clean IK warm-
        start (from the post-open pose pyroki returns high solutions and the descend stalls)."""
        nonlocal grasp_p, grasp_q
        for gi, (gp, gq, sc) in enumerate(cands[:GRASP_TRIES]):
            prev = None                          # reach TRUE home (coarse interp needs repeats)
            for _ in range(4):
                t["goto_home_joint_position"]()
                cur = ee()
                if prev is not None and np.linalg.norm(cur - prev) < 0.01:
                    break
                prev = cur
            reset_ik_warmstart()
            t["open_gripper"]()
            pre = gp + np.array([0.0, 0.0, BOWL_PRE_UP])
            try:
                t["move_to_joints"](np.asarray(t["solve_ik"](pre, gq), float))
            except Exception:
                continue
            for k in range(1, 8):
                z = pre[2] + (gp[2] - pre[2]) * k / 7
                try:
                    t["move_to_joints"](np.asarray(t["solve_ik"]([gp[0], gp[1], z], gq), float))
                except Exception:
                    break
            if ee()[2] > gp[2] + HAND_ABOVE_TCP + 0.05:   # descend didn't reach the grasp
                log.llm("verify grasp candidate -> descend short, next",
                        inputs="proprio(achieved TCP z)", decides=f"grasp{gi}: ee_z={ee()[2]:.3f} -> skip")
                continue
            t["close_gripper"]()
            held = grip()
            if GRIP_LO < held < GRIP_HI:
                grasp_p, grasp_q = gp, gq
                log.llm("evaluate bowl grasp -> holding",
                        inputs="proprio(descend reached + closed-grip)",
                        decides=f"grasp{gi} (score {sc:.2f}): ee_z={ee()[2]:.3f} grip={held:.3f} -> HOLD")
                return True
            log.llm("evaluate bowl grasp -> empty/over-grip, next",
                    inputs="proprio(closed-grip)", decides=f"grasp{gi}: grip={held:.3f} -> skip")
        return False

    log.local("Contact-GraspNet+pyroki-IK", "home -> grasp candidate -> robust descend",
              inputs="bowl mask + depth -> grasp poses", outputs="grasp")
    holding = descend_and_grasp()
    if debug:
        debug("bowl_grasped")
    if holding:
        # lift straight up to carry height, in fine Cartesian steps (the rim pinch grasp
        # holds a vertical load but pops out under a jerky move -- move_to_joints interpolates
        # coarsely, so big steps = inertia = slip). Re-close between steps to keep it seated.
        lift = grasp_p + np.array([0.0, 0.0, CARRY_UP])
        gx0, gy0 = grasp_p[0], grasp_p[1]
        for u in np.linspace(0.0, 1.0, 18):
            z = grasp_p[2] + (lift[2] - grasp_p[2]) * u
            try:
                t["move_to_joints"](np.asarray(t["solve_ik"]([gx0, gy0, z], grasp_q), float))
            except Exception:
                break
            t["close_gripper"]()  # RE-tighten every step: the gripper is position-held, so as the
                                  # bowl slides down through the pinch under its weight the grip
                                  # loosens (0.10 -> 0.05) and slips; re-closing re-seats it.
        if debug:
            debug("bowl_lift")

        # ---------------- PHASE C: place into the open drawer ----------------
        # Carry HIGH above the drop via solve_ik waypoints (NOT the planner): the bowl is
        # lifted above everything, so a high straight transit is collision-free, and using
        # solve_ik keeps api.cfg in sync (a planner move would desync the IK warm-start and
        # break the following descend, the same stale-cfg bug as the pick). Then descend
        # straight DOWN into the drawer -- coming in high and dropping vertically also keeps
        # the arm off the protruding open drawer front (a level carry shoved it closed:
        # qpos -0.16 -> -0.11, retracting the success region).
        # Aim the TCP so the BOWL CENTER lands at the region center: the bowl center sits at
        # (bcen - grasp_p) relative to the TCP (graspnet grasps the rim off-centre), so offset
        # the TCP target by that vector. Drop above the box and let the bowl settle in.
        place = drop.copy()
        place[:2] = drop[:2] - (bcen[:2] - grasp_p[:2])
        high = place + np.array([0.0, 0.0, HIGH_APPROACH])
        log.local("pyroki-IK", "carry bowl high above the open drawer (solve_ik waypoints)",
                  inputs="lift pose -> high-above-drop", outputs="joint waypoints")
        src = ee() - np.array([0.0, 0.0, HAND_ABOVE_TCP])   # current TCP
        for u in np.linspace(0.0, 1.0, 40):                 # MANY fine waypoints: a smooth,
            wp = src * (1 - u) + high * u                   # low-inertia transit so the pinched
            try:                                            # rim grasp doesn't slip (cfg stays
                t["move_to_joints"](np.asarray(t["solve_ik"](wp, grasp_q), float))  # in sync too)
            except Exception:
                break
            t["close_gripper"]()  # re-tighten each waypoint to hold the bowl through the carry
        t["close_gripper"]()      # re-seat before descending into the drawer
        if debug:
            debug("above_drop")
        # descend straight down into the open drawer (stepwise IK re-solve, TCP frame) + release
        rel = place + np.array([0.0, 0.0, RELEASE_UP])
        tcp_start_z = ee()[2] - HAND_ABOVE_TCP   # current TCP z (solve_ik targets the TCP)
        n = 7
        for k in range(1, n + 1):
            z = tcp_start_z + (rel[2] - tcp_start_z) * k / n
            try:
                t["move_to_joints"](np.asarray(t["solve_ik"]([rel[0], rel[1], z],
                                                             grasp_q), float))
            except Exception:
                break
        if debug:
            debug("at_release")
        t["open_gripper"]()
        placed_ok = True
        # retreat up and back, clear of the cabinet
        up = rel + np.array([0.0, outward[1] * 0.06, CARRY_UP])
        try:
            ikj = np.asarray(t["solve_ik"](up, grasp_q), float)
            cur = jts()
            for u in np.linspace(0.0, 1.0, 6):
                t["move_to_joints"](cur * (1 - u) + ikj * u)
        except Exception:
            pass
        if debug:
            debug("done")

    return {"phase": "place", "opened": True, "placed": placed_ok,
            "drop": drop, "bowl": bcen, **{k: openres[k] for k in ("handle",) if k in openres}}
