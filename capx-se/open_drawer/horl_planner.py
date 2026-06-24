"""Collision-aware (RRT+trajopt) motion planning for cap-x via HORL's pyroki solver.

HORL (`HumanOnlyRobotLearning`) ships a purpose-built, precompiled pure-JAX pyroki
trajectory solver with **fixed-shape collision pools** (env spheres/boxes/halfspaces
+ an attached-object pool) switched by masks — so positions change every call but
JIT never recompiles, as long as the pool sizes and horizon T are fixed.

We point it at cap-x's Franka assets and feed it a collision world built from the
**agentview depth point cloud** (the bowl/plate/etc.), so the reach routes around
other objects instead of plowing through them. Supports attaching a grasped object
(for the drawer pull). Runs in-process in .venv-libero (pyroki+jax present).

Key design (per the recompile constraint): N obstacle spheres and horizon T are
FIXED; each plan subsamples/pads the obstacle cloud to exactly N points.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HORL_SRC = os.environ.get(
    "HORL_SRC", "/home/zwa0839/Documents/Projects/HumanOnlyRobotLearning/src")
if _HORL_SRC not in sys.path:
    sys.path.append(_HORL_SRC)

_CAPX = os.environ.get("CAPX_ROOT", "/home/zwa0839/Documents/Projects/cap-x")

# Franka, using cap-x's own assets (curobo URDF + the pyroki server's sphere json).
ROBOT = {
    "urdf_path": os.path.join(
        _CAPX, "capx/third_party/curobo/src/curobo/content/assets/robot/"
        "franka_description/franka_panda.urdf"),
    "sphere_json_path": os.path.join(_CAPX, "capx/serving/assets/panda_spheres.json"),
    "target_link_name": "panda_hand",
    "eef_offset_local": [0.0, 0.0, 0.1034],   # flange -> TCP (grasp point) along local z
    "arm_num_dof": 7,
}
FINGERS_OPEN = [0.04, 0.04]


def _fixed_count(pts: np.ndarray, n: int) -> np.ndarray:
    """Subsample or pad a point set to exactly n rows (pad far away, inert)."""
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    if len(pts) == 0:
        return np.full((n, 3), 100.0, np.float32)         # all far away
    if len(pts) >= n:
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]
    pad = np.full((n - len(pts), 3), 100.0, np.float32)   # pad with far points
    return np.vstack([pts, pad])


class _SC:
    """Minimal scenario bag for OMPLPlanner / Harness (terminal-goal mode)."""
    def __init__(self, start_cfg, goal_pos, goal_wxyz, objects, T, attached=None):
        self.start_cfg = np.asarray(start_cfg, float)
        self._gp = np.asarray(goal_pos, float).reshape(3)
        self._gq = np.asarray(goal_wxyz, float).reshape(4)
        self.objects = objects
        self.timesteps = int(T)
        self.object_poses = np.array([[0, 0, 0, 0, 0, 0, 1.0]], float)  # scene at identity (xyzw)
        self.world_active = np.array([1])
        self.attached = attached
        self.workspace = None
        self.endpoint_sphere = None
        self.ref_pos = None
        self.ref_wxyz = None
        self.mode = "goal"

    def target_pos_wxyz(self):
        return self._gp, self._gq


class HorlPlanner:
    """Full RRT(OMPL RRTConnect) + trajopt-finish planner, robot=cap-x Franka,
    collision world from the depth point cloud. RRT is probabilistically complete
    so it reaches the goal whenever a collision-free path exists; the trajopt pass
    smooths it. Fixed T + sphere-pool size => the pyroki kernels compile once."""

    def __init__(self, n_spheres: int = 96, timesteps: int = 32,
                 sphere_radius: float = 0.025, plan_time: float = 5.0):
        # Bypass HORL's asset check (it points at HORL's own franka assets); we use
        # cap-x's. Harness then builds the solver from our ROBOT dict.
        import retargeting_solver.motion_plan.self_evolve.runner as _runner
        _runner.assert_assets_exist = lambda: None
        from retargeting_solver.motion_plan.ompl_planner import OMPLPlanner
        self.N = int(n_spheres)
        self.T = int(timesteps)
        self.r = float(sphere_radius)
        self.h = _runner.Harness(robot=ROBOT, project_root=_CAPX)
        self.solver = self.h.solver
        self.dof = int(self.h.dof)
        self.ompl = OMPLPlanner(self.h, plan_time=plan_time, trajopt=True,
                                trajopt_iters=80)

    def full_cfg(self, arm7) -> np.ndarray:
        """7 arm joints -> full actuated cfg (append open fingers if the model has them)."""
        arm7 = np.asarray(arm7, float).reshape(7)
        if self.dof >= 9:
            return np.concatenate([arm7, np.asarray(FINGERS_OPEN[: self.dof - 7])])
        return arm7

    def _world(self, obstacle_pts):
        from retargeting_solver.motion_plan.collision_pool import CollisionWorld
        pts = _fixed_count(obstacle_pts, self.N)
        objs = [{"name": "scene",
                 "spheres": [{"center": c.tolist(), "radius": self.r} for c in pts]}]
        return CollisionWorld(objs, None)

    def plan(self, start_arm7, goal_pos, goal_wxyz, obstacle_pts, attached=None):
        """RRT+trajopt plan to a TCP (grasp) goal in the EEF frame (solver back-
        projects the eef_offset). ``attached`` = {"object_index":0, "T_eef_obj":4x4}
        to carry the scene as a grasped object (drawer pull). Returns (traj_arm (T,7),
        info-dict with 'ompl_status')."""
        objs = self._scene_objects(obstacle_pts)
        sc = _SC(self.full_cfg(start_arm7), goal_pos, goal_wxyz, objs, self.T,
                 attached=({"object_index": 0,
                            "T_eef_obj": np.asarray(attached["T_eef_obj"], float)}
                           if attached is not None else None))
        out = self.ompl.solve(sc)
        return np.asarray(out["traj"], float)[:, :7], out

    def _scene_objects(self, obstacle_pts):
        pts = _fixed_count(obstacle_pts, self.N)
        return [{"name": "scene",
                 "spheres": [{"center": c.tolist(), "radius": self.r} for c in pts]}]

    def plan_trajopt(self, start_arm7, goal_pos, goal_wxyz, obstacle_pts,
                     pos_weight=60.0, terminal_boost=40.0, max_iterations=80):
        """Collision-aware trajopt (soft) start->goal. Use for the SHORT final
        segment / pull where the goal sits near obstacles (RRT's hard goal-validity
        rejects it, but soft trajopt reaches it while minimizing collision). Returns
        (traj_arm (T,7), info)."""
        from retargeting_solver.motion_plan.collision_pool import CollisionWorld
        world = CollisionWorld(self._scene_objects(obstacle_pts), None)
        poses = {"scene": np.array([0, 0, 0, 0, 0, 0, 1.0], float)}
        ws, wsm, wb, wbm, wh, whm = world.world_geoms(poses, {"scene": True})
        att, attm = world.attached_geoms(None)
        T = self.T
        ref_pos = np.tile(np.asarray(goal_pos, float).reshape(3), (T, 1))
        ref_wxyz = np.tile(np.asarray(goal_wxyz, float).reshape(4), (T, 1))
        ref_link = self.solver._back_project_batch(ref_pos, ref_wxyz)
        traj, info = self.solver.solve(
            self.full_cfg(start_arm7), ws, wsm, wb, wbm, wh, whm, att, attm,
            ref_link, ref_wxyz, np.zeros(T, np.float32), T,
            reference_pos_weight=pos_weight, terminal_boost=terminal_boost,
            max_iterations=max_iterations)
        return np.asarray(traj, float)[:, :7], info
