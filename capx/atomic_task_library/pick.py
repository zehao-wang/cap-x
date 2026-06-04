"""Atomic task: ``pick`` — collision-aware grasp of an object.

Promoted from the self-evolve loop (a human-confirmed Piper pick trial, then
generalized by the Feedback Postprocessor). Distilled lesson: a robust pick must

  1. lift the sampled grasp by a small WORLD-Z margin so it isn't too deep,
  2. approach through a pre-grasp pose above the target, and
  3. pass through intermediate clearance waypoints so the path planner is bounded
     away from the environment (it won't cut a colliding straight-line path).

Clearance heights adapt to the target object's measured geometry; the magnitudes
below are the atomic-task config (hyper-parameters). The function calls the
agent's primitives (``open_gripper`` / ``sample_grasp_pose`` / ``get_object_pose``
/ ``goto_pose`` / ``close_gripper``) as free globals — it is injected into the
code-execution namespace where those primitives live.
"""

import numpy as np

# --- pick atomic-task config (hyper-parameters) -----------------------------
PICK_GRASP_Z_MARGIN = 0.02        # world-Z lift on the sampled grasp (avoid too-deep grasps)
PICK_APPROACH_PULLBACK = 0.05     # horizontal pull-back of the entry waypoint (along -X)
PICK_HIGH_CLEARANCE = 0.20        # entry-waypoint clearance above the object top
PICK_MID_CLEARANCE = 0.10         # intermediate-waypoint clearance above the object top
PICK_PRE_GRASP_CLEARANCE = 0.05   # pre-grasp clearance above the object top before descent
PICK_FALLBACK_OBJECT_HEIGHT = 0.10  # used when the object's bbox extent is unavailable


def pick(object_name):
    """Pick up ``object_name`` with a collision-aware, multi-waypoint approach.

    Opens the gripper, samples a grasp pose, lifts it by ``PICK_GRASP_Z_MARGIN``,
    then approaches through high/mid/pre-grasp waypoints (heights derived from the
    object's measured top) before descending, closing, and lifting clear. Returns
    the final grasp pose ``(position, quaternion_wxyz)``.
    """
    open_gripper()

    pos, quat = sample_grasp_pose(object_name)
    pos = np.asarray(pos, dtype=float)
    pos[2] += PICK_GRASP_Z_MARGIN

    # Measure the object so clearance heights adapt to its size, not fixed numbers.
    obj_pos, _, obj_ext = get_object_pose(object_name, return_bbox_extent=True)
    obj_height = float(obj_ext[2]) if obj_ext is not None else PICK_FALLBACK_OBJECT_HEIGHT
    top_z = float(obj_pos[2]) + obj_height

    wp_high = pos.copy()
    wp_high[0] -= PICK_APPROACH_PULLBACK
    wp_high[2] = max(pos[2] + PICK_HIGH_CLEARANCE, top_z + PICK_HIGH_CLEARANCE)

    wp_mid = pos.copy()
    wp_mid[2] = max(pos[2] + PICK_MID_CLEARANCE, top_z + PICK_MID_CLEARANCE)

    pre_grasp = pos.copy()
    pre_grasp[2] = max(pos[2] + PICK_PRE_GRASP_CLEARANCE, top_z + PICK_PRE_GRASP_CLEARANCE)

    # Bounded approach: entry -> intermediate -> pre-grasp -> descend.
    goto_pose(wp_high, quat)
    goto_pose(wp_mid, quat)
    goto_pose(pre_grasp, quat)
    goto_pose(pos, quat)

    close_gripper()

    # Lift back to the high clearance waypoint so the object clears the scene.
    goto_pose(wp_high, quat)
    return pos, quat
