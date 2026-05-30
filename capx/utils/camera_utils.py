"""
Camera utility functions for processing observation data.
"""

from typing import Any

import numpy as np


def obs_get_rgb(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Recursively search through observation dictionary to find RGB images.

    Args:
        obs: Observation dictionary that may contain nested camera data

    Returns:
        Dictionary mapping camera names to RGB image arrays
    """
    rgb_dict = {}

    for key, value in obs.items():
        if isinstance(value, dict):
            # Check if this dict contains images with rgb data
            if "images" in value and isinstance(value["images"], dict):
                if "rgb" in value["images"]:
                    rgb_dict[key] = value["images"]["rgb"]
            else:
                # Recursively search in nested dictionaries
                nested_rgb = obs_get_rgb(value)
                rgb_dict.update(nested_rgb)

    return rgb_dict


def build_history_cameras(
    obs: dict[str, Any],
    render_camera_names: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the ``{name: {"image", "pose_xyz_wxyz", "fov"}}`` dict that
    :meth:`capx.utils.viser_history.ViserFrameHistory.record` expects.

    The recorded views are restricted to the task config's **standard
    observation views** (``render_camera_names``): exactly one for a single-view
    config, both for a wrist+external stereo config. Several simulators add an
    alias key to the raw observation for code-side convenience (e.g.
    ``robot0_robotview`` pointing at the *same* image as ``agentview``); that
    alias is not in ``render_camera_names``, so it is dropped here. Without this
    the viser history stitched two identical views for single-view tasks.

    Falls back to every RGB camera when ``render_camera_names`` is empty/None
    (e.g. envs that don't declare it), preserving prior behaviour for those.
    """
    rgb = obs_get_rgb(obs)
    names = [n for n in (render_camera_names or []) if n in rgb] or list(rgb)
    cameras: dict[str, dict[str, Any]] = {}
    for name in names:
        entry = obs.get(name)
        meta = entry if isinstance(entry, dict) else {}
        cameras[name] = {
            "image": rgb[name],
            "pose_xyz_wxyz": meta.get("pose"),
            "fov": meta.get("fov"),
        }
    return cameras
