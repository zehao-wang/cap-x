"""Visualization utilities for image overlays and annotations.

These are pure functions (no class state) shared across all control API
implementations.  They are used both for debug saves and for rich
execution-log images sent to the web UI.

Usage:
    from capx.utils.visualization_utils import (
        overlay_segmentation_masks,
        draw_oriented_bounding_box,
        draw_molmo_point,
        render_cylinder_axis,
    )
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as SciRotation


# ---------------------------------------------------------------------------
# Segmentation mask overlay
# ---------------------------------------------------------------------------

_PALETTE_HEX = [
    {"fill": "#feeeb2", "border": "#f9c500"},  # 1. Yellow
    {"fill": "#76b900", "border": "#265600"},  # 2. Green
    {"fill": "#f9d4ff", "border": "#952fc6"},  # 3. Purple
    {"fill": "#cbf5ff", "border": "#0074df"},  # 4. Blue
    {"fill": "#ffd7d7", "border": "#e52020"},  # 5. Red
]


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def overlay_segmentation_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    opacity: float = 0.5,
) -> np.ndarray:
    """Overlay up to 5 segmentation masks on an RGB image.

    Colors (in order): yellow, green, purple, blue, red.

    Args:
        image: (H, W, 3) uint8 RGB image.
        masks: List of boolean masks, each (H, W).
        opacity: Fill blend factor (0–1).

    Returns:
        (H, W, 3) uint8 image with masks overlaid.
    """
    if len(masks) > 5:
        masks = masks[:5]

    output = image.copy()

    for i in reversed(range(len(masks))):
        mask = masks[i]
        colors = _PALETTE_HEX[i]
        fill_rgb = _hex_to_rgb(colors["fill"])
        border_rgb = _hex_to_rgb(colors["border"])

        mask_indices = np.where(mask)
        roi = output[mask_indices[0], mask_indices[1]]
        blended = (opacity * roi + (1 - opacity) * np.array(fill_rgb)).astype(np.uint8)
        output[mask_indices[0], mask_indices[1]] = blended

        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, border_rgb, thickness=2)

    return output


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    opacity: float = 0.5,
) -> np.ndarray:
    """Overlay segmentation masks + bounding boxes + labels with scores.

    Args:
        image: (H, W, 3) uint8 RGB image.
        detections: List of dicts each with keys:
            ``"mask"`` (H, W bool, optional), ``"box"`` ([x1, y1, x2, y2],
            optional), ``"label"`` (str, optional), ``"score"`` (float, optional).
            Up to 5 detections are drawn (palette has 5 colors).
        opacity: Mask fill blend factor (0–1).

    Returns:
        (H, W, 3) uint8 image with detections overlaid.
    """
    output = image.copy()
    for i, det in enumerate(detections[:5]):
        colors = _PALETTE_HEX[i]
        fill_rgb = _hex_to_rgb(colors["fill"])
        border_rgb = _hex_to_rgb(colors["border"])

        mask = det.get("mask")
        if mask is not None:
            mask_idx = np.where(mask)
            roi = output[mask_idx[0], mask_idx[1]]
            blended = (opacity * roi + (1 - opacity) * np.array(fill_rgb)).astype(np.uint8)
            output[mask_idx[0], mask_idx[1]] = blended
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, border_rgb, thickness=2)

        box = det.get("box")
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(output, (x1, y1), (x2, y2), border_rgb, 2)

            label = det.get("label", "")
            score = det.get("score")
            if score is not None:
                text = f"{label} {score:.2f}" if label else f"{score:.2f}"
            else:
                text = label
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ty = max(y1 - 4, th + 4)
                cv2.rectangle(output, (x1, ty - th - 4), (x1 + tw + 4, ty), border_rgb, -1)
                cv2.putText(
                    output, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

    return output


# ---------------------------------------------------------------------------
# Oriented bounding box
# ---------------------------------------------------------------------------

def draw_oriented_bounding_box(
    image: np.ndarray,
    bbox_data: dict,
    world_to_camera_tf: np.ndarray,
    camera_intrinsics: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a 3D oriented bounding box projected onto an image.

    Args:
        image: (H, W, 3) input image.
        bbox_data: Dict with keys ``"center"`` (3,), ``"extent"`` (3,),
            ``"R"`` (3, 3) – as returned by
            ``get_oriented_bounding_box_from_3d_points``.
        world_to_camera_tf: (4, 4) extrinsic matrix (world → camera).
        camera_intrinsics: (3, 3) intrinsic matrix.
        color: RGB line colour.
        thickness: Line thickness.

    Returns:
        (H, W, 3) annotated image copy.
    """
    img_draw = image.copy()

    center = bbox_data["center"]
    extent = bbox_data["extent"]
    R_box = bbox_data["R"]

    dx, dy, dz = extent / 2.0
    corners_local = np.array([
        [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz],
    ])
    corners_world = (R_box @ corners_local.T).T + center

    corners_hom = np.hstack((corners_world, np.ones((8, 1))))
    corners_cam_hom = (world_to_camera_tf @ corners_hom.T).T
    corners_cam = corners_cam_hom[:, :3]

    corners_2d_hom = (camera_intrinsics @ corners_cam.T).T
    z_coords = corners_2d_hom[:, 2].copy()
    z_coords[z_coords == 0] = 1e-5
    corners_2d = (corners_2d_hom[:, :2] / z_coords[:, np.newaxis]).astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for s, e in edges:
        if z_coords[s] > 0 and z_coords[e] > 0:
            cv2.line(img_draw, tuple(corners_2d[s]), tuple(corners_2d[e]), color, thickness)

    return img_draw


# ---------------------------------------------------------------------------
# Molmo point annotation
# ---------------------------------------------------------------------------

def draw_molmo_point(
    image: np.ndarray,
    result: dict[str, tuple[int | None, int | None]],
    outer_radius: int = 12,
    inner_radius: int = 8,
    outer_color: tuple[int, int, int] = (255, 255, 255),
    inner_color: tuple[int, int, int] = (240, 82, 156),
) -> np.ndarray:
    """Draw Molmo point-prompt results on an image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        result: Mapping of object name → (x, y) pixel coordinate or ``None``.
        outer_radius: Radius of the outer circle.
        inner_radius: Radius of the inner circle.
        outer_color: RGB colour for outer ring.
        inner_color: RGB colour for inner dot.

    Returns:
        (H, W, 3) annotated image copy.
    """
    img_draw = image.copy()
    for _name, point in result.items():
        if point is not None:
            x, y = point
            cv2.circle(img_draw, (x, y), outer_radius, outer_color, -1)
            cv2.circle(img_draw, (x, y), inner_radius, inner_color, -1)
    return img_draw


_SAM3_DUMP_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 80, 255), (255, 200, 0),
    (255, 0, 255), (0, 255, 255),
]


def resolve_dump_dir(env, env_var_name: str, leaf: str):
    """Resolve where SAM3/Molmo intermediate dumps for this call should land.

    Prefers ``env.trial_artifact_dir`` (set by the trial runner at reset
    time, so each episode's intermediates land in their own subdir). Falls
    back to the legacy flat ``$CAPX_*_DUMP_DIR`` so existing setups keep
    working when the env attribute isn't set.
    """
    import os
    import pathlib

    trial_dir = getattr(env, "trial_artifact_dir", None)
    if trial_dir:
        return pathlib.Path(trial_dir) / leaf
    flat = os.environ.get(env_var_name)
    return pathlib.Path(flat) if flat else None


def _render_sam3_overlay(
    rgb: np.ndarray,
    sorted_results: list,
    top_k_overlay: int,
    point: "tuple[int, int] | None" = None,
) -> "Image.Image":
    """Compose mask+box (and optional point marker) overlay on top of RGB."""
    from PIL import ImageDraw

    rgb_pil = Image.fromarray(rgb).convert("RGBA")
    composed = rgb_pil.copy()
    for i, r in enumerate(sorted_results[:top_k_overlay]):
        color = _SAM3_DUMP_PALETTE[i % len(_SAM3_DUMP_PALETTE)]
        mask = r.get("mask")
        if mask is not None:
            mask_arr = np.asarray(mask).astype(bool)
            mask_layer = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
            mask_layer[mask_arr] = (*color, 110)
            composed = Image.alpha_composite(composed, Image.fromarray(mask_layer, mode="RGBA"))
    draw = ImageDraw.Draw(composed)
    for i, r in enumerate(sorted_results[:top_k_overlay]):
        color = _SAM3_DUMP_PALETTE[i % len(_SAM3_DUMP_PALETTE)]
        box = r.get("box")
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)
            draw.text((x1 + 2, y1 + 2), f"{i}:{r.get('score', 0):.2f}", fill=(*color, 255))
    if point is not None:
        x, y = int(point[0]), int(point[1])
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 0, 255), width=2)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 0, 255))
    return composed


def _save_sam3_dump(
    composed: "Image.Image",
    rgb: np.ndarray,
    results: list,
    sorted_results: list,
    dump_dir: "str | os.PathLike",
    base: str,
    title: str,
    extra_meta: dict,
) -> None:
    """Save a composed SAM3 overlay + meta JSON pair under <dump_dir>/<base>."""
    import json
    import pathlib

    from PIL import ImageDraw

    p = pathlib.Path(dump_dir)
    p.mkdir(parents=True, exist_ok=True)

    title_h = 24
    out_w, out_h = composed.size
    titled = Image.new("RGB", (out_w, out_h + title_h), (0, 0, 0))
    titled.paste(composed.convert("RGB"), (0, title_h))
    ImageDraw.Draw(titled).text((4, 4), title, fill=(255, 255, 255))
    titled.save(p / f"{base}.png")

    meta = {
        **extra_meta,
        "n_results": len(results),
        "image_hw": list(rgb.shape[:2]),
        "results": [
            {
                "score": float(r.get("score", 0) or 0),
                "box": list(r["box"]) if r.get("box") is not None else None,
                "mask_area_px": int(np.asarray(r["mask"]).sum()) if r.get("mask") is not None else None,
            }
            for r in results
        ],
    }
    (p / f"{base}.json").write_text(json.dumps(meta, indent=2))


def dump_sam3_call(
    rgb: np.ndarray,
    text_prompt: str,
    results: list,
    dump_dir: "str | os.PathLike",
    top_k_overlay: int = 5,
) -> None:
    """Persist a SAM3 text-prompt call: overlay PNG (top-K) + meta JSON (all)."""
    import os
    import time

    safe_prompt = "".join(c if c.isalnum() or c in "_-" else "_" for c in text_prompt)[:40]
    base = f"{time.time_ns()}_pid{os.getpid()}_{safe_prompt}_n{len(results)}"
    sorted_results = sorted(results, key=lambda d: d.get("score", 0) or 0, reverse=True)
    composed = _render_sam3_overlay(rgb, sorted_results, top_k_overlay)
    if results:
        title = (
            f"prompt={text_prompt!r}  n={len(results)}  "
            f"top1_score={sorted_results[0].get('score', 0):.3f}"
        )
    else:
        title = f"prompt={text_prompt!r}  n=0"
    _save_sam3_dump(
        composed, rgb, results, sorted_results, dump_dir, base, title,
        extra_meta={"text_prompt": text_prompt},
    )


def dump_sam3_point_call(
    rgb: np.ndarray,
    point_coords: "tuple[float, float]",
    results: list,
    dump_dir: "str | os.PathLike",
    top_k_overlay: int = 5,
) -> None:
    """Persist a SAM3 point-prompt call: overlay PNG (with point marker) + meta JSON."""
    import os
    import time

    x, y = int(point_coords[0]), int(point_coords[1])
    base = f"{time.time_ns()}_pid{os.getpid()}_pt_x{x}_y{y}_n{len(results)}"
    sorted_results = sorted(results, key=lambda d: d.get("score", 0) or 0, reverse=True)
    composed = _render_sam3_overlay(rgb, sorted_results, top_k_overlay, point=(x, y))
    if results:
        title = (
            f"point=({x},{y})  n={len(results)}  "
            f"top1_score={sorted_results[0].get('score', 0):.3f}"
        )
    else:
        title = f"point=({x},{y})  n=0"
    _save_sam3_dump(
        composed, rgb, results, sorted_results, dump_dir, base, title,
        extra_meta={"point_coords": [x, y]},
    )


def dump_molmo_call(
    rgb: np.ndarray,
    text_prompt: str,
    result: dict[str, tuple[int | None, int | None]],
    dump_dir: "str | os.PathLike",
) -> None:
    """Persist a Molmo point-prompt call: overlay PNG + meta JSON.

    Filenames are timestamp + pid + sanitized prompt to avoid collisions across
    parallel workers. Mirror of dump_sam3_call so SAM3 and Molmo dumps land in
    sibling directories.
    """
    import json
    import os
    import pathlib
    import time

    p = pathlib.Path(dump_dir)
    p.mkdir(parents=True, exist_ok=True)
    safe_prompt = "".join(c if c.isalnum() or c in "_-" else "_" for c in text_prompt)[:40]
    base = f"{time.time_ns()}_pid{os.getpid()}_{safe_prompt}"

    overlay = draw_molmo_point(rgb, result)
    # Annotate with prompt + parsed coord(s) on a top banner.
    title_h = 24
    out_h, out_w = overlay.shape[:2]
    titled = np.zeros((out_h + title_h, out_w, 3), dtype=np.uint8)
    titled[title_h:, :, :] = overlay
    pts_str = ", ".join(
        f"{name}={pt}" for name, pt in result.items()
    )
    cv2.putText(
        titled, f"prompt={text_prompt!r}  {pts_str}",
        (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    Image.fromarray(titled).save(p / f"{base}.png")

    meta = {
        "text_prompt": text_prompt,
        "image_hw": list(rgb.shape[:2]),
        "result": {
            name: list(pt) if pt is not None and pt[0] is not None else None
            for name, pt in result.items()
        },
    }
    (p / f"{base}.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# EEF axis overlay (cylinder rendering via pyrender)
# ---------------------------------------------------------------------------

def render_cylinder_axis(
    img_bg: np.ndarray,
    intrinsics: np.ndarray,
    world_to_cam: np.ndarray,
    pose_quat_wxyz: np.ndarray,
    pose_trans_xyz: np.ndarray,
    radius: float = 0.008,
    length: float = 0.1,
    opacity: float = 0.7,
) -> np.ndarray:
    """Render RGB XYZ axis cylinders at a given pose and overlay on an image.

    Requires ``pyrender`` and ``trimesh`` (lazy-imported so visualisation
    utils can be imported cheaply when these are not installed).

    Args:
        img_bg: (H, W, 3) background image.
        intrinsics: (3, 3) camera intrinsic matrix.
        world_to_cam: (4, 4) camera extrinsic (world → camera).
        pose_quat_wxyz: (4,) WXYZ quaternion of the frame to visualise.
        pose_trans_xyz: (3,) position of the frame to visualise.
        radius: Cylinder radius in metres.
        length: Cylinder length in metres.
        opacity: Blend factor for the overlay.

    Returns:
        (H, W, 3) image with axis cylinders overlaid.
    """
    import pyrender
    import trimesh

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    # Z (blue)
    cyl_z = trimesh.creation.cylinder(radius=radius, height=length, sections=20)
    cyl_z.visual.vertex_colors = [0, 0, 255, 255]
    cyl_z.apply_translation([0, 0, length / 2])

    # X (red)
    cyl_x = cyl_z.copy()
    cyl_x.visual.vertex_colors = [255, 0, 0, 255]
    cyl_x.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

    # Y (green)
    cyl_y = cyl_z.copy()
    cyl_y.visual.vertex_colors = [0, 255, 0, 255]
    cyl_y.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    axes_mesh = trimesh.util.concatenate([cyl_x, cyl_y, cyl_z])
    mesh = pyrender.Mesh.from_trimesh(axes_mesh)

    rot = SciRotation.from_quat(pose_quat_wxyz, scalar_first=True).as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rot
    pose_matrix[:3, 3] = pose_trans_xyz

    scene.add_node(pyrender.Node(mesh=mesh, matrix=pose_matrix))

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=10.0)

    flip_x = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ])
    cam_pose = world_to_cam @ flip_x

    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=img_bg.shape[1], viewport_height=img_bg.shape[0])
    color, depth = r.render(scene)

    mask = depth > 0
    img_result = img_bg.copy()
    img_result[mask] = (color[mask] * opacity + img_result[mask] * (1 - opacity)).astype(np.uint8)
    return img_result
