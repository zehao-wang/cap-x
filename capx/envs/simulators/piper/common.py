"""Shared helpers for the real Agilex PIPER integration."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

from scipy.spatial.transform import Rotation as SciRotation


DEFAULT_PIPER_URDF = os.environ.get(
    "PIPER_URDF_PATH",
    str(
        Path.home()
        / "Documents/Projects/robodata_Agilex/assets/piper_description/urdf/piper_description.urdf"
    ),
)
DEFAULT_PIPER_EEF_LINK = "gripper_base"

RAD_TO_RAW_MDEG = 1000.0 * 180.0 / math.pi
RAW_MDEG_TO_RAD = 1.0 / RAD_TO_RAW_MDEG
RAW_MM_TO_M = 0.001
PIPER_GRIPPER_RANGE_M = 0.08
PIPER_GRIPPER_EFFORT = 2000
PIPER_GRIPPER_CONTACT_EFFORT = 200
PIPER_GRIPPER_STATUS = 0x03
VISER_DEPTH_NEAR_M = float(os.environ.get("PIPER_VISER_DEPTH_NEAR_M", "0.05"))
VISER_DEPTH_FAR_M = float(os.environ.get("PIPER_VISER_DEPTH_FAR_M", "1.5"))
VISER_WRIST_DEPTH_NEAR_M = float(
    os.environ.get("PIPER_VISER_WRIST_DEPTH_NEAR_M", "0.0")
)
VISER_WRIST_DEPTH_FAR_M = float(
    os.environ.get("PIPER_VISER_WRIST_DEPTH_FAR_M", "1.0")
)


def _resolve_piper_mesh(fname: str) -> str:
    prefix = "package://piper_description/"
    if fname.startswith(prefix):
        base = Path(DEFAULT_PIPER_URDF).resolve().parent.parent
        return str(base / fname[len(prefix):])
    return fname


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_list(name: str, default: list[float], n: int) -> list[float]:
    value = os.environ.get(name)
    if not value:
        return default
    parts = value.replace(",", " ").split()
    if len(parts) != n:
        raise ValueError(f"{name} must contain {n} floats, got: {value!r}")
    return [float(v) for v in parts]


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = SciRotation.from_euler("xyz", rpy).as_matrix()
    mat[:3, 3] = xyz
    return mat


def _pose_xyz_wxyz_from_matrix(mat: np.ndarray) -> np.ndarray:
    rot = SciRotation.from_matrix(mat[:3, :3])
    xyzw = rot.as_quat()
    return np.array(
        [mat[0, 3], mat[1, 3], mat[2, 3], xyzw[3], xyzw[0], xyzw[1], xyzw[2]],
        dtype=np.float64,
    )


def _load_extrinsics(extrinsics_file: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Load camera pose in the robot base frame as ``(xyz, quat_wxyz)``."""
    if not extrinsics_file:
        return np.zeros(3, dtype=np.float64), np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
    if yaml is None:
        raise RuntimeError("pyyaml is required to read camera extrinsics")
    with open(extrinsics_file) as f:
        data = yaml.safe_load(f)
    pos = np.asarray(data.get("position", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    rpy = np.asarray(
        data.get("rpy_radians", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(3)
    xyzw = SciRotation.from_euler("xyz", rpy).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return pos, wxyz


def _depth_to_color_image(
    depth: np.ndarray,
    near: float = VISER_DEPTH_NEAR_M,
    far: float = VISER_DEPTH_FAR_M,
) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.clip((depth - near) / max(far - near, 1e-6), 0.0, 1.0)
    norm = np.where(valid, norm, 0.0)
    try:
        from matplotlib import colormaps  # type: ignore

        rgb = (colormaps["turbo"](norm)[..., :3] * 255).astype(np.uint8)
    except Exception:
        gray = ((1.0 - norm) * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~valid] = (40, 40, 40)
    return rgb


def _parse_urdf_vector(value: str | None, default: list[float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in value.split()], dtype=np.float64)


def _axis_motion_matrix(axis: np.ndarray, q: float, joint_type: str) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return mat
    axis = axis / norm
    if joint_type in {"revolute", "continuous"}:
        mat[:3, :3] = SciRotation.from_rotvec(axis * q).as_matrix()
    elif joint_type == "prismatic":
        mat[:3, 3] = axis * q
    return mat


class _UrdfLinkFk:
    """Small FK helper for dynamic wrist-camera pose, avoiding PyRoKi startup."""

    def __init__(self, urdf_path: str, target_link_name: str) -> None:
        self.urdf_path = urdf_path
        self.target_link_name = target_link_name
        root = ET.parse(urdf_path).getroot()

        self.joints: list[dict[str, Any]] = []
        self.children_by_parent: dict[str, list[dict[str, Any]]] = {}
        link_names = {el.attrib["name"] for el in root.findall("link") if "name" in el.attrib}
        child_links: set[str] = set()

        for joint_el in root.findall("joint"):
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            if parent_el is None or child_el is None:
                continue
            parent = parent_el.attrib.get("link")
            child = child_el.attrib.get("link")
            if not parent or not child:
                continue

            origin_el = joint_el.find("origin")
            xyz = _parse_urdf_vector(
                None if origin_el is None else origin_el.attrib.get("xyz"),
                [0.0, 0.0, 0.0],
            )
            rpy = _parse_urdf_vector(
                None if origin_el is None else origin_el.attrib.get("rpy"),
                [0.0, 0.0, 0.0],
            )
            axis_el = joint_el.find("axis")
            axis = _parse_urdf_vector(
                None if axis_el is None else axis_el.attrib.get("xyz"),
                [1.0, 0.0, 0.0],
            )
            joint = {
                "name": joint_el.attrib.get("name", ""),
                "type": joint_el.attrib.get("type", "fixed"),
                "parent": parent,
                "child": child,
                "origin": _transform_from_xyz_rpy(xyz, rpy),
                "axis": axis,
            }
            self.joints.append(joint)
            self.children_by_parent.setdefault(parent, []).append(joint)
            child_links.add(child)

        roots = sorted(link_names - child_links) or sorted(link_names)
        self.active_joint_names = [
            j["name"]
            for j in self.joints
            if j["type"] in {"revolute", "continuous", "prismatic"}
        ]
        self.root_link = ""
        chain = None
        for root_link in roots:
            candidate = self._find_chain(root_link, target_link_name)
            if candidate is not None:
                self.root_link = root_link
                chain = candidate
                break
        if chain is None:
            raise ValueError(
                f"Could not find link {target_link_name!r} in URDF {urdf_path!r}"
            )
        self.chain = chain

    def _find_chain(
        self, link_name: str, target_link_name: str
    ) -> list[dict[str, Any]] | None:
        if link_name == target_link_name:
            return []
        for joint in self.children_by_parent.get(link_name, []):
            subchain = self._find_chain(joint["child"], target_link_name)
            if subchain is not None:
                return [joint] + subchain
        return None

    def transform(self, joints: np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        q_by_name = {
            name: float(q[i])
            for i, name in enumerate(self.active_joint_names[: q.shape[0]])
        }
        mat = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            mat = mat @ joint["origin"]
            mat = mat @ _axis_motion_matrix(
                joint["axis"],
                q_by_name.get(joint["name"], 0.0),
                joint["type"],
            )
        return mat


__all__ = [
    "DEFAULT_PIPER_EEF_LINK",
    "DEFAULT_PIPER_URDF",
    "RAD_TO_RAW_MDEG",
    "RAW_MDEG_TO_RAD",
    "RAW_MM_TO_M",
    "VISER_DEPTH_FAR_M",
    "VISER_DEPTH_NEAR_M",
    "VISER_WRIST_DEPTH_FAR_M",
    "VISER_WRIST_DEPTH_NEAR_M",
    "PIPER_GRIPPER_CONTACT_EFFORT",
    "PIPER_GRIPPER_EFFORT",
    "PIPER_GRIPPER_RANGE_M",
    "PIPER_GRIPPER_STATUS",
    "_UrdfLinkFk",
    "_depth_to_color_image",
    "_env_bool",
    "_env_float_list",
    "_load_extrinsics",
    "_pose_xyz_wxyz_from_matrix",
    "_resolve_piper_mesh",
    "_transform_from_xyz_rpy",
]
