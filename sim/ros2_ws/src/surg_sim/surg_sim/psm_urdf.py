"""Minimal PSM URDF for gz_ros2_control's controller_manager.

`controller_manager` (running inside the gz_ros2_control Gazebo plugin —
see `blender/export_sdf.py`'s `_add_ros2_control_block`) does NOT read the
`<ros2_control>` block embedded in the exported Gazebo SDF model on its
own: `hardware_interface::parse_control_resources_from_urdf` requires a
URDF-format `robot_description` published on the `/robot_description`
topic — confirmed against gz_ros2_control's own source — which nothing in
this stack provided before. This module builds that URDF (published by
`robot_state_publisher`, wired into `launch/_common.py`) with the same
per-joint command/state interfaces as the SDF, and with per-joint
`<origin>` elements computed from the same forward-kinematics transforms
(`blender/blenderproc_rerender.py:psm_link_transforms`) already relied on
elsewhere — not a second, independently hand-typed DH table that could
quietly drift from the first.

Per-joint origins here are provably exact, not approximate: for the
modified-DH convention `_dh_transform` uses (`R = Rx(alpha) @ Rz(theta)`),
`M(theta_offset + q) = M(theta_offset) @ Rz(q)` for a revolute joint and
`M(d_offset + q) = M(d_offset) @ Translate(0,0,q)` for the prismatic one
— both exactly, algebraically, for any q — which is precisely how a URDF
joint's `<origin>` (fixed) plus its `<axis>` (variable) composes. So
`axis="0 0 1"` and `origin` = the cumulative transform at q=0 reproduces
DH's own per-joint transform exactly for every q, not just at zero.

No `<visual>`/`<collision>`/`<inertial>`: those come from the
Blender-exported SDF that Gazebo actually renders and simulates. This
URDF exists only to satisfy `parse_control_resources_from_urdf` and give
`robot_state_publisher` a valid tree to publish TF from.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from blender.blenderproc_rerender import psm_link_transforms
from blender.export_sdf import PSM_LINKS
from kinematics.psm_kinematics import JOINT_LIMITS


def _origin_attrs(T: np.ndarray) -> dict:
    xyz = T[:3, 3]
    R = T[:3, :3]
    # XYZ-order roll/pitch/yaw extraction (URDF's <origin rpy="r p y">
    # convention), guarding the pitch = +-90 deg gimbal-lock case.
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if np.isclose(np.cos(pitch), 0.0, atol=1e-8):
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return {
        "xyz": f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}",
        "rpy": f"{roll:.6f} {pitch:.6f} {yaw:.6f}",
    }


def build_psm_urdf(links: list = PSM_LINKS) -> str:
    """Returns the PSM's URDF as an XML string. (No `<parameters>`/YAML
    reference here — that lives on the SDF-side plugin declaration in
    `blender/export_sdf.py`; this URDF's `<ros2_control>` block only
    needs to describe the hardware interfaces themselves.)
    """
    robot = ET.Element("robot", name="psm")

    link_names = [entry[0] for entry in links]
    ET.SubElement(robot, "link", name=link_names[0])  # base link, at the world origin

    cumulative = [np.eye(4)] + list(psm_link_transforms(np.zeros(6)))
    joint_idx = 0
    for i in range(1, len(links)):
        link_name, _blender_name, joint_name, joint_type, _sdf_axis = links[i]
        ET.SubElement(robot, "link", name=link_name)

        joint = ET.SubElement(robot, "joint", name=joint_name, type=joint_type)
        ET.SubElement(joint, "parent", link=link_names[i - 1])
        ET.SubElement(joint, "child", link=link_name)
        T_relative = np.linalg.inv(cumulative[i - 1]) @ cumulative[i]
        ET.SubElement(joint, "origin", **_origin_attrs(T_relative))
        ET.SubElement(joint, "axis", xyz="0 0 1")

        lo, hi = JOINT_LIMITS[joint_idx]
        ET.SubElement(joint, "limit", lower=f"{lo:.6f}", upper=f"{hi:.6f}", effort="10", velocity="1")
        joint_idx += 1

    ros2_control = ET.SubElement(robot, "ros2_control", name="GazeboSimSystem", type="system")
    hardware = ET.SubElement(ros2_control, "hardware")
    ET.SubElement(hardware, "plugin").text = "gz_ros2_control/GazeboSimSystem"
    for entry in links:
        joint_name = entry[2]
        if joint_name is None:
            continue
        joint_el = ET.SubElement(ros2_control, "joint", name=joint_name)
        ET.SubElement(joint_el, "command_interface", name="position")
        ET.SubElement(joint_el, "state_interface", name="position")
        ET.SubElement(joint_el, "state_interface", name="velocity")

    return ET.tostring(robot, encoding="unicode")
