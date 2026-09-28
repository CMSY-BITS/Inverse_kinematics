"""psm_urdf.py lives inside the surg_sim ROS 2 package (not a top-level
repo package), so it needs sim/ros2_ws/src/surg_sim on sys.path here --
the same thing colcon's generated setup.bash does for the real ROS 2
environment. No bpy/blenderproc/rclpy needed: this is pure XML/numpy.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_SURG_SIM_SRC = Path(__file__).resolve().parent.parent / "sim" / "ros2_ws" / "src" / "surg_sim"
if str(_SURG_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(_SURG_SIM_SRC))

from surg_sim.psm_urdf import build_psm_urdf  # noqa: E402

from blender.export_sdf import PSM_LINKS  # noqa: E402
from kinematics.psm_kinematics import PSMKinematics  # noqa: E402


def test_urdf_is_well_formed_xml():
    root = ET.fromstring(build_psm_urdf())  # raises if malformed
    assert root.tag == "robot"
    assert root.get("name") == "psm"


def test_link_count_and_names_match_psm_links():
    root = ET.fromstring(build_psm_urdf())
    links = root.findall("link")
    assert [link.get("name") for link in links] == [entry[0] for entry in PSM_LINKS]


def test_joint_count_and_names_match_psm_links():
    root = ET.fromstring(build_psm_urdf())
    expected_names = [entry[2] for entry in PSM_LINKS if entry[2] is not None]
    assert [j.get("name") for j in root.findall("joint")] == expected_names


def test_every_joint_axis_is_local_z():
    # Provably correct for this DH convention -- see psm_urdf.py's module
    # docstring for the algebraic derivation, not just asserted here.
    root = ET.fromstring(build_psm_urdf())
    for joint in root.findall("joint"):
        assert joint.find("axis").get("xyz") == "0 0 1"


def test_joint_limits_match_kinematics_joint_limits():
    from kinematics.psm_kinematics import JOINT_LIMITS

    root = ET.fromstring(build_psm_urdf())
    for i, joint in enumerate(root.findall("joint")):
        limit = joint.find("limit")
        lo, hi = JOINT_LIMITS[i]
        np.testing.assert_allclose(float(limit.get("lower")), lo, atol=1e-6)
        np.testing.assert_allclose(float(limit.get("upper")), hi, atol=1e-6)


def test_ros2_control_block_has_matching_hardware_interfaces():
    root = ET.fromstring(build_psm_urdf())
    ros2_control = root.find("ros2_control")
    assert ros2_control is not None
    assert ros2_control.find("hardware/plugin").text == "gz_ros2_control/GazeboSimSystem"

    expected_names = [entry[2] for entry in PSM_LINKS if entry[2] is not None]
    assert [j.get("name") for j in ros2_control.findall("joint")] == expected_names

    for joint in ros2_control.findall("joint"):
        assert [c.get("name") for c in joint.findall("command_interface")] == ["position"]
        assert [s.get("name") for s in joint.findall("state_interface")] == ["position", "velocity"]


def _homogeneous_from_urdf_origin(origin_el) -> np.ndarray:
    xyz = np.array([float(v) for v in origin_el.get("xyz").split()])
    roll, pitch, yaw = (float(v) for v in origin_el.get("rpy").split())
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx  # URDF's rpy convention: R = Rz(yaw) Ry(pitch) Rx(roll)
    T[:3, 3] = xyz
    return T


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def _translate_z(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


def test_joint_origins_exactly_reproduce_forward_kinematics_at_nonzero_q():
    """The real correctness claim in psm_urdf.py's docstring: composing
    each joint's fixed <origin> (computed at q=0) with a live rotation/
    translation about its own local z-axis by the joint's *current* value
    must exactly reproduce PSMKinematics.forward(q) for any q -- not just
    at q=0, where the origins happen to have been derived from.
    """
    root = ET.fromstring(build_psm_urdf())
    joint_entries = [entry for entry in PSM_LINKS if entry[2] is not None]

    rng = np.random.default_rng(0)
    q = rng.uniform(-0.3, 0.3, size=6)

    T_urdf = np.eye(4)
    for joint, qi, entry in zip(root.findall("joint"), q, joint_entries):
        origin_T = _homogeneous_from_urdf_origin(joint.find("origin"))
        motion = _translate_z(qi) if entry[3] == "prismatic" else _rot_z(qi)
        T_urdf = T_urdf @ origin_T @ motion

    kin = PSMKinematics()
    # T_urdf accumulates through the wrist-yaw link only (no final
    # tool-tip offset, same as psm_link_transforms' last entry) -- add the
    # same fixed offset PSMKinematics.forward applies for a fair comparison.
    tip_from_urdf = T_urdf[:3, 3] + T_urdf[:3, :3] @ np.array([0.0, kin.params.l_yaw2ctrlpnt, 0.0])
    tip_from_forward = kin.forward(q)[:3, 3]

    np.testing.assert_allclose(tip_from_urdf, tip_from_forward, atol=1e-6)
