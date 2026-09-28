"""export_sdf.py's mesh-export functions need Blender (bpy), but the
gz_ros2_control XML block it builds is pure ElementTree manipulation and
is fully tested here without it — this is exactly the part that was wrong
twice in a row (wrong plugin filename, then wrong plugin class name, then
an unresolved `$(find pkg)` path) before ever reaching the user's Gazebo,
so it's worth pinning down precisely.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from blender.export_sdf import DEFAULT_CONTROLLERS_YAML, PSM_LINKS, _add_ros2_control_block


def test_default_controllers_yaml_resolves_to_a_real_file():
    # This is the exact bug that shipped once already: a placeholder like
    # "$(find surg_sim)/..." that gz_ros2_control never resolves itself.
    # The default must be a real, existing, absolute path.
    path = Path(DEFAULT_CONTROLLERS_YAML)
    assert path.is_absolute()
    assert path.is_file(), f"{path} does not exist"
    assert path.name == "psm_controllers.yaml"


def test_plugin_element_has_the_real_class_name_confirmed_from_gazebos_own_error_output():
    model = ET.Element("model", name="psm")
    _add_ros2_control_block(model)

    plugin = model.find("plugin")
    assert plugin is not None
    assert plugin.get("filename") == "gz_ros2_control-system"
    # This exact string is what Gazebo's own "Detected Plugins" diagnostic
    # reported when the wrong name was used -- not a guess.
    assert plugin.get("name") == "gz_ros2_control::GazeboSimROS2ControlPlugin"


def test_plugin_parameters_element_is_not_an_unresolved_substitution():
    model = ET.Element("model", name="psm")
    _add_ros2_control_block(model, controllers_yaml="/some/absolute/path.yaml")

    params = model.find("plugin/parameters")
    assert params is not None
    assert params.text == "/some/absolute/path.yaml"
    assert "$(find" not in params.text  # gz_ros2_control does not resolve this itself


def test_ros2_control_block_has_one_joint_per_non_base_psm_link():
    model = ET.Element("model", name="psm")
    _add_ros2_control_block(model)

    ros2_control = model.find("ros2_control")
    assert ros2_control is not None
    assert ros2_control.get("type") == "system"
    assert ros2_control.find("hardware/plugin").text == "gz_ros2_control/GazeboSimSystem"

    expected_joint_names = [j[2] for j in PSM_LINKS if j[2] is not None]
    actual_joint_names = [j.get("name") for j in ros2_control.findall("joint")]
    assert actual_joint_names == expected_joint_names


def test_each_joint_has_a_position_command_and_position_velocity_state_interfaces():
    model = ET.Element("model", name="psm")
    _add_ros2_control_block(model)

    for joint in model.findall("ros2_control/joint"):
        command_interfaces = [c.get("name") for c in joint.findall("command_interface")]
        state_interfaces = [s.get("name") for s in joint.findall("state_interface")]
        assert command_interfaces == ["position"]
        assert state_interfaces == ["position", "velocity"]


def test_psm_links_axis_is_uniformly_local_z():
    # Regression test for a real latent bug: an earlier version of this
    # table had varying per-joint axis values (e.g. (1,0,0), (0,1,0)),
    # which is inconsistent with the DH convention psm_kinematics.py uses
    # -- see PSM_LINKS' own comment and psm_urdf.py's docstring for the
    # proof. Every joint's axis must be local z.
    for entry in PSM_LINKS:
        axis = entry[4]
        if axis is not None:
            assert axis == (0, 0, 1), f"{entry[0]} has axis {axis}, expected (0, 0, 1)"


def test_custom_links_list_is_respected():
    model = ET.Element("model", name="toy")
    toy_links = [
        ("base", "Base", None, None, None),
        ("arm", "Arm", "arm_joint", "revolute", (0, 0, 1)),
    ]
    _add_ros2_control_block(model, links=toy_links, controllers_yaml="/x.yaml")

    joint_names = [j.get("name") for j in model.findall("ros2_control/joint")]
    assert joint_names == ["arm_joint"]
