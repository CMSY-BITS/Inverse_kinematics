"""Shared launch-description builder for the four task launch files.
Not a launch file itself — imported by needle_reach.launch.py etc. so the
Gazeb+ros2_control wiring is written once.
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from blender.export_sdf import CAMERA_IMAGE_TOPIC
from surg_sim.psm_urdf import build_psm_urdf


def generate_task_launch_description(task_name: str, node_executable: str) -> LaunchDescription:
    pkg_share = get_package_share_directory("surg_sim")
    world_path = f"{pkg_share}/worlds/{task_name}.sdf"
    controllers_yaml = f"{pkg_share}/config/psm_controllers.yaml"

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": f"-r {world_path}"}.items(),
    )

    # Bridges Gazebo's /clock (so sim-time nodes stay in sync with the
    # 1 kHz physics step, config/psm_controllers.yaml) and the endoscope
    # camera sensor's image topic (blender/export_sdf.py's
    # _add_camera_sensor) one-way into ROS 2. CAMERA_IMAGE_TOPIC is used
    # on both sides deliberately -- it's the same constant the sensor's
    # <topic> was built from, so this bridge can't silently point at the
    # wrong GZ topic.
    gz_ros_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            f"{CAMERA_IMAGE_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image",
        ],
        output="screen",
    )

    # gz_ros2_control's controller_manager (inside the Gazebo plugin, see
    # blender/export_sdf.py's _add_ros2_control_block) does NOT read the
    # <ros2_control> block embedded in the Gazebo SDF on its own -- it
    # blocks forever waiting for a URDF-format robot_description on this
    # topic. robot_state_publisher is what actually publishes it, built
    # from surg_sim.psm_urdf (kept in sync with the SDF's version via the
    # same forward-kinematics transforms, not duplicated by hand).
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": build_psm_urdf(), "use_sim_time": True}],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--param-file", controllers_yaml],
        output="screen",
    )

    position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["position_controller", "--param-file", controllers_yaml],
        output="screen",
    )

    task_node = Node(
        package="surg_sim",
        executable=node_executable,
        name=f"{task_name}_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            gz_sim,
            gz_ros_bridge,
            robot_state_publisher,
            joint_state_broadcaster_spawner,
            position_controller_spawner,
            task_node,
        ]
    )
