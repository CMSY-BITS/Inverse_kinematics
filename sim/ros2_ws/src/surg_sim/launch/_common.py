"""Shared launch-description builder for the four task launch files.
Not a launch file itself — imported by needle_reach.launch.py etc. so the
Gazeb+ros2_control wiring is written once.
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


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

    # Bridges Gazebo's /clock to ROS 2 so nodes using sim time stay in sync
    # with the 1 kHz physics step (config/psm_controllers.yaml).
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
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
            clock_bridge,
            joint_state_broadcaster_spawner,
            position_controller_spawner,
            task_node,
        ]
    )
