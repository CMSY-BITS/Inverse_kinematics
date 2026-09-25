import os
from glob import glob

from setuptools import find_packages, setup

package_name = "surg_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py") + glob("launch/_common.py")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="JEPA-IK project",
    maintainer_email="p20250419@pilani.bits-pilani.ac.in",
    description=(
        "Gazebo + ROS 2 dVRK PSM simulation for the four benchmark tasks "
        "used to evaluate JEPA-based visual inverse kinematics."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "needle_reach_node = surg_sim.task_nodes.needle_reach_node:main",
            "gauze_retrieve_node = surg_sim.task_nodes.gauze_retrieve_node:main",
            "peg_transfer_node = surg_sim.task_nodes.peg_transfer_node:main",
            "needle_pick_node = surg_sim.task_nodes.needle_pick_node:main",
        ],
    },
)
