"""Common ROS 2 node shared by all four benchmark tasks.

Runs the "Perception + Planner + IK/RCM" side of the real-time loop in
docs/evaluation_plan.md §4 as one Python node (the 1 kHz joint-servo loop
itself lives in `ros2_control`/`gz_sim`, outside this node, per the
controller config in `config/psm_controllers.yaml`).

Wiring, at a glance:
    /camera/image_raw  --\\
                           >--[encode]--[plan: CEM or inverse model]--[IK + RCM clamp]--> /position_controller/commands
    /joint_states      --/

Requires a ROS 2 Jazzy workspace (rclpy) and this repo's `kinematics`,
`models`, and `supervisor` packages on the PYTHONPATH — from the repo
root, `pip install -e .` once (see README.md), then colcon-build this
package as usual. It will not import on the cloud scaffolding container.
"""
from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from kinematics import PSMKinematics, clamp_to_rcm
from models.cem_planner import CEMPlanner

PLANNER_RATE_HZ = 10.0  # within the 5-10 Hz CEM planner band from the eval plan
N_JOINTS = 6


class BaseTaskNode(Node):
    """Subclass and set `task_name` (+ override `_load_goal` for
    task-specific goal handling); everything else is shared."""

    task_name = "base_task"

    def __init__(self):
        super().__init__(f"{self.task_name}_node")
        self._bridge = CvBridge()
        self._kin = PSMKinematics()
        self._planner = CEMPlanner(
            action_dim=N_JOINTS,
            action_low=-0.1,   # per-step joint delta limits (rad or m for insertion), conservative default
            action_high=0.1,
        )

        self._latest_image = None
        self._latest_q = np.zeros(N_JOINTS)
        self._goal_xyz = self._load_goal()

        self.create_subscription(Image, "camera/image_raw", self._on_image, QoSPresetProfiles.SENSOR_DATA.value)
        self.create_subscription(JointState, "joint_states", self._on_joint_state, 10)
        self._cmd_pub = self.create_publisher(Float64MultiArray, "position_controller/commands", 10)

        self.create_timer(1.0 / PLANNER_RATE_HZ, self._on_planner_tick)
        self.get_logger().info(f"{self.task_name}: ready, planning at {PLANNER_RATE_HZ} Hz")

    def _load_goal(self) -> np.ndarray:
        """Task-specific target tip position (meters, base frame). Override
        in subclasses; base default is a stand-in until wired to the
        world's task-prop pose (e.g. via a Gazebo model-state service call
        or a `/task/goal_pose` topic published by the world plugin)."""
        return np.array([0.05, 0.0, 0.15])

    def _on_image(self, msg: Image) -> None:
        self._latest_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # TODO: encode with models.jepa_wrapper.VJEPA2Encoder once a
        # checkpoint is available on this machine; the CEM cost below uses
        # ground-truth-ish forward kinematics in the meantime so the
        # control loop is exercisable before the perception model is wired in.

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_q = np.array(msg.position[:N_JOINTS])

    def _on_planner_tick(self) -> None:
        if self._latest_image is None:
            return  # no frame yet

        def cost_fn(action_sequences: np.ndarray) -> np.ndarray:
            # Roll each candidate joint-delta sequence forward through
            # forward kinematics and score by final distance to goal.
            # Swap for the AC-head latent rollout (models/ac_head.py) once
            # the encoder is wired in; this FK-based cost is a
            # perception-free stand-in that still exercises the whole loop.
            costs = np.empty(len(action_sequences))
            for i, seq in enumerate(action_sequences):
                q = self._latest_q.copy()
                for a in seq:
                    q = q + a
                tip = self._kin.forward(q)[:3, 3]
                costs[i] = np.linalg.norm(tip - self._goal_xyz)
            return costs

        result = self._planner.plan(cost_fn)
        q_cmd = self._latest_q + result.first_action

        rcm_result = clamp_to_rcm(self._kin, q_cmd, mode="clamp_outer_joints")
        msg = Float64MultiArray()
        msg.data = rcm_result.q_used.tolist()
        self._cmd_pub.publish(msg)


def spin_task_node(node_cls) -> None:
    rclpy.init()
    node = node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
