"""Forward and inverse kinematics for the dVRK Patient Side Manipulator (PSM).

The PSM is a remote-center-of-motion (RCM) mechanism: joints 1-3 position the
tool shaft through a fixed trocar point, joints 4-6 orient the wrist, and
joint 7 (not modeled here, see `jaw_angle`) opens/closes the gripper.

The DH table below is the one commonly used across dVRK research code
(cisst/SAW, dvrk_python, AMBF). The link-length constants
(`PSMParams` defaults) are representative da Vinci Si/S PSM values in
meters, NOT calibrated numbers for any specific unit. Replace them with the
values from your robot's `console-*.json` / URDF before trusting absolute
millimeter accuracy; relative behavior (IK convergence, RCM constraint) is
correct regardless.

This module is the "oracle" IK baseline referenced in the evaluation plan
(A0 row in the baseline table, "Damped least squares IK / PBVS") and is also
what the real-time "IK + RCM filter" control-loop node wraps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PSMParams:
    """Link lengths for the PSM DH chain, in meters."""

    l_rcc: float = 0.4318   # RCM-to-shaft-origin offset (outer pitch axis to RCM)
    l_tool: float = 0.4162  # shaft length, RCM to wrist pitch axis
    l_pitch2yaw: float = 0.0091   # wrist pitch axis to wrist yaw axis
    l_yaw2ctrlpnt: float = 0.0102  # wrist yaw axis to tool-tip control point


# Joint limits (rad for revolute, meters for the prismatic insertion joint 3),
# representative of the PSM's mechanical range of motion.
JOINT_LIMITS = np.array(
    [
        [-1.605, 1.605],   # q1 outer yaw
        [-0.93, 0.93],     # q2 outer pitch
        [0.0, 0.235],      # q3 insertion (relative to RCM; 0 = tip at RCM)
        [-4.36, 4.36],     # q4 tool roll
        [-1.57, 1.57],     # q5 wrist pitch
        [-1.57, 1.57],     # q6 wrist yaw
    ]
)


def _dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class PSMKinematics:
    """Forward kinematics + damped-least-squares (DLS) inverse kinematics."""

    def __init__(self, params: PSMParams | None = None):
        self.params = params or PSMParams()

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def forward(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 tool-tip pose in the RCM (base) frame for joints
        q = [q1..q6] (outer yaw, outer pitch, insertion, roll, wrist pitch,
        wrist yaw)."""
        q = np.asarray(q, dtype=float)
        assert q.shape == (6,), f"expected 6 joints, got {q.shape}"
        p = self.params
        q1, q2, q3, q4, q5, q6 = q

        dh = [
            (np.pi / 2, 0.0, 0.0, q1 + np.pi / 2),
            (-np.pi / 2, 0.0, 0.0, q2 - np.pi / 2),
            (np.pi / 2, 0.0, q3 - p.l_rcc, 0.0),
            (0.0, 0.0, p.l_tool, q4),
            (-np.pi / 2, 0.0, 0.0, q5 - np.pi / 2),
            (-np.pi / 2, p.l_pitch2yaw, 0.0, q6 - np.pi / 2),
        ]
        T = np.eye(4)
        for alpha, a, d, theta in dh:
            T = T @ _dh_transform(alpha, a, d, theta)

        # Offset from wrist yaw axis to the tool-tip control point, along
        # the wrist's local z after the yaw rotation already applied above.
        tip_offset = np.array([0.0, p.l_yaw2ctrlpnt, 0.0, 1.0])
        tip = T @ tip_offset
        T_tip = T.copy()
        T_tip[:3, 3] = tip[:3]
        return T_tip

    def shaft_line(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (point_on_shaft, direction) for the rigid shaft segment
        between the RCM and the wrist, used for the RCM-deviation check.
        The shaft passes through the origin (RCM) by construction of the
        DH chain (joint 3 translates along it), so the origin is always a
        point on the ideal line; `direction` is the shaft's unit axis in
        the base frame for the given q1, q2.

        Note: `direction` depends only on q1 here, not q2. This isn't a
        bug specific to this table — it's the general DH identity that a
        joint's own theta can never change its own frame's resulting
        z-axis (only the fixed twist `alpha` does; theta only spins
        around that same axis), and q2 is the last joint before the frame
        whose z-axis this method reports. `kinematics/rcm.py`'s
        `clamp_to_rcm` relies on this: it only actually needs to search q1.
        """
        q = np.asarray(q, dtype=float)
        p = self.params
        q1, q2 = q[0], q[1]
        T = _dh_transform(np.pi / 2, 0.0, 0.0, q1 + np.pi / 2) @ _dh_transform(
            -np.pi / 2, 0.0, 0.0, q2 - np.pi / 2
        )
        direction = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        origin = np.zeros(3)  # RCM is the base-frame origin by construction
        return origin, direction

    # ------------------------------------------------------------------
    # Numerical Jacobian (finite differences) — simple and robust; swap for
    # an analytic Jacobian later if profiling shows it's the bottleneck.
    # ------------------------------------------------------------------
    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """3x6 position Jacobian d(tip_xyz)/d(q)."""
        q = np.asarray(q, dtype=float)
        p0 = self.forward(q)[:3, 3]
        J = np.zeros((3, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            p1 = self.forward(q + dq)[:3, 3]
            J[:, i] = (p1 - p0) / eps
        return J

    # ------------------------------------------------------------------
    # Inverse kinematics (position-only damped least squares)
    # ------------------------------------------------------------------
    def inverse(
        self,
        target_xyz: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iters: int = 100,
        tol_m: float = 1e-4,
        damping: float = 1e-2,
        step_clip: float = 0.2,
    ) -> tuple[np.ndarray, bool, int]:
        """Damped least-squares IK to a target tip position (meters, base
        frame). Returns (q, converged, n_iters).

        This is the "oracle" baseline: it has no perception in the loop and
        assumes the target position is already known exactly, unlike the
        JEPA-based policies which must infer it from images.
        """
        target_xyz = np.asarray(target_xyz, dtype=float)
        q = np.array(q_init, dtype=float) if q_init is not None else np.zeros(6)
        q[2] = max(q[2], 0.01)  # keep insertion off the RCM singularity

        for it in range(max_iters):
            pos = self.forward(q)[:3, 3]
            err = target_xyz - pos
            if np.linalg.norm(err) < tol_m:
                return self._clip_limits(q), True, it

            J = self.jacobian(q)
            JJt = J @ J.T
            damped = JJt + (damping**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(damped, err)
            dq = np.clip(dq, -step_clip, step_clip)
            q = self._clip_limits(q + dq)

        pos = self.forward(q)[:3, 3]
        converged = np.linalg.norm(target_xyz - pos) < tol_m
        return q, converged, max_iters

    @staticmethod
    def _clip_limits(q: np.ndarray) -> np.ndarray:
        return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
