"""Remote-center-of-motion (RCM) safety constraint.

On the physical dVRK the RCM is enforced mechanically: the shaft passes
through the trocar and the mechanism has no way to move it. In simulation
(and in any learned policy that doesn't know that) nothing stops a commanded
pose from dragging the shaft sideways through the abdominal wall point, so
this module is the software equivalent of the mechanical constraint. It is
the "IK + RCM filter" node in the real-time loop (eval plan §4): every
commanded joint vector passes through here before being sent to the
controller, budgeted at <= 2 ms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .psm_kinematics import PSMKinematics


class RCMViolation(RuntimeError):
    """Raised by `clamp_to_rcm` in `mode="raise"` when deviation exceeds
    tolerance and cannot be corrected within the joint limits."""


@dataclass
class RCMCheckResult:
    deviation_m: float
    within_tolerance: bool
    q_used: np.ndarray


def rcm_deviation(kin: PSMKinematics, q: np.ndarray, rcm_point: np.ndarray | None = None) -> float:
    """Perpendicular distance (meters) from the shaft's line of action to
    the fixed RCM/trocar point. `rcm_point` defaults to the kinematic
    origin, which is where the DH chain places the RCM; pass an explicit
    point if your world frame's RCM has been offset (e.g. by a registration
    error you want to test robustness against).
    """
    origin, direction = kin.shaft_line(q)
    p = rcm_point if rcm_point is not None else np.zeros(3)
    w = p - origin
    proj = np.dot(w, direction) * direction
    perp = w - proj
    return float(np.linalg.norm(perp))


def clamp_to_rcm(
    kin: PSMKinematics,
    q: np.ndarray,
    rcm_point: np.ndarray | None = None,
    tol_m: float = 5e-4,
    mode: str = "clamp_outer_joints",
) -> RCMCheckResult:
    """Check (and optionally correct) a commanded joint vector against the
    RCM constraint.

    mode:
      - "clamp_outer_joints": if deviation exceeds `tol_m`, re-solve q1/q2
        (outer yaw/pitch, the two joints that set the shaft's line) via a
        1-D line search so the shaft again passes through `rcm_point`,
        leaving q3-q6 (insertion, roll, wrist) untouched. This is the
        real-time-safe path used in the control loop.
      - "raise": don't correct, raise `RCMViolation` if out of tolerance.
        Use this in the closed-loop evaluator so violations are counted as
        failures rather than silently fixed.
    """
    q = np.array(q, dtype=float)
    dev = rcm_deviation(kin, q, rcm_point)
    if dev <= tol_m:
        return RCMCheckResult(deviation_m=dev, within_tolerance=True, q_used=q)

    if mode == "raise":
        raise RCMViolation(f"RCM deviation {dev * 1000:.3f} mm exceeds tolerance {tol_m * 1000:.3f} mm")

    if mode != "clamp_outer_joints":
        raise ValueError(f"unknown mode: {mode}")

    # Re-solve q1/q2 with damped least squares against the 2-vector
    # perpendicular-residual, holding q3-q6 fixed. In THIS DH chain q2
    # (outer pitch) turns out to have zero effect on `direction`: by the
    # standard DH identity, a joint's own theta can never change its own
    # frame's resulting z-axis (only the fixed twist `alpha` sets that;
    # theta only spins around that same axis), and q2 is the last joint
    # before the frame whose z-axis `shaft_line` reports. So only q1
    # actually moves `direction` here — q2 is carried along in the solve
    # below for generality (e.g. if `PSMParams`/the DH table are extended
    # later) but its Jacobian column is exactly zero and it stays fixed in
    # practice. See kinematics/psm_kinematics.py's `shaft_line` docstring.
    p = rcm_point if rcm_point is not None else np.zeros(3)
    q_fixed = q.copy()

    def perp_error(q12: np.ndarray) -> np.ndarray:
        qq = q_fixed.copy()
        qq[0], qq[1] = q12
        origin, direction = kin.shaft_line(qq)
        w = p - origin
        proj = np.dot(w, direction) * direction
        return w - proj  # residual we want to drive to zero, NOT target-minus-current

    # The cost ||w - (w.d)d||^2 = |w|^2 - (w.d)^2 has a *zero* gradient
    # w.r.t. the shaft direction d exactly when w is already perpendicular
    # to d (a genuine saddle point of this residual, not a numerical
    # artifact: -(w.d) is the gradient's magnitude, and it's 0 there by
    # construction) — in fact q1=0 is then a local *maximum* of the cost
    # along q1, so plain gradient descent started exactly there has
    # nowhere to go. A small fixed jitter breaks the exact symmetry so the
    # solve below has a real downhill direction to follow.
    q12 = q[:2].copy() + np.array([5e-2, 0.0])
    for _ in range(20):
        err = perp_error(q12)
        if np.linalg.norm(err) <= tol_m:
            break
        eps = 1e-6
        J = np.zeros((3, 2))
        for i in range(2):
            dq = np.zeros(2)
            dq[i] = eps
            J[:, i] = (perp_error(q12 + dq) - err) / eps
        JJt = J @ J.T + (1e-4**2) * np.eye(3)
        # `err` is the residual itself (not target-minus-current, unlike
        # PSMKinematics.inverse's `err`), so the Gauss-Newton descent step
        # needs the minus sign here: solve J dq = -err, not J dq = err.
        dq12 = -J.T @ np.linalg.solve(JJt, err)
        q12 = q12 + np.clip(dq12, -0.2, 0.2)

    q_corrected = q_fixed.copy()
    q_corrected[0], q_corrected[1] = q12
    q_corrected = kin._clip_limits(q_corrected)
    dev_after = rcm_deviation(kin, q_corrected, rcm_point)
    return RCMCheckResult(
        deviation_m=dev_after, within_tolerance=dev_after <= tol_m, q_used=q_corrected
    )
