import numpy as np
import pytest

from kinematics.psm_kinematics import JOINT_LIMITS, PSMKinematics


@pytest.fixture
def kin():
    return PSMKinematics()


def test_forward_at_zero_is_finite(kin):
    T = kin.forward(np.zeros(6))
    assert T.shape == (4, 4)
    assert np.all(np.isfinite(T))
    # bottom row of a homogeneous transform is always [0,0,0,1]
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_forward_rotation_is_orthonormal(kin):
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        R = kin.forward(q)[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_insertion_moves_tip_further_from_rcm(kin):
    q_low = np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    q_high = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0])
    d_low = np.linalg.norm(kin.forward(q_low)[:3, 3])
    d_high = np.linalg.norm(kin.forward(q_high)[:3, 3])
    assert d_high > d_low


def test_inverse_converges_to_a_reachable_target(kin):
    rng = np.random.default_rng(1)
    q_true = rng.uniform(
        np.maximum(JOINT_LIMITS[:, 0], -0.5),
        np.minimum(JOINT_LIMITS[:, 1], 0.5),
    )
    q_true[2] = abs(q_true[2]) + 0.05  # keep insertion well off its lower limit
    target = kin.forward(q_true)[:3, 3]

    q_sol, converged, n_iters = kin.inverse(target, q_init=np.zeros(6))

    assert converged, f"IK failed to converge in {n_iters} iterations"
    tip_err_mm = np.linalg.norm(kin.forward(q_sol)[:3, 3] - target) * 1000.0
    assert tip_err_mm < 0.5


def test_inverse_respects_joint_limits(kin):
    # A target far outside the workspace should leave the solver at the
    # boundary of the joint limits rather than diverging.
    far_target = np.array([10.0, 10.0, 10.0])
    q_sol, _, _ = kin.inverse(far_target, max_iters=50)
    assert np.all(q_sol >= JOINT_LIMITS[:, 0] - 1e-9)
    assert np.all(q_sol <= JOINT_LIMITS[:, 1] + 1e-9)


def test_jacobian_matches_finite_difference_of_forward(kin):
    q = np.array([0.1, -0.2, 0.08, 0.3, -0.15, 0.25])
    J = kin.jacobian(q)
    eps = 1e-5
    J_check = np.zeros((3, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        p1 = kin.forward(q + dq)[:3, 3]
        p0 = kin.forward(q - dq)[:3, 3]
        J_check[:, i] = (p1 - p0) / (2 * eps)
    np.testing.assert_allclose(J, J_check, atol=1e-4)
