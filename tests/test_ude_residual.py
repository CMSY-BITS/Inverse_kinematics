import numpy as np

from models._torch_optional import TORCH_AVAILABLE
from models.ude_residual import UDEParams, UDEResidual, cable_hysteresis_step


def test_backlash_holds_within_deadband():
    q = np.array([0.0])
    u = np.array([0.005])  # within default backlash=0.01: cable is slack
    q_next = cable_hysteresis_step(q, u, dt=0.01)
    np.testing.assert_allclose(q_next, q)


def test_moves_toward_command_outside_deadband_but_not_all_the_way():
    q = np.array([0.0])
    u = np.array([0.05])  # well beyond the deadband
    q_next = cable_hysteresis_step(q, u, dt=0.01, tau=0.05, backlash=0.01)
    assert q_next[0] > 0.0
    assert q_next[0] < u[0]  # first-order lag: doesn't jump straight to u


def test_repeated_steps_converge_to_within_backlash_of_command():
    q = np.array([0.0])
    u = np.array([0.05])
    for _ in range(3000):
        q = cable_hysteresis_step(q, u, dt=0.01, tau=0.05, backlash=0.01)
    assert abs(u[0] - q[0]) <= 0.01 + 1e-6


def test_symmetric_for_negative_commands():
    q = np.array([0.0])
    u = np.array([-0.05])
    q_next = cable_hysteresis_step(q, u, dt=0.01, tau=0.05, backlash=0.01)
    assert q_next[0] < 0.0


def test_multi_joint_is_independent_per_joint():
    q = np.zeros(3)
    u = np.array([0.005, 0.05, -0.05])  # joint 0 in-deadband, 1 and 2 move
    q_next = cable_hysteresis_step(q, u, dt=0.01, tau=0.05, backlash=0.01)
    assert q_next[0] == 0.0
    assert q_next[1] > 0.0
    assert q_next[2] < 0.0


def test_ude_residual_construction_never_requires_torch():
    # Unlike ACHead/InverseModel/MMDAdapter, UDEResidual must build even
    # without torch installed (physics-only mode).
    ude = UDEResidual(n_joints=1, params=UDEParams(tau=0.05, backlash=0.01))
    assert ude.n_joints == 1


def test_ude_residual_step_matches_physics_when_no_residual_net():
    ude = UDEResidual(n_joints=1, params=UDEParams(tau=0.05, backlash=0.01))
    q, u = np.array([0.0]), np.array([0.05])
    q_next = ude.step(q, u, dt=0.01)
    expected_physics_only = cable_hysteresis_step(q, u, dt=0.01, tau=0.05, backlash=0.01)
    if TORCH_AVAILABLE:
        # a residual net exists and may perturb the output; just check shape/type
        assert q_next.shape == expected_physics_only.shape
    else:
        np.testing.assert_allclose(q_next, expected_physics_only)
