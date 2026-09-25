import numpy as np
import pytest

from kinematics.psm_kinematics import PSMKinematics
from kinematics.rcm import RCMViolation, clamp_to_rcm, rcm_deviation


@pytest.fixture
def kin():
    return PSMKinematics()


def test_zero_deviation_at_default_pose(kin):
    # The DH chain places the RCM at the base-frame origin by
    # construction, so any q with q1=q2=0 has zero shaft deviation there.
    q = np.array([0.0, 0.0, 0.1, 0.2, 0.1, -0.1])
    dev = rcm_deviation(kin, q)
    assert dev < 1e-9


def test_nonzero_outer_joints_still_pass_through_origin(kin):
    # q1/q2 rotate the shaft's *direction* but the shaft still passes
    # through the RCM origin — deviation should stay ~0 for any q1,q2.
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = np.array([rng.uniform(-1, 1), rng.uniform(-0.8, 0.8), 0.1, 0.0, 0.0, 0.0])
        assert rcm_deviation(kin, q) < 1e-9


def test_clamp_corrects_an_offset_rcm_point(kin):
    q = np.array([0.0, 0.0, 0.1, 0.2, 0.1, -0.1])
    # Pretend the "true" RCM (trocar) is 5 mm away from where the model
    # thinks it is — a registration-error scenario. q1 (yaw) and q2 (pitch)
    # only ever swing the shaft direction within the base frame's x-z
    # plane (never off it — see psm_kinematics.py:shaft_line), so a
    # reachable offset needs its deviation off the x-z plane's own line
    # through the current direction, not off the y-axis (which the shaft
    # can never point along at all).
    offset_rcm = np.array([0.0, 0.0, 0.005])
    dev_before = rcm_deviation(kin, q, rcm_point=offset_rcm)
    assert dev_before > 1e-4

    result = clamp_to_rcm(kin, q, rcm_point=offset_rcm, tol_m=1e-4)
    assert result.within_tolerance
    assert result.deviation_m < 1e-3  # solver tolerance is loose (line search), not exact IK


def test_clamp_leaves_insertion_and_wrist_untouched(kin):
    q = np.array([0.3, -0.2, 0.12, 0.4, -0.2, 0.15])
    offset_rcm = np.array([0.002, -0.001, 0.001])
    result = clamp_to_rcm(kin, q, rcm_point=offset_rcm, tol_m=1e-4)
    np.testing.assert_allclose(result.q_used[2:], q[2:])


def test_raise_mode_flags_a_violation(kin):
    q = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    offset_rcm = np.array([0.0, 0.02, 0.0])  # off-axis, well beyond any reasonable tolerance
    with pytest.raises(RCMViolation):
        clamp_to_rcm(kin, q, rcm_point=offset_rcm, tol_m=5e-4, mode="raise")


def test_raise_mode_does_not_raise_when_within_tolerance(kin):
    q = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    result = clamp_to_rcm(kin, q, tol_m=5e-4, mode="raise")
    assert result.within_tolerance
