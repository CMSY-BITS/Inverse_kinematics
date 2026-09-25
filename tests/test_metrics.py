import numpy as np

from eval import metrics as M


def test_tip_error_mm_basic():
    pred = np.array([0.0, 0.0, 0.0])
    target = np.array([0.001, 0.0, 0.0])  # 1 mm away
    err = M.tip_error_mm(pred, target)
    np.testing.assert_allclose(err, 1.0)


def test_tip_error_mm_is_batched():
    pred = np.zeros((5, 3))
    target = np.zeros((5, 3))
    target[:, 0] = 0.003  # 3 mm
    errs = M.tip_error_mm(pred, target)
    assert errs.shape == (5,)
    np.testing.assert_allclose(errs, 3.0)


def test_success_rate_threshold():
    errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert M.success_rate(errs, threshold_mm=3.0) == 0.6  # 3 of 5 <= 3mm


def test_orientation_error_deg_zero_for_identical_rotations():
    R = np.eye(3)
    err = M.orientation_error_deg(R, R)
    assert err < 1e-6


def test_orientation_error_deg_90_for_perpendicular_axes():
    R_identity = np.eye(3)
    # 90 degree rotation about z
    R_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    err = M.orientation_error_deg(R_identity, R_90)
    np.testing.assert_allclose(err, 90.0, atol=1e-6)


def test_latency_percentiles_monotonic():
    latencies = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    stats = M.latency_percentiles(latencies)
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms


def test_deadline_miss_rate():
    latencies = np.array([50, 60, 150, 200, 40])
    rate = M.deadline_miss_rate(latencies, deadline_ms=100)
    assert rate == 0.4  # 2 of 5 exceed 100ms


def test_achieved_control_rate_hz():
    # perfectly regular 10ms ticks -> 100 Hz
    ts = np.arange(0, 1.0, 0.01)
    rate = M.achieved_control_rate_hz(ts)
    np.testing.assert_allclose(rate, 100.0, rtol=1e-6)


def test_real_time_factor():
    assert M.real_time_factor(sim_time_elapsed_s=10.0, wall_time_elapsed_s=5.0) == 2.0


def test_rcm_violation_rate():
    devs = np.array([0.0001, 0.0002, 0.001, 0.002, 0.0003])
    rate = M.rcm_violation_rate(devs, tol_m=5e-4)
    assert rate == 0.4  # 2 of 5 exceed 0.5mm


def test_expected_calibration_error_perfect_calibration_is_zero():
    rng = np.random.default_rng(0)
    n = 5000
    confidences = rng.uniform(0, 1, size=n)
    correct = rng.uniform(0, 1, size=n) < confidences  # calibrated by construction
    ece = M.expected_calibration_error(confidences, correct, n_bins=10)
    assert ece < 0.03


def test_expected_calibration_error_overconfident_is_high():
    n = 1000
    confidences = np.full(n, 0.95)
    correct = np.zeros(n, dtype=bool)  # always wrong despite high confidence
    ece = M.expected_calibration_error(confidences, correct)
    assert ece > 0.9


def test_bootstrap_ci_covers_true_mean_of_normal_data():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=1.0, size=200)
    ci = M.bootstrap_ci(data, seed=1, n_boot=2000)
    assert ci.ci_low < 5.0 < ci.ci_high
    assert ci.ci_low < ci.point_estimate < ci.ci_high


def test_wilcoxon_paired_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.1, size=50)   # method A: small errors
    b = rng.normal(loc=2.0, scale=0.1, size=50)   # method B: much larger errors
    result = M.wilcoxon_paired_test(a, b)
    assert result.significant_at_05
    assert result.p_value < 0.05


def test_wilcoxon_paired_no_difference_for_identical_arrays_shifted_by_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(size=30)
    b = a + rng.normal(scale=1e-9, size=30)  # effectively identical
    # Wilcoxon requires at least one non-zero difference; this should not
    # raise and should report a high p-value (no significant difference).
    result = M.wilcoxon_paired_test(a, b)
    assert result.p_value > 0.05
