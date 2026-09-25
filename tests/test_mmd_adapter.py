import numpy as np

from models.mmd_adapter import compute_mmd, median_heuristic_sigma


def test_mmd_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8))
    mmd = compute_mmd(x, x.copy())
    assert mmd < 1e-6


def test_mmd_positive_for_clearly_shifted_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, size=(200, 8))
    y = rng.normal(loc=5.0, size=(200, 8))
    mmd = compute_mmd(x, y)
    assert mmd > 0.1


def test_mmd_increases_with_larger_domain_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, size=(150, 4))
    y_small_shift = rng.normal(loc=0.5, size=(150, 4))
    y_large_shift = rng.normal(loc=5.0, size=(150, 4))
    # fix a shared bandwidth so the comparison isn't confounded by the
    # median heuristic picking a different sigma per pair
    sigma = median_heuristic_sigma(x, y_large_shift)
    mmd_small = compute_mmd(x, y_small_shift, sigma=sigma)
    mmd_large = compute_mmd(x, y_large_shift, sigma=sigma)
    assert mmd_large > mmd_small


def test_median_heuristic_sigma_is_positive():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    y = rng.normal(size=(50, 4))
    sigma = median_heuristic_sigma(x, y)
    assert sigma > 0
