"""Metric definitions from the evaluation plan §2: precision, latency,
safety, supervisor calibration, and the statistics used to compare methods
(bootstrap CIs, Wilcoxon signed-rank vs. the strongest baseline).

Pure numpy/scipy, no simulator or GPU dependency, so every function here is
directly unit-tested against synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def tip_error_mm(pred_xyz_m: np.ndarray, target_xyz_m: np.ndarray) -> np.ndarray:
    """Euclidean tip error in millimeters. Inputs in meters, any leading
    batch shape, last axis = 3."""
    pred = np.asarray(pred_xyz_m, dtype=float)
    target = np.asarray(target_xyz_m, dtype=float)
    return np.linalg.norm(pred - target, axis=-1) * 1000.0


def success_rate(tip_errors_mm: np.ndarray, threshold_mm: float = 3.0) -> float:
    tip_errors_mm = np.asarray(tip_errors_mm, dtype=float)
    return float(np.mean(tip_errors_mm <= threshold_mm))


def orientation_error_deg(pred_R: np.ndarray, target_R: np.ndarray) -> np.ndarray:
    """Geodesic rotation error in degrees between predicted and target 3x3
    rotation matrices (batched: (..., 3, 3))."""
    pred_R = np.asarray(pred_R, dtype=float)
    target_R = np.asarray(target_R, dtype=float)
    rel = np.swapaxes(pred_R, -1, -2) @ target_R
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ---------------------------------------------------------------------------
# Latency / real-time performance
# ---------------------------------------------------------------------------
@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


def latency_percentiles(latencies_ms: np.ndarray) -> LatencyStats:
    latencies_ms = np.asarray(latencies_ms, dtype=float)
    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])
    return LatencyStats(p50_ms=float(p50), p95_ms=float(p95), p99_ms=float(p99), mean_ms=float(latencies_ms.mean()))


def deadline_miss_rate(latencies_ms: np.ndarray, deadline_ms: float) -> float:
    latencies_ms = np.asarray(latencies_ms, dtype=float)
    return float(np.mean(latencies_ms > deadline_ms))


def achieved_control_rate_hz(step_timestamps_s: np.ndarray) -> float:
    """Given monotonic timestamps of successive control-loop ticks, the
    achieved rate — not the nominal target rate."""
    step_timestamps_s = np.asarray(step_timestamps_s, dtype=float)
    if len(step_timestamps_s) < 2:
        return float("nan")
    dt = np.diff(step_timestamps_s)
    return float(1.0 / np.mean(dt))


def real_time_factor(sim_time_elapsed_s: float, wall_time_elapsed_s: float) -> float:
    """>1 means the sim ran faster than real time."""
    if wall_time_elapsed_s <= 0:
        return float("nan")
    return sim_time_elapsed_s / wall_time_elapsed_s


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def rcm_violation_rate(rcm_deviations_m: np.ndarray, tol_m: float = 5e-4) -> float:
    rcm_deviations_m = np.asarray(rcm_deviations_m, dtype=float)
    return float(np.mean(rcm_deviations_m > tol_m))


# ---------------------------------------------------------------------------
# Supervisor (Jev) calibration
# ---------------------------------------------------------------------------
def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bin predictions by confidence, compare each bin's
    mean confidence to its accuracy, weight by bin size."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Statistics: bootstrap CIs and paired significance testing
# ---------------------------------------------------------------------------
@dataclass
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    alpha: float


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapCI:
    """Percentile bootstrap confidence interval for `statistic` (default:
    the mean) of `data`. Used for every metric in the results tables —
    3 seeds x 100 episodes per task, per the evaluation plan's statistics
    protocol."""
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(point_estimate=float(statistic(data)), ci_low=float(lo), ci_high=float(hi), alpha=alpha)


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    significant_at_05: bool


def wilcoxon_paired_test(a: np.ndarray, b: np.ndarray) -> WilcoxonResult:
    """Wilcoxon signed-rank test between paired per-episode results of
    method `a` (e.g. an ablation rung) and method `b` (the strongest
    baseline), matched by seed/episode index. Non-parametric: doesn't
    assume normally distributed errors, which per-episode success metrics
    generally aren't."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    result = stats.wilcoxon(a, b)
    return WilcoxonResult(
        statistic=float(result.statistic), p_value=float(result.pvalue), significant_at_05=result.pvalue < 0.05
    )
