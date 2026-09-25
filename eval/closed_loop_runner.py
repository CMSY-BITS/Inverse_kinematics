"""Generic closed-loop episode runner.

Deliberately knows nothing about Gazebo, ROS 2, or V-JEPA: it drives any
`env` that exposes `reset() -> obs` and `step(action) -> (obs, done, info)`
against any `policy` that exposes `act(obs) -> action`, and logs the
metrics from `eval/metrics.py`. The real Gazebo dVRK env
(`sim/ros2_ws/src/surg_sim`) and each ablation-ladder policy
(`models/cem_planner.py` etc.) both implement these two small interfaces,
so this runner is what `eval/report.py`-style scripts call for every row of
the results tables, and it's what `tests/test_closed_loop_runner.py`
exercises with a toy env that needs neither ROS 2 nor a GPU.

`info` from `env.step` may carry `tip_xyz_m`, `target_xyz_m`, and
`rcm_deviation_m` — whichever of those a given env provides get folded into
the episode result; a minimal env that provides none of them still runs,
just without those metrics populated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from . import metrics as M


class Env(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> tuple[Any, bool, dict]: ...


class Policy(Protocol):
    def act(self, obs: Any) -> Any: ...


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    final_tip_error_mm: float | None
    rcm_violations: int
    step_latencies_ms: list = field(default_factory=list)
    wall_time_s: float = 0.0
    sim_time_s: float | None = None


def run_episode(
    env: Env,
    policy: Policy,
    max_steps: int = 500,
    success_threshold_mm: float = 3.0,
    rcm_tol_m: float = 5e-4,
    jev_gate=None,
) -> EpisodeResult:
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    step_latencies_ms: list[float] = []
    last_tip_error_mm: float | None = None
    rcm_violations = 0
    wall_start = time.monotonic()
    step = -1  # so `steps=step+1` below is well-defined even if max_steps == 0

    for step in range(max_steps):
        t0 = time.monotonic()
        action = policy.act(obs)
        step_latencies_ms.append((time.monotonic() - t0) * 1000.0)

        obs, done, info = env.step(action)

        if "tip_xyz_m" in info and "target_xyz_m" in info:
            last_tip_error_mm = float(M.tip_error_mm(np.asarray(info["tip_xyz_m"]), np.asarray(info["target_xyz_m"])))
        if info.get("rcm_deviation_m") is not None and info["rcm_deviation_m"] > rcm_tol_m:
            rcm_violations += 1
        if jev_gate is not None:
            jev_gate.submit_plan(
                {
                    "planned_action": np.asarray(action).tolist(),
                    "rcm_deviation_m": info.get("rcm_deviation_m"),
                    "episode_step": step,
                }
            )

        if done:
            break

    wall_time_s = time.monotonic() - wall_start
    success = last_tip_error_mm is not None and last_tip_error_mm <= success_threshold_mm
    return EpisodeResult(
        success=success,
        steps=step + 1,
        final_tip_error_mm=last_tip_error_mm,
        rcm_violations=rcm_violations,
        step_latencies_ms=step_latencies_ms,
        wall_time_s=wall_time_s,
    )


def run_eval(
    env_factory,
    policy: Policy,
    n_episodes: int = 100,
    seeds: list[int] | None = None,
    **episode_kwargs,
) -> list[EpisodeResult]:
    """`env_factory(seed) -> Env`, called once per episode so each gets a
    fresh, seeded environment instance (e.g. a new Gazebo world reset with
    a different needle/gauze pose)."""
    seeds = seeds if seeds is not None else list(range(n_episodes))
    results = []
    for seed in seeds:
        env = env_factory(seed)
        results.append(run_episode(env, policy, **episode_kwargs))
    return results


def aggregate(results: list[EpisodeResult]) -> dict:
    """Roll a list of `EpisodeResult` into the summary numbers that go in a
    results table row: success rate + bootstrap CI, latency percentiles,
    RCM violation rate."""
    successes = np.array([r.success for r in results], dtype=float)
    all_latencies = np.concatenate([np.asarray(r.step_latencies_ms) for r in results if r.step_latencies_ms])
    rcm_rate = np.mean([r.rcm_violations > 0 for r in results])
    return {
        "n_episodes": len(results),
        "success_rate": M.bootstrap_ci(successes),
        "latency": M.latency_percentiles(all_latencies) if len(all_latencies) else None,
        "rcm_violation_episode_rate": float(rcm_rate),
    }
