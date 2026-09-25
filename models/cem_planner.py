"""Cross-entropy method (CEM) planner — the A0 baseline ("V-JEPA 2-AC + CEM")
and the fallback planner behind every rung of the ablation ladder in the
evaluation plan (A1-A4 add a faster policy but keep CEM as the safety net
when that policy is out of distribution).

Deliberately has no dependency on V-JEPA, torch, or any specific dynamics
model: it optimizes over whatever batched cost function you give it. In
training this is `cost_fn = lambda actions: goal_distance(rollout(latent, actions))`
using the V-JEPA 2-AC predictor; in tests it's a plain quadratic bowl. That
separation is what makes the planner unit-testable without a GPU.

Sized for the eval plan's 100 ms/step budget: N=128, H=3, 3 iterations
(models/README or eval/metrics.py records the measured wall-clock).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CEMResult:
    action_sequence: np.ndarray  # (horizon, action_dim) — the elite mean at convergence
    first_action: np.ndarray     # (action_dim,) — action_sequence[0], what the controller executes
    best_cost: float
    cost_history: list[float] = field(default_factory=list)


class CEMPlanner:
    def __init__(
        self,
        action_dim: int,
        action_low: np.ndarray | float,
        action_high: np.ndarray | float,
        horizon: int = 3,
        n_samples: int = 128,
        n_elite: int = 16,
        n_iters: int = 3,
        init_std: float = 0.3,
        alpha: float = 0.1,
        seed: int | None = None,
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.init_std = init_std
        self.alpha = alpha  # mean/std smoothing between iterations (0 = no memory)
        self.action_low = np.broadcast_to(np.asarray(action_low, dtype=float), (action_dim,)).copy()
        self.action_high = np.broadcast_to(np.asarray(action_high, dtype=float), (action_dim,)).copy()
        self._rng = np.random.default_rng(seed)

        if n_elite > n_samples:
            raise ValueError("n_elite must be <= n_samples")

    def plan(self, cost_fn, init_mean: np.ndarray | None = None) -> CEMResult:
        """cost_fn(action_sequences: (n_samples, horizon, action_dim)) ->
        costs: (n_samples,) float array, lower is better. Called once per
        iteration so a GPU-backed cost_fn can batch the whole population in
        one forward pass.
        """
        mean = (
            np.array(init_mean, dtype=float)
            if init_mean is not None
            else np.zeros((self.horizon, self.action_dim))
        )
        std = np.full((self.horizon, self.action_dim), self.init_std)
        history: list[float] = []
        elites_mean = mean
        best_cost = np.inf

        for _ in range(self.n_iters):
            samples = self._rng.normal(
                loc=mean, scale=std, size=(self.n_samples, self.horizon, self.action_dim)
            )
            samples = np.clip(samples, self.action_low, self.action_high)

            costs = np.asarray(cost_fn(samples), dtype=float)
            if costs.shape != (self.n_samples,):
                raise ValueError(f"cost_fn must return shape ({self.n_samples},), got {costs.shape}")

            elite_idx = np.argsort(costs)[: self.n_elite]
            elites = samples[elite_idx]
            elites_mean = elites.mean(axis=0)
            elites_std = elites.std(axis=0) + 1e-6

            mean = self.alpha * mean + (1 - self.alpha) * elites_mean
            std = self.alpha * std + (1 - self.alpha) * elites_std

            iter_best = float(costs[elite_idx[0]])
            history.append(iter_best)
            best_cost = min(best_cost, iter_best)

        return CEMResult(
            action_sequence=elites_mean,
            first_action=elites_mean[0],
            best_cost=best_cost,
            cost_history=history,
        )
