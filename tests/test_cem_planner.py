import numpy as np

from models.cem_planner import CEMPlanner


def test_cem_finds_a_known_optimum_on_a_quadratic_bowl():
    # cost(actions) = sum over the horizon of ||a_t - target||^2 — minimized
    # exactly when every action in the sequence equals `target`.
    target = np.array([0.3, -0.2, 0.1])
    action_dim = len(target)

    def cost_fn(action_sequences: np.ndarray) -> np.ndarray:
        # action_sequences: (n_samples, horizon, action_dim)
        return np.sum((action_sequences - target) ** 2, axis=(1, 2))

    planner = CEMPlanner(
        action_dim=action_dim,
        action_low=-1.0,
        action_high=1.0,
        horizon=4,
        n_samples=256,
        n_elite=32,
        n_iters=6,
        init_std=0.5,
        seed=0,
    )
    result = planner.plan(cost_fn)

    np.testing.assert_allclose(result.first_action, target, atol=0.05)
    # best_cost sums squared error over a 4-step x 3-dim action sequence,
    # so even a tight per-component fit (~0.05 atol above) yields a cost
    # well above 0; 0.1 is a generous margin above the observed ~0.03 while
    # still far below an unconverged sequence's cost (e.g. ~0.56 at a=0).
    assert result.best_cost < 0.1


def test_cem_cost_history_is_non_increasing_on_average():
    target = np.array([0.0])

    def cost_fn(action_sequences: np.ndarray) -> np.ndarray:
        return np.sum((action_sequences - target) ** 2, axis=(1, 2))

    planner = CEMPlanner(
        action_dim=1, action_low=-1.0, action_high=1.0, horizon=2, n_samples=128, n_elite=16, n_iters=5, seed=1
    )
    result = planner.plan(cost_fn)
    # elite cost at the final iteration should be no worse than at the first
    assert result.cost_history[-1] <= result.cost_history[0] + 1e-9


def test_cem_respects_action_bounds():
    target = np.array([5.0])  # outside [-1, 1]

    def cost_fn(action_sequences: np.ndarray) -> np.ndarray:
        return np.sum((action_sequences - target) ** 2, axis=(1, 2))

    planner = CEMPlanner(
        action_dim=1, action_low=-1.0, action_high=1.0, horizon=1, n_samples=64, n_elite=8, n_iters=3, seed=2
    )
    result = planner.plan(cost_fn)
    assert -1.0 <= result.first_action[0] <= 1.0
