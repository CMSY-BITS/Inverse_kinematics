"""Exercises `eval/closed_loop_runner.py` against a toy point-mass env, so
the runner's success/RCM-violation/aggregation logic is tested without
Gazebo, ROS 2, or a GPU."""
import numpy as np

from eval.closed_loop_runner import aggregate, run_episode, run_eval


class ToyEnv:
    """A point in 3-space chasing a fixed target; stands in for the real
    Gazebo dVRK env, which exposes the same `reset`/`step` shape."""

    def __init__(self, seed: int = 0, target: np.ndarray | None = None):
        self.rng = np.random.default_rng(seed)
        self.target = target if target is not None else np.array([0.01, 0.0, 0.0])
        self.pos = np.zeros(3)
        self.t = 0
        self.rcm_deviation_m = 0.0

    def reset(self):
        self.pos = np.zeros(3)
        self.t = 0
        return self.pos.copy()

    def step(self, action):
        self.pos = self.pos + np.asarray(action)
        self.t += 1
        tip_err_mm = float(np.linalg.norm(self.pos - self.target) * 1000.0)
        done = tip_err_mm <= 3.0 or self.t >= 50
        info = {
            "tip_xyz_m": self.pos.copy(),
            "target_xyz_m": self.target.copy(),
            "rcm_deviation_m": self.rcm_deviation_m,
        }
        return self.pos.copy(), done, info


class ToyPolicy:
    def __init__(self, target: np.ndarray):
        self.target = target

    def act(self, obs):
        direction = self.target - obs
        return np.clip(direction, -0.002, 0.002)  # 2 mm max step per axis


def test_run_episode_reaches_success():
    target = np.array([0.01, 0.0, 0.0])
    result = run_episode(ToyEnv(target=target), ToyPolicy(target), max_steps=50)
    assert result.success
    assert result.final_tip_error_mm <= 3.0
    assert result.steps <= 50
    assert len(result.step_latencies_ms) == result.steps


def test_run_episode_never_reaching_target_is_not_success():
    target = np.array([1.0, 0.0, 0.0])  # far outside what 2mm steps in 5 ticks can reach
    result = run_episode(ToyEnv(target=target), ToyPolicy(target), max_steps=5)
    assert not result.success
    assert result.steps == 5


def test_run_episode_counts_rcm_violations():
    class ViolatingEnv(ToyEnv):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.rcm_deviation_m = 0.01  # always over the default 0.5mm tolerance

    target = np.array([0.01, 0.0, 0.0])
    env = ViolatingEnv(target=target)
    result = run_episode(env, ToyPolicy(target), max_steps=10)
    assert result.rcm_violations == result.steps
    assert result.rcm_violations > 0


def test_run_eval_and_aggregate():
    target = np.array([0.01, 0.0, 0.0])

    def env_factory(seed):
        return ToyEnv(seed=seed, target=target)

    results = run_eval(env_factory, ToyPolicy(target), n_episodes=5, max_steps=50)
    assert len(results) == 5

    summary = aggregate(results)
    assert summary["n_episodes"] == 5
    assert 0.0 <= summary["success_rate"].point_estimate <= 1.0
    assert summary["latency"] is not None
    assert summary["rcm_violation_episode_rate"] == 0.0  # ToyEnv never violates by default
