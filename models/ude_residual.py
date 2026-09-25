"""Universal differential equation (UDE) for cable-drive hysteresis.

A UDE here means: a known physics term (a backlash-with-lag model of the
PSM's cable transmission) plus a learned neural residual that absorbs
whatever the physics term gets wrong. The physics term alone is pure numpy
and needs no GPU/torch — it's the ablation ladder's A2 baseline
("+ UDE residual") minus the network — so it's directly unit-testable.
The residual network is optional and torch-guarded; when torch isn't
installed, `UDEResidual.step` still runs on physics alone (with the
residual contributing zero), so the module degrades rather than breaking.

Physics term — a first-order actuator lag with a backlash (dead-zone) band:
cable slack means small commanded moves don't reach the joint at all, and
once the slack is taken up, the joint chases the command with time
constant `tau`. This is the classic play/backlash operator used for
cable- and gear-driven mechanisms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._torch_optional import TORCH_AVAILABLE, TorchModuleBase, nn


def cable_hysteresis_step(
    q: np.ndarray,
    u: np.ndarray,
    dt: float,
    tau: float = 0.05,
    backlash: float = 0.01,
) -> np.ndarray:
    """One explicit-Euler physics step for N independent joints.

    q: (N,) current joint position. u: (N,) commanded (motor-side) position.
    tau: actuator lag time constant, seconds. backlash: half-width of the
    dead zone, same units as q/u (rad for revolute joints, m for the
    prismatic insertion joint).

    dq/dt = 0                          if |u - q| <= backlash   (cable slack)
    dq/dt = (u - q - backlash*sign(u-q)) / tau   otherwise      (cable taut, lagging)
    """
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    gap = u - q
    slack = np.abs(gap) <= backlash
    dq = np.where(slack, 0.0, (gap - backlash * np.sign(gap)) / tau)
    return q + dq * dt


@dataclass
class UDEParams:
    tau: np.ndarray | float = 0.05
    backlash: np.ndarray | float = 0.01


class UDEResidual(TorchModuleBase):
    """Combines `cable_hysteresis_step` with an optional learned residual:
    `q_{t+1} = physics_step(q_t, u_t) + residual(q_t, u_t)`.

    Constructing this class never requires torch — only calling
    `fit_residual`/using it in gradient-based training does, and at that
    point it raises the same clear `ImportError` as the other model
    classes. `step()` always works (residual = 0 without torch).
    """

    def __init__(self, n_joints: int = 6, params: UDEParams | None = None, hidden_dim: int = 64):
        # Deliberately does NOT call require_torch(): this class is usable
        # in physics-only mode without torch, unlike the other model
        # classes in this package.
        if TORCH_AVAILABLE:
            super().__init__()
        self.n_joints = n_joints
        self.params = params or UDEParams()
        self._residual_net = self._build_residual_net(hidden_dim) if TORCH_AVAILABLE else None

    def _build_residual_net(self, hidden_dim: int):
        return nn.Sequential(
            nn.Linear(2 * self.n_joints, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_joints),
        )

    def step(self, q: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        q_phys = cable_hysteresis_step(q, u, dt, tau=self.params.tau, backlash=self.params.backlash)
        if self._residual_net is None:
            return q_phys

        import torch

        with torch.no_grad():
            x = torch.tensor(np.concatenate([q, u]), dtype=torch.float32).unsqueeze(0)
            residual = self._residual_net(x).squeeze(0).numpy()
        return q_phys + residual
